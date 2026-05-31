"""训练编排：支持 FastSpeech2-lite 和 VITS 两种模式（优化版 + 归一化训练）"""
import os
import json
import time
import logging
import threading
import numpy as np
import torch
import torch.nn as nn

from .device import get_device, get_model_device
from .text_processor import text_to_sequence, get_symbol_count
from . import audio_processor
from .dataset import create_dataloader
from .fastspeech2 import FastSpeech2Lite
from .vits_lite import VITSLite
from .augment import SpecAugment
from .vocoder import mel_to_wav, unload_vocoder

logger = logging.getLogger(__name__)

# Mel 归一化常量（与 audio_processor 保持一致）
MEL_MEAN = audio_processor.MEL_MEAN
MEL_STD = audio_processor.MEL_STD

_training_jobs = {}


class TrainingJob:
    """单次训练任务。"""

    def __init__(self, job_id: str, clips: list,
                 model_type: str = "fastspeech2",
                 text_mode: str = "pinyin",
                 voice_name: str = "我的声音",
                 params: dict = None):
        self.job_id = job_id
        self.clips = clips
        self.audio_path = clips[0]["path"] if clips else ""
        self.text = clips[0]["text"] if clips else ""
        self.model_type = model_type
        self.text_mode = text_mode
        self.voice_name = voice_name
        self.params = params or {}

        # 训练参数
        self.epochs = self.params.get("epochs", 300 if model_type == "fastspeech2" else 50)
        self.lr = self.params.get("lr", 0.0002)
        self.batch_size = self.params.get("batch_size", 1)

        # 状态
        self.status = "pending"
        self.current_epoch = 0
        self.current_step = 0
        self.total_steps = 0
        self.loss_history = []
        self.current_loss = {}
        self.error_msg = ""
        self.start_time = 0
        self.device_label = ""

        # 输出目录
        self.output_dir = os.path.join("models", job_id)
        os.makedirs(self.output_dir, exist_ok=True)

        # SSE 事件队列
        self._events = []
        self._event_lock = threading.Lock()
        self._event_ready = threading.Event()

    def add_event(self, data: dict):
        with self._event_lock:
            self._events.append(data)
            self._event_ready.set()

    def get_events(self):
        with self._event_lock:
            events = self._events[:]
            self._events.clear()
            self._event_ready.clear()
        return events

    def wait_for_event(self, timeout=1.0):
        self._event_ready.wait(timeout)

    def _notify(self):
        elapsed = time.time() - self.start_time if self.start_time else 0
        remaining = 0
        if self.current_step > 0 and self.total_steps > 0:
            remaining = elapsed / self.current_step * (self.total_steps - self.current_step)

        self.add_event({
            "type": "progress",
            "status": self.status,
            "epoch": self.current_epoch,
            "total_epochs": self.epochs,
            "step": self.current_step,
            "total_steps": self.total_steps,
            "loss": self.current_loss,
            "loss_history": self.loss_history[-200:],
            "elapsed": round(elapsed, 1),
            "remaining": round(remaining, 1),
            "device": self.device_label,
            "progress_pct": round(self.current_step / max(self.total_steps, 1) * 100, 1),
        })

    def run(self):
        """执行训练（在子线程中调用）。"""
        try:
            self.status = "training"
            self.start_time = time.time()
            device, device_label = get_device()
            self.device_label = device_label

            logger.info(f"[Train] 开始训练 job={self.job_id}, model={self.model_type}, device={device_label}")
            self._notify()

            if self.model_type == "fastspeech2":
                self._train_fastspeech2(device)
            else:
                self._train_vits(device)

            # 导出 ONNX（AMD 下导出需用 CPU）
            self.status = "exporting"
            self._notify()
            import gc
            gc.collect()
            unload_vocoder()

            export_device, _ = get_model_device(for_export=True)

            from .export_onnx import export_model
            export_model(
                model_type=self.model_type,
                checkpoint_path=os.path.join(self.output_dir, "checkpoint.pt"),
                output_dir=self.output_dir,
                voice_name=self.voice_name,
                reference_text=self.clips[0]["text"] if self.clips else "",
                audio_path=self.clips[0]["path"] if self.clips else "",
                text_mode=self.text_mode,
                device=export_device,
            )

            self.status = "done"
            elapsed = time.time() - self.start_time
            self.add_event({
                "type": "complete",
                "status": "done",
                "elapsed": round(elapsed, 1),
                "output_dir": self.output_dir,
                "message": f"训练完成！耗时 {elapsed/60:.1f} 分钟",
            })
            logger.info(f"[Train] 训练完成: {self.job_id}, 耗时 {elapsed:.0f}s")

        except Exception as e:
            self.status = "error"
            self.error_msg = str(e)
            self.add_event({
                "type": "error",
                "status": "error",
                "message": f"训练失败: {e}",
            })
            logger.error(f"[Train] 训练失败: {self.job_id}: {e}", exc_info=True)

    def _train_fastspeech2(self, device):
        """FastSpeech2-lite 训练（mel 归一化 + 梯度裁剪 + Early Stopping）。"""
        vocab_size = get_symbol_count(self.text_mode)
        model = FastSpeech2Lite(vocab_size).to(device)

        dataloader = create_dataloader(self.clips, self.text_mode, self.batch_size)

        optimizer = torch.optim.Adam(
            model.parameters(), lr=self.lr,
            betas=(0.9, 0.98), eps=1e-9,
            weight_decay=1e-4,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)

        mel_criterion = nn.MSELoss()
        duration_criterion = nn.MSELoss()
        spec_aug = SpecAugment(freq_mask_param=10, time_mask_param=20, p=0.5)

        total_samples = len(dataloader.dataset)
        self.total_steps = self.epochs * total_samples

        logger.info(f"[FastSpeech2] 数据集: {total_samples} 样本, batch={self.batch_size}, 总步数: {self.total_steps}")

        best_loss = float('inf')
        patience = 20
        patience_counter = 0

        for epoch in range(self.epochs):
            model.train()
            epoch_loss = 0

            for batch in dataloader:
                text_seq = batch["text_seq"].to(device)
                mel_target = batch["mel"].to(device)        # 已归一化
                text_len = batch["text_len"].to(device)
                mel_len = batch["mel_len"].to(device)

                batch_size = text_seq.size(0)

                # SpecAugment
                mel_target_aug = torch.stack([spec_aug(m) for m in mel_target])

                # 真实 duration（来自 Whisper 对齐）
                durations_gt = batch["durations"].to(device)
                dur_targets_log = torch.log(durations_gt + 1.0)

                # Duration 精度补偿：确保 LengthRegulator 输出长度与 mel 匹配
                durations_rounded = durations_gt.round().long()
                for b in range(batch_size):
                    current = durations_rounded[b].sum().item()
                    target = int(mel_len[b].item())
                    diff = target - current
                    if diff != 0 and text_len[b] > 2:
                        valid_len = int(text_len[b].item())
                        idx = durations_rounded[b, :valid_len].argmax()
                        durations_rounded[b, idx] = max(durations_rounded[b, idx] + diff, 1)

                # Pitch / Energy（简化：零填充）
                T_mel = mel_target_aug.size(2)
                pitches = torch.zeros(batch_size, T_mel, device=device)
                energies = torch.zeros(batch_size, T_mel, device=device)

                # Forward
                output = model(text_seq, durations_rounded.float(), pitches, energies)

                # Loss（目标已归一化，loss 在归一化空间计算）
                T_pred = output["mel_postnet"].size(1)
                T_gt = mel_target_aug.size(2)
                T_min = min(T_pred, T_gt)

                mel_loss = mel_criterion(
                    output["mel_postnet"][:, :T_min, :].transpose(1, 2),
                    mel_target_aug[:, :, :T_min]
                )
                mel_loss_pre = mel_criterion(
                    output["mel_pred"][:, :T_min, :].transpose(1, 2),
                    mel_target_aug[:, :, :T_min]
                )

                # Duration loss (log-normalized)
                dur_pred = output["duration_pred"].squeeze(-1)
                T_dur = min(dur_pred.size(1), dur_targets_log.size(1))
                dur_loss = duration_criterion(dur_pred[:, :T_dur], dur_targets_log[:, :T_dur])

                loss = mel_loss + mel_loss_pre + dur_loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                self.current_step += 1
                epoch_loss += loss.item()

                self.current_loss = {
                    "mel": round(mel_loss.item(), 6),
                    "mel_pre": round(mel_loss_pre.item(), 6),
                    "duration": round(dur_loss.item(), 6),
                    "total": round(loss.item(), 6),
                }

            scheduler.step()
            avg_loss = epoch_loss / max(len(dataloader), 1)
            self.current_epoch = epoch + 1
            self.loss_history.append({
                "epoch": epoch + 1,
                "loss": round(avg_loss, 6),
            })

            # Early Stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                self._save_checkpoint(model, optimizer, epoch + 1, is_best=True)
            else:
                patience_counter += 1

            if (epoch + 1) % 5 == 0 or epoch == 0:
                self._notify()
                logger.info(f"[FastSpeech2] Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.6f}, Best: {best_loss:.6f}")

            if patience_counter >= patience:
                logger.info(f"[FastSpeech2] Early stopping at epoch {epoch+1}")
                break

            if (epoch + 1) % 10 == 0:
                self._save_checkpoint(model, optimizer, epoch + 1)

        # === HiFi-GAN Vocoder 训练 ===
        logger.info("[HiFi-GAN] 开始训练声码器...")
        from .hifigan_lite import HiFiGANGenerator
        generator = HiFiGANGenerator().to(device)
        g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.9))

        hifigan_epochs = min(100, self.epochs)
        for epoch in range(hifigan_epochs):
            generator.train()
            epoch_loss = 0
            for batch in dataloader:
                mel = batch["mel"].to(device)
                wav = batch["wav"].to(device)

                # 反归一化 mel → log-mel
                mel_denorm = mel * MEL_STD + MEL_MEAN

                wav_pred = generator(mel_denorm)

                min_len = min(wav_pred.size(2), wav.size(1))
                loss = torch.nn.functional.l1_loss(
                    wav_pred.squeeze(1)[:, :min_len],
                    wav[:, :min_len]
                )

                g_optimizer.zero_grad()
                loss.backward()
                g_optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / max(len(dataloader), 1)
            if (epoch + 1) % 10 == 0 or epoch == 0:
                logger.info(f"[HiFi-GAN] Epoch {epoch+1}/{hifigan_epochs}, Loss: {avg_loss:.6f}")

        # 最终 checkpoint 包含 hifigan 权重
        self._save_checkpoint(model, optimizer, self.current_epoch, hifigan=generator)

    def _train_vits(self, device):
        """VITS-lite 训练（mel 归一化 + KL 散度 + Early Stopping）。"""
        vocab_size = get_symbol_count(self.text_mode)
        model = VITSLite(vocab_size).to(device)

        dataloader = create_dataloader(self.clips, self.text_mode, self.batch_size)

        optimizer = torch.optim.Adam(
            model.parameters(), lr=self.lr,
            betas=(0.9, 0.98), eps=1e-9,
            weight_decay=1e-4,
        )

        total_samples = len(dataloader.dataset)
        self.total_steps = self.epochs * total_samples

        logger.info(f"[VITS] 数据集: {total_samples} 样本, batch={self.batch_size}, 总步数: {self.total_steps}")

        best_loss = float('inf')
        patience = 20
        patience_counter = 0

        for epoch in range(self.epochs):
            model.train()
            epoch_loss = 0

            for batch in dataloader:
                text_seq = batch["text_seq"].to(device)
                mel_target = batch["mel"].to(device)      # 已归一化
                text_len = batch["text_len"].to(device)
                mel_len = batch["mel_len"].to(device)

                batch_size = text_seq.size(0)

                # 真实 duration
                durations_gt = batch["durations"].to(device)
                durations_rounded = durations_gt.round().long()
                for b in range(batch_size):
                    current = durations_rounded[b].sum().item()
                    target = int(mel_len[b].item())
                    diff = target - current
                    if diff != 0 and text_len[b] > 2:
                        valid_len = int(text_len[b].item())
                        idx = durations_rounded[b, :valid_len].argmax()
                        durations_rounded[b, idx] = max(durations_rounded[b, idx] + diff, 1)

                # Generator forward
                output = model(text_seq, mel_target, durations_rounded.float())
                mel_pred = output["mel_pred"]

                # Mel loss（归一化空间）
                T_mel_min = min(mel_pred.size(1), mel_target.size(2))
                mel_loss = nn.functional.mse_loss(
                    mel_pred[:, :T_mel_min, :].transpose(1, 2),
                    mel_target[:, :, :T_mel_min]
                )

                # KL divergence
                kl_loss = torch.tensor(0.0, device=device)
                if output["z_mean"] is not None and output["q_mean"] is not None:
                    z_mean = output["z_mean"].transpose(1, 2)
                    z_logvar = output["z_logvar"].transpose(1, 2)
                    q_mean = output["q_mean"]
                    q_logvar = output["q_logvar"]
                    T_kl = min(z_mean.size(2), q_mean.size(2))
                    kl_loss = 0.5 * torch.mean(
                        q_logvar[:, :, :T_kl] - z_logvar[:, :, :T_kl] +
                        (torch.exp(z_logvar[:, :, :T_kl]) + (z_mean[:, :, :T_kl] - q_mean[:, :, :T_kl]) ** 2) /
                        torch.exp(q_logvar[:, :, :T_kl]) - 1
                    )

                loss = mel_loss * 45 + kl_loss * 1.0

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                self.current_step += 1
                epoch_loss += loss.item()

                self.current_loss = {
                    "mel": round(mel_loss.item(), 6),
                    "kl": round(kl_loss.item(), 6) if isinstance(kl_loss, torch.Tensor) else 0,
                    "total": round(loss.item(), 6),
                }

            avg_loss = epoch_loss / max(len(dataloader), 1)
            self.current_epoch = epoch + 1
            self.loss_history.append({
                "epoch": epoch + 1,
                "loss": round(avg_loss, 6),
            })

            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                self._save_checkpoint_vits(model, optimizer, epoch + 1, is_best=True)
            else:
                patience_counter += 1

            if (epoch + 1) % 5 == 0 or epoch == 0:
                self._notify()
                logger.info(f"[VITS] Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.6f}, Best: {best_loss:.6f}")

            if patience_counter >= patience:
                logger.info(f"[VITS] Early stopping at epoch {epoch+1}")
                break

            if (epoch + 1) % 10 == 0:
                self._save_checkpoint_vits(model, optimizer, epoch + 1)

        # === HiFi-GAN Vocoder 训练 ===
        logger.info("[HiFi-GAN] 开始训练声码器...")
        from .hifigan_lite import HiFiGANGenerator
        generator = HiFiGANGenerator().to(device)
        g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.9))

        hifigan_epochs = min(100, self.epochs)
        for epoch in range(hifigan_epochs):
            generator.train()
            epoch_loss = 0
            for batch in dataloader:
                mel = batch["mel"].to(device)
                wav = batch["wav"].to(device)

                mel_denorm = mel * MEL_STD + MEL_MEAN
                wav_pred = generator(mel_denorm)

                min_len = min(wav_pred.size(2), wav.size(1))
                loss = torch.nn.functional.l1_loss(
                    wav_pred.squeeze(1)[:, :min_len],
                    wav[:, :min_len]
                )

                g_optimizer.zero_grad()
                loss.backward()
                g_optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / max(len(dataloader), 1)
            if (epoch + 1) % 10 == 0 or epoch == 0:
                logger.info(f"[HiFi-GAN] Epoch {epoch+1}/{hifigan_epochs}, Loss: {avg_loss:.6f}")

        self._save_checkpoint_vits(model, optimizer, self.current_epoch, hifigan=generator)

    def _save_checkpoint(self, model, optimizer, epoch, is_best=False, hifigan=None):
        path = os.path.join(self.output_dir, "checkpoint.pt")
        data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_type": self.model_type,
            "text_mode": self.text_mode,
            "vocab_size": model.embedding.num_embeddings,
        }
        if hifigan is not None:
            data["hifigan_state_dict"] = hifigan.state_dict()
        torch.save(data, path)
        tag = " (best)" if is_best else ""
        logger.info(f"[Train] Checkpoint saved{tag}: {path}")

    def _save_checkpoint_vits(self, model, optimizer, epoch, is_best=False, hifigan=None):
        path = os.path.join(self.output_dir, "checkpoint.pt")
        data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_type": "vits",
            "text_mode": self.text_mode,
            "vocab_size": model.text_encoder.embedding.num_embeddings,
        }
        if hifigan is not None:
            data["hifigan_state_dict"] = hifigan.state_dict()
        torch.save(data, path)
        tag = " (best)" if is_best else ""
        logger.info(f"[Train] Checkpoint saved{tag}: {path}")


def start_training(job_id: str, clips: list,
                   model_type: str = "fastspeech2",
                   text_mode: str = "pinyin",
                   voice_name: str = "我的声音",
                   params: dict = None) -> TrainingJob:
    """启动训练任务。"""
    job = TrainingJob(
        job_id=job_id,
        clips=clips,
        model_type=model_type,
        text_mode=text_mode,
        voice_name=voice_name,
        params=params,
    )
    _training_jobs[job_id] = job

    thread = threading.Thread(target=job.run, daemon=True)
    thread.start()

    logger.info(f"[Train] 任务已启动: {job_id}")
    return job


def get_job(job_id: str) -> TrainingJob:
    """获取训练任务。"""
    return _training_jobs.get(job_id)


def get_all_jobs() -> dict:
    """获取所有训练任务。"""
    return _training_jobs
