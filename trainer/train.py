"""训练编排：支持 FastSpeech2-lite 和 VITS 两种模式"""
import os
import json
import time
import logging
import threading
import numpy as np
import torch
import torch.nn as nn

from .device import get_device
from .text_processor import text_to_sequence, get_symbol_count, estimate_duration_from_text
from . import audio_processor
from .dataset import create_dataloader
from .fastspeech2 import FastSpeech2Lite
from .vits_lite import VITSLite
from .hifigan_lite import HiFiGANGenerator, HiFiGANDiscriminator

logger = logging.getLogger(__name__)

# 训练状态存储
_training_jobs = {}


class TrainingJob:
    """单次训练任务。"""

    def __init__(self, job_id: str, clips: list,
                 model_type: str = "fastspeech2",
                 text_mode: str = "pinyin",
                 voice_name: str = "我的声音",
                 params: dict = None):
        self.job_id = job_id
        self.clips = clips  # [{"path": ..., "text": ..., "emotion": ...}, ...]
        # 兼容旧接口：用第一段音频作为主音频
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
        self.status = "pending"  # pending / training / exporting / done / error
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
        """添加 SSE 事件。"""
        with self._event_lock:
            self._events.append(data)
            self._event_ready.set()

    def get_events(self):
        """获取并清空事件队列。"""
        with self._event_lock:
            events = self._events[:]
            self._events.clear()
            self._event_ready.clear()
        return events

    def wait_for_event(self, timeout=1.0):
        """等待新事件。"""
        self._event_ready.wait(timeout)

    def _notify(self):
        """推送当前状态到 SSE。"""
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
            "loss_history": self.loss_history[-200:],  # 最近 200 个点
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
                # Train HiFi-GAN vocoder on reference audio
                self._train_hifigan(device)
            else:
                self._train_vits(device)

            # 导出 ONNX
            self.status = "exporting"
            self._notify()
            import gc
            gc.collect()

            from .export_onnx import export_model
            export_model(
                model_type=self.model_type,
                checkpoint_path=os.path.join(self.output_dir, "checkpoint.pt"),
                output_dir=self.output_dir,
                voice_name=self.voice_name,
                reference_text=self.clips[0]["text"] if self.clips else "",
                audio_path=self.clips[0]["path"] if self.clips else "",
                text_mode=self.text_mode,
                device=device,
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
        """FastSpeech2-lite 训练。"""
        from .fastspeech2 import FastSpeech2Lite

        vocab_size = get_symbol_count(self.text_mode)
        model = FastSpeech2Lite(vocab_size).to(device)

        # 创建数据集（多段音频）
        dataloader = create_dataloader(
            self.clips, self.text_mode, self.batch_size
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, betas=(0.9, 0.98), eps=1e-9)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)

        # 损失函数
        mel_criterion = nn.MSELoss()
        duration_criterion = nn.MSELoss()

        total_samples = len(dataloader.dataset)
        self.total_steps = self.epochs * total_samples

        logger.info(f"[FastSpeech2] 数据集大小: {total_samples}, 总步数: {self.total_steps}")

        for epoch in range(self.epochs):
            model.train()
            epoch_loss = 0

            for batch in dataloader:
                text_seq = batch["text_seq"].to(device)
                mel_target = batch["mel"].to(device)  # (B, n_mels, T_mel)
                text_len = batch["text_len"].to(device)
                mel_len = batch["mel_len"].to(device)

                # 估计 duration（简化方案），使用 log 归一化
                batch_size = text_seq.size(0)
                T_text = text_seq.size(1)
                T_mel = mel_target.size(2)

                durations = torch.zeros(batch_size, T_text, device=device)
                dur_targets_log = torch.zeros(batch_size, T_text, device=device)
                for b in range(batch_size):
                    t_len = int(text_len[b].item())
                    m_len = int(mel_len[b].item())
                    if t_len > 0 and m_len > 0:
                        base_dur = m_len / t_len
                        for t in range(t_len):
                            durations[b, t] = base_dur
                            dur_targets_log[b, t] = torch.log(torch.tensor(base_dur) + 1.0)

                # Pitch（简化：用 mel 均值代替）
                pitches = torch.zeros(batch_size, T_mel, device=device)
                energies = torch.zeros(batch_size, T_mel, device=device)

                # Forward
                output = model(text_seq, durations, pitches, energies)

                # Loss
                T_pred = output["mel_postnet"].size(1)
                T_gt = mel_target.size(2)
                T_min = min(T_pred, T_gt)

                mel_loss = mel_criterion(
                    output["mel_postnet"][:, :T_min, :].transpose(1, 2),
                    mel_target[:, :, :T_min]
                )
                mel_loss_pre = mel_criterion(
                    output["mel_pred"][:, :T_min, :].transpose(1, 2),
                    mel_target[:, :, :T_min]
                )

                # Duration loss (log-normalized, higher weight)
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

            if (epoch + 1) % 5 == 0 or epoch == 0:
                self._notify()
                logger.info(f"[FastSpeech2] Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.6f}")

            # 保存 checkpoint
            if (epoch + 1) % 10 == 0:
                self._save_checkpoint(model, optimizer, epoch + 1)

        # 最终保存
        self._save_checkpoint(model, optimizer, self.epochs)

    def _train_vits(self, device):
        """VITS-lite 训练。"""
        vocab_size = get_symbol_count(self.text_mode)
        model = VITSLite(vocab_size).to(device)
        hifigan = HiFiGANGenerator().to(device)
        discriminator = HiFiGANDiscriminator().to(device)

        dataloader = create_dataloader(
            self.clips, self.text_mode, self.batch_size
        )

        optimizer_g = torch.optim.Adam(
            list(model.parameters()) + list(hifigan.parameters()),
            lr=self.lr, betas=(0.9, 0.98), eps=1e-9
        )
        optimizer_d = torch.optim.Adam(
            discriminator.parameters(),
            lr=self.lr, betas=(0.9, 0.98), eps=1e-9
        )

        total_samples = len(dataloader.dataset)
        self.total_steps = self.epochs * total_samples

        logger.info(f"[VITS] 数据集大小: {total_samples}, 总步数: {self.total_steps}")

        for epoch in range(self.epochs):
            model.train()
            hifigan.train()
            discriminator.train()
            epoch_loss = 0

            for batch in dataloader:
                text_seq = batch["text_seq"].to(device)
                mel_target = batch["mel"].to(device)  # (B, n_mels, T_mel)
                wav_target = batch["wav"].to(device)
                text_len = batch["text_len"].to(device)
                mel_len = batch["mel_len"].to(device)

                batch_size = text_seq.size(0)
                T_text = text_seq.size(1)

                # 估计 duration
                durations = torch.zeros(batch_size, T_text, device=device)
                for b in range(batch_size):
                    t_len = int(text_len[b].item())
                    m_len = int(mel_len[b].item())
                    if t_len > 0 and m_len > 0:
                        base_dur = m_len / t_len
                        for t in range(t_len):
                            durations[b, t] = base_dur

                # ---- Generator forward ----
                output = model(text_seq, mel_target, durations)
                mel_pred = output["mel_pred"]  # (B, T_mel, n_mels)

                # 通过 HiFi-GAN 生成波形
                T_mel_pred = mel_pred.size(1)
                wav_pred = hifigan(mel_pred.transpose(1, 2))  # (B, 1, T_wav)

                # 裁剪到目标长度
                T_wav_pred = wav_pred.size(2)
                T_wav_gt = wav_target.size(1)
                T_wav_min = min(T_wav_pred, T_wav_gt)

                # ---- Discriminator ----
                wav_real = wav_target.unsqueeze(1) if wav_target.dim() == 2 else wav_target
                wav_fake = wav_pred[:, :, :T_wav_min]
                wav_real = wav_real[:, :, :T_wav_min]

                mpd_r, mpd_f, msd_r, msd_f, mpd_fr, mpd_ff, msd_fr, msd_ff = discriminator(wav_real, wav_fake)

                # D loss
                loss_d = 0
                for dr, df in zip(mpd_r, mpd_f):
                    loss_d += torch.mean((1 - dr) ** 2) + torch.mean(df ** 2)
                for dr, df in zip(msd_r, msd_f):
                    loss_d += torch.mean((1 - dr) ** 2) + torch.mean(df ** 2)

                optimizer_d.zero_grad()
                loss_d.backward()
                optimizer_d.step()

                # ---- Generator loss ----
                # Single D forward pass: real (detached) + fake (gradients to G)
                mpd_r_g, mpd_f_g, msd_r_g, msd_f_g, mpd_fr_g, mpd_ff_g, msd_fr_g, msd_ff_g = discriminator(wav_real.detach(), wav_fake)

                loss_g_gan = 0
                for df in mpd_f_g:
                    loss_g_gan += torch.mean((1 - df) ** 2)
                for df in msd_f_g:
                    loss_g_gan += torch.mean((1 - df) ** 2)

                # Feature matching loss
                loss_fm = 0
                for fr, ff in zip(mpd_fr_g, mpd_ff_g):
                    for r, f in zip(fr, ff):
                        loss_fm += torch.mean(torch.abs(r - f))
                for fr, ff in zip(msd_fr_g, msd_ff_g):
                    for r, f in zip(fr, ff):
                        loss_fm += torch.mean(torch.abs(r - f))

                # Mel loss
                T_mel_min = min(mel_pred.size(1), mel_target.size(2))
                mel_loss = nn.functional.mse_loss(
                    mel_pred[:, :T_mel_min, :].transpose(1, 2),
                    mel_target[:, :, :T_mel_min]
                )

                # KL divergence
                kl_loss = 0
                if output["z_mean"] is not None and output["q_mean"] is not None:
                    z_mean = output["z_mean"].transpose(1, 2)  # (B, D, T)
                    z_logvar = output["z_logvar"].transpose(1, 2)
                    q_mean = output["q_mean"]
                    q_logvar = output["q_logvar"]
                    T_kl = min(z_mean.size(2), q_mean.size(2))
                    kl_loss = 0.5 * torch.mean(
                        q_logvar[:, :, :T_kl] - z_logvar[:, :, :T_kl] +
                        (torch.exp(z_logvar[:, :, :T_kl]) + (z_mean[:, :, :T_kl] - q_mean[:, :, :T_kl]) ** 2) /
                        torch.exp(q_logvar[:, :, :T_kl]) - 1
                    )

                loss_g = loss_g_gan + loss_fm * 2 + mel_loss * 45 + kl_loss * 1.0

                optimizer_g.zero_grad()
                loss_g.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer_g.step()

                self.current_step += 1
                epoch_loss += loss_g.item()

                self.current_loss = {
                    "mel": round(mel_loss.item(), 6),
                    "kl": round(kl_loss.item(), 6) if isinstance(kl_loss, torch.Tensor) else 0,
                    "gan_g": round(loss_g_gan.item(), 6),
                    "gan_d": round(loss_d.item(), 6),
                    "fm": round(loss_fm.item(), 6),
                    "total": round(loss_g.item(), 6),
                }

            avg_loss = epoch_loss / max(len(dataloader), 1)
            self.current_epoch = epoch + 1
            self.loss_history.append({
                "epoch": epoch + 1,
                "loss": round(avg_loss, 6),
            })

            if (epoch + 1) % 5 == 0 or epoch == 0:
                self._notify()
                logger.info(f"[VITS] Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.6f}")

            if (epoch + 1) % 10 == 0:
                self._save_checkpoint_vits(model, hifigan, optimizer_g, epoch + 1)

        self._save_checkpoint_vits(model, hifigan, optimizer_g, self.epochs)


    def _train_hifigan(self, device):
        """HiFi-GAN vocoder 训练：用参考音频的 mel 训练声码器。"""
        from .hifigan_lite import HiFiGANGenerator, HiFiGANDiscriminator
        from . import audio_processor

        logger.info("[HiFi-GAN] 开始训练声码器...")

        # 加载参考音频
        audio_path = self.clips[0]["path"] if self.clips else ""
        if not audio_path or not os.path.exists(audio_path):
            logger.warning("[HiFi-GAN] 无参考音频，跳过")
            return

        wav_full = audio_processor.load_audio(audio_path, sr=audio_processor.SAMPLE_RATE)
        mel_full = audio_processor.wav_to_mel(wav_full, audio_processor.SAMPLE_RATE)

        # 切分为 1 秒片段训练
        sr = audio_processor.SAMPLE_RATE
        hop = audio_processor.HOP_LENGTH
        chunk_mel = 100  # ~1s
        chunks_mel = []
        chunks_wav = []
        for i in range(0, mel_full.shape[1] - chunk_mel, chunk_mel // 2):
            chunks_mel.append(mel_full[:, i:i + chunk_mel])
            wav_start = i * hop
            wav_end = (i + chunk_mel) * hop
            if wav_end <= len(wav_full):
                chunks_wav.append(wav_full[wav_start:wav_end])

        if not chunks_mel:
            logger.warning("[HiFi-GAN] 音频太短，跳过")
            return

        generator = HiFiGANGenerator().to(device)
        discriminator = HiFiGANDiscriminator().to(device)

        opt_g = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.8, 0.99))
        opt_d = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.8, 0.99))

        n_steps = min(100, len(chunks_mel) * 20)
        logger.info(f"[HiFi-GAN] {len(chunks_mel)} 个片段, {n_steps} 步")

        for step in range(n_steps):
            idx = step % len(chunks_mel)
            mel_in = torch.FloatTensor(chunks_mel[idx]).unsqueeze(0).to(device)
            wav_gt = torch.FloatTensor(chunks_wav[idx]).unsqueeze(0).unsqueeze(0).to(device)

            # Generator
            wav_pred = generator(mel_in)

            # Discriminator (detach fake to avoid graph reuse)
            T_min = min(wav_pred.size(2), wav_gt.size(2))
            mpd_r, mpd_f, msd_r, msd_f, _, _, _, _ = discriminator(
                wav_gt[:, :, :T_min], wav_pred[:, :, :T_min].detach())

            # D loss
            loss_d = 0
            for dr, df in zip(mpd_r, mpd_f):
                loss_d += torch.mean((1 - dr) ** 2) + torch.mean(df ** 2)
            for dr, df in zip(msd_r, msd_f):
                loss_d += torch.mean((1 - dr) ** 2) + torch.mean(df ** 2)
            opt_d.zero_grad()
            loss_d.backward()
            opt_d.step()

            # G loss (fresh D forward with gradients through wav_pred)
            mpd_r_g, mpd_f_g, msd_r_g, msd_f_g, mpd_fr_g, mpd_ff_g, msd_fr_g, msd_ff_g = discriminator(
                wav_gt[:, :, :T_min].detach(), wav_pred[:, :, :T_min])
            loss_g_gan = sum(torch.mean((1 - df) ** 2) for df in mpd_f_g)
            loss_g_gan += sum(torch.mean((1 - df) ** 2) for df in msd_f_g)
            loss_fm = 0
            for fr, ff in zip(mpd_fr_g, mpd_ff_g):
                for r, f in zip(fr, ff):
                    loss_fm += torch.mean(torch.abs(r - f))
            for fr, ff in zip(msd_fr_g, msd_ff_g):
                for r, f in zip(fr, ff):
                    loss_fm += torch.mean(torch.abs(r - f))
            loss_mel = torch.nn.functional.mse_loss(wav_pred[:, :, :T_min], wav_gt[:, :, :T_min])
            loss_g = loss_g_gan + loss_fm * 2 + loss_mel * 45
            opt_g.zero_grad()
            loss_g.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
            opt_g.step()

            if (step + 1) % 20 == 0 or step == 0:
                logger.info(f"[HiFi-GAN] Step {step+1}/{n_steps}, D={loss_d.item():.4f}, G={loss_g.item():.4f}")
                pct = round((step + 1) / n_steps * 100, 1)
                self.add_event({
                    "type": "progress", "status": "vocoder",
                    "epoch": self.epochs, "total_epochs": self.epochs,
                    "step": self.total_steps, "total_steps": self.total_steps,
                    "loss": {"vocoder": round(loss_g.item(), 4)},
                    "loss_history": self.loss_history[-200:],
                    "progress_pct": 100,
                    "device": self.device_label,
                    "vocoder_pct": pct,
                })

        # Save
        hifigan_path = os.path.join(self.output_dir, "hifigan.pt")
        torch.save(generator.state_dict(), hifigan_path)
        self._hifigan_state = generator.state_dict()
        logger.info(f"[HiFi-GAN] 训练完成, 保存: {hifigan_path}")

    def _save_checkpoint(self, model, optimizer, epoch):
        path = os.path.join(self.output_dir, "checkpoint.pt")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_type": self.model_type,
            "text_mode": self.text_mode,
            "vocab_size": model.embedding.num_embeddings,
        }, path)
        logger.info(f"[Train] Checkpoint saved: {path}")

    def _save_checkpoint_vits(self, model, hifigan, optimizer, epoch):
        path = os.path.join(self.output_dir, "checkpoint.pt")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "hifigan_state_dict": hifigan.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_type": "vits",
            "text_mode": self.text_mode,
            "vocab_size": model.text_encoder.embedding.num_embeddings,
        }, path)
        logger.info(f"[Train] Checkpoint saved: {path}")


def start_training(job_id: str, clips: list,
                   model_type: str = "fastspeech2",
                   text_mode: str = "pinyin",
                   voice_name: str = "我的声音",
                   params: dict = None) -> TrainingJob:
    """启动训练任务。

    Args:
        job_id: 唯一标识
        clips: [{"path": str, "text": str, "emotion": str}, ...]
        model_type: "fastspeech2" 或 "vits"
        text_mode: "pinyin" 或 "char"
        voice_name: 音色名称
        params: 额外训练参数

    Returns:
        TrainingJob 实例
    """
    job = TrainingJob(
        job_id=job_id,
        clips=clips,
        model_type=model_type,
        text_mode=text_mode,
        voice_name=voice_name,
        params=params,
    )
    _training_jobs[job_id] = job

    # 后台线程执行
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


