"""训练数据集：支持多段音频 + 对应文本"""
import logging
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from . import text_processor
from . import audio_processor

logger = logging.getLogger(__name__)


class VoiceDataset(Dataset):
    """多音频训练数据集。

    音频按静音切分为 ~10s 片段，文本按时间比例分配到各片段，
    确保每个片段的 mel 和文本正确对齐。
    """

    def __init__(self, clips: list, text_mode: str = "pinyin",
                 max_segments_per_clip: int = 10):
        """
        Args:
            clips: [{"path": str, "text": str, "emotion": str}, ...]
            text_mode: "pinyin" 或 "char"
            max_segments_per_clip: 每段音频最大切分数
        """
        super().__init__()
        self.text_mode = text_mode
        self.samples = []

        for ci, clip in enumerate(clips):
            audio_path = clip["path"]
            text = clip["text"]

            # 文本编码
            text_seq = text_processor.text_to_sequence(text, text_mode)

            # 音频处理
            try:
                wav = audio_processor.load_audio(audio_path)
            except Exception as e:
                logger.warning(f"[Dataset] 跳过音频 {audio_path}: {e}")
                continue

            sr = audio_processor.SAMPLE_RATE
            total_dur = len(wav) / sr

            # 切分为 ~10s 片段
            segments = audio_processor.split_audio_by_silence(
                wav, sr, min_segment_sec=3.0, max_segment_sec=12.0
            )
            if not segments:
                segments = [wav]

            # 限制段数
            if len(segments) > max_segments_per_clip:
                indices = np.linspace(0, len(segments) - 1, max_segments_per_clip, dtype=int)
                segments = [segments[i] for i in indices]

            # 计算每个片段的时间位置，按比例分配文本
            seg_starts = []
            offset = 0
            for seg in segments:
                seg_starts.append(offset)
                offset += len(seg)
            total_samples = offset

            for si, seg in enumerate(segments):
                if len(seg) < sr * 0.5:
                    continue

                # 按时间比例分配文本
                t_start = seg_starts[si] / total_samples
                t_end = (seg_starts[si] + len(seg)) / total_samples
                char_start = int(t_start * len(text_seq))
                char_end = max(int(t_end * len(text_seq)), char_start + 1)
                char_end = min(char_end, len(text_seq))
                seg_text = text_seq[char_start:char_end]

                if len(seg_text) < 1:
                    seg_text = text_seq[:max(1, len(text_seq) // len(segments))]

                mel = audio_processor.wav_to_mel(seg, sr)
                self.samples.append({
                    "text_seq": torch.LongTensor(seg_text),
                    "mel": torch.FloatTensor(mel),
                    "wav": torch.FloatTensor(seg),
                })

            logger.info(
                f"[Dataset] 音频 {ci+1}: {total_dur:.1f}s -> {len(segments)} 段, "
                f"文本总长: {len(text_seq)}"
            )

        if not self.samples:
            # Fallback: 至少有一个样本
            raise ValueError("没有有效的训练样本，请检查音频文件")

        logger.info(f"[Dataset] 总训练样本: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "text_seq": s["text_seq"],
            "mel": s["mel"],
            "wav": s["wav"],
            "text_len": len(s["text_seq"]),
            "mel_len": s["mel"].shape[1],
        }


def collate_fn(batch):
    """自定义 collate：padding 到同长度。"""
    text_seqs = [item["text_seq"] for item in batch]
    mels = [item["mel"] for item in batch]
    wavs = [item["wav"] for item in batch]
    text_lens = [item["text_len"] for item in batch]
    mel_lens = [item["mel_len"] for item in batch]

    max_text_len = max(text_lens)
    text_padded = torch.zeros(len(batch), max_text_len, dtype=torch.long)
    for i, seq in enumerate(text_seqs):
        text_padded[i, :len(seq)] = seq

    max_mel_len = max(mel_lens)
    n_mels = mels[0].shape[0]
    mel_padded = torch.zeros(len(batch), n_mels, max_mel_len)
    for i, mel in enumerate(mels):
        mel_padded[i, :, :mel.shape[1]] = mel

    max_wav_len = max(len(w) for w in wavs)
    wav_padded = torch.zeros(len(batch), max_wav_len)
    for i, wav in enumerate(wavs):
        wav_padded[i, :len(wav)] = wav

    return {
        "text_seq": text_padded,
        "mel": mel_padded,
        "wav": wav_padded,
        "text_len": torch.LongTensor(text_lens),
        "mel_len": torch.LongTensor(mel_lens),
    }


def create_dataloader(clips: list,
                      text_mode: str = "pinyin",
                      batch_size: int = 1,
                      max_segments_per_clip: int = 10) -> DataLoader:
    """创建训练 DataLoader（多段音频版本）。

    Args:
        clips: [{"path": str, "text": str, "emotion": str}, ...]
        text_mode: 文本编码模式
        batch_size: batch 大小
        max_segments_per_clip: 每段音频最大切分数

    Returns:
        DataLoader
    """
    dataset = VoiceDataset(clips, text_mode, max_segments_per_clip)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        drop_last=False,
    )
