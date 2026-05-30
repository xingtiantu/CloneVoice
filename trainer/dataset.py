"""训练数据集：基于 Whisper 强制对齐获取真实 Duration。

支持自动切分超长音频（> 20 秒）为短段，避免模型自注意力 OOM。
"""
import logging
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from . import text_processor
from . import audio_processor
from . import aligner

logger = logging.getLogger(__name__)

# 每段最长 mel 帧数（对应约 20 秒音频 @ 22050Hz, hop=256）
MAX_SEGMENT_FRAMES = 1720
HOP = audio_processor.HOP_LENGTH
SR = audio_processor.SAMPLE_RATE


def _count_phonemes_per_char(text: str, text_mode: str) -> list:
    """估算每个字对应的音素数量（用于 duration 分配）。"""
    if text_mode == "char":
        return [1] * len([c for c in text if c.strip()])

    try:
        from pypinyin import pinyin, Style
    except ImportError:
        return [3] * len(text)  # 粗略估计

    py_result = pinyin(text, style=Style.TONE3, neutral_tone_with_five=True)
    counts = []
    for item in py_result:
        syllable = item[0]
        if not syllable or syllable.strip() == "":
            continue
        if syllable in text_processor.PUNCTUATIONS:
            counts.append(1)
            continue

        body = syllable
        if body and body[-1].isdigit():
            body = body[:-1]
        body = body.replace("ü", "v")
        body = text_processor._PINYIN_REPLACE.get(body, body)

        initial, final = text_processor._split_pinyin(body)
        count = 0
        if initial and initial in text_processor.PINYIN_SYMBOL_TO_ID:
            count += 1
        if final:
            count += 1  # final 或 fallback 到 <unk>
        count += 1  # tone
        counts.append(max(count, 1))
    return counts


class VoiceDataset(Dataset):
    """基于强制对齐的多音频训练数据集。

    关键改进：
    1. 不再按静音切分音频（避免边界破坏字级对齐）。
    2. 使用 Whisper 字级时间戳建立精确音频-文本映射。
    3. 计算每个音素的真实 duration（mel 帧数），取代粗暴的平均值估算。
    """

    def __init__(self, clips: list, text_mode: str = "pinyin"):
        """
        Args:
            clips: [{"path": str, "text": str, "emotion": str}, ...]
            text_mode: "pinyin" 或 "char"
        """
        super().__init__()
        self.text_mode = text_mode
        self.samples = []

        for ci, clip in enumerate(clips):
            audio_path = clip["path"]
            text = clip.get("text", "").strip()

            if not text:
                logger.warning(f"[Dataset] 跳过无文本音频: {audio_path}")
                continue

            # 完整文本编码
            text_seq = text_processor.text_to_sequence(text, text_mode)
            if len(text_seq) <= 2:
                logger.warning(f"[Dataset] 文本太短，跳过: {audio_path}")
                continue

            # 加载音频
            try:
                wav = audio_processor.load_audio(audio_path)
            except Exception as e:
                logger.warning(f"[Dataset] 跳过音频 {audio_path}: {e}")
                continue

            sr = audio_processor.SAMPLE_RATE
            total_dur = len(wav) / sr
            if total_dur < 1.0:
                logger.warning(f"[Dataset] 音频太短 ({total_dur:.1f}s)，跳过: {audio_path}")
                continue

            # 计算 mel 谱（获取真实 mel 长度），训练时归一化到零附近
            mel = audio_processor.wav_to_mel(wav, sr)
            mel = audio_processor.normalize_mel(mel)  # 归一化提升训练稳定性
            mel_len = mel.shape[1]

            # 强制对齐
            try:
                alignment = aligner.align_audio_text(audio_path, text)
            except Exception as e:
                logger.warning(f"[Dataset] 对齐失败，回退到时间比例: {audio_path} ({e})")
                alignment = []

            # 超长音频自动切分为短段（避免解码器自注意力 OOM）
            segments = self._split_long_clip(wav, sr, alignment, text, text_mode)
            if segments:
                logger.info(f"[Dataset] 音频 {ci+1} 较长 ({total_dur:.1f}s)，切分为 {len(segments)} 段")
                for seg_idx, (seg_wav, seg_text, seg_alignment) in enumerate(segments):
                    seg_mel = audio_processor.wav_to_mel(seg_wav, sr)
                    seg_mel = audio_processor.normalize_mel(seg_mel)
                    seg_mel_len = seg_mel.shape[1]
                    seg_text_seq = text_processor.text_to_sequence(seg_text, text_mode)
                    if len(seg_text_seq) <= 2:
                        continue
                    seg_durations = self._compute_durations(
                        len(seg_text_seq), seg_alignment, seg_text, text_mode, seg_mel_len
                    )
                    self.samples.append({
                        "text_seq": torch.LongTensor(seg_text_seq),
                        "mel": torch.FloatTensor(seg_mel),
                        "wav": torch.FloatTensor(seg_wav),
                        "durations": torch.FloatTensor(seg_durations),
                        "text_len": len(seg_text_seq),
                        "mel_len": seg_mel_len,
                    })
                    logger.info(
                        f"  → 段 {seg_idx + 1}: {len(seg_wav) / sr:.1f}s, "
                        f"mel={seg_mel_len}, text_len={len(seg_text_seq)}"
                    )
            else:
                # 无需切分，直接添加
                durations = self._compute_durations(
                    len(text_seq), alignment, text, text_mode, mel_len
                )
                self.samples.append({
                    "text_seq": torch.LongTensor(text_seq),
                    "mel": torch.FloatTensor(mel),
                    "wav": torch.FloatTensor(wav),
                    "durations": torch.FloatTensor(durations),
                    "text_len": len(text_seq),
                    "mel_len": mel_len,
                })
                logger.info(
                    f"[Dataset] 音频 {ci+1}: {total_dur:.1f}s, "
                    f"mel_len={mel_len}, text_len={len(text_seq)}"
                )

        if not self.samples:
            raise ValueError("没有有效的训练样本，请检查音频文件和文本")

        logger.info(f"[Dataset] 总训练样本: {len(self.samples)}")

    @staticmethod
    def _split_long_clip(wav, sr, alignment, text, text_mode):
        """将超长音频按句子边界切分为 ≤ MAX_SEGMENT_FRAMES 的短段。

        Returns:
            list of (wav_chunk, text_chunk, alignment_chunk) 或 None（无需切分）
        """
        total_frames = len(wav) // HOP
        if total_frames <= MAX_SEGMENT_FRAMES or not alignment:
            return None

        clean_text = text.strip().replace(" ", "")
        sentence_enders = set("。！？.!?\n")

        segments = []
        seg_char_start = 0
        seg_time_start = 0.0

        for i, a in enumerate(alignment):
            end_time = a["end"]
            seg_dur_frames = int(end_time * SR / HOP) - int(seg_time_start * SR / HOP)
            is_sentence_end = a["char"] in sentence_enders

            # 达到切分条件：超过最大值且遇到句子结束
            if is_sentence_end and seg_dur_frames >= MAX_SEGMENT_FRAMES * 0.8:
                # 切分点：把这段文本和音频取出来
                chunk_text = clean_text[seg_char_start:i + 1]
                chunk_alignment = alignment[seg_char_start:i + 1]
                start_s = int(seg_time_start * SR)
                end_s = int(end_time * SR)
                chunk_wav = wav[start_s:end_s]

                if len(chunk_text.strip()) >= 5:
                    segments.append((chunk_wav, chunk_text, chunk_alignment))

                seg_char_start = i + 1
                seg_time_start = end_time

        # 剩余尾部
        if seg_char_start < len(alignment):
            chunk_text = clean_text[seg_char_start:]
            chunk_alignment = alignment[seg_char_start:]
            start_s = int(seg_time_start * SR)
            chunk_wav = wav[start_s:]
            if len(chunk_text.strip()) >= 5:
                segments.append((chunk_wav, chunk_text, chunk_alignment))

        return segments if segments else None

    def _compute_durations(self, text_seq_len: int, alignment: list,
                           text: str, text_mode: str, mel_len: int) -> list:
        """基于对齐结果计算每个音素的 duration（mel 帧数）。

        Args:
            text_seq_len: 音素序列总长度（含 <bos> <eos>）
            alignment: Whisper 对齐结果 [{"char": str, "start": float, "end": float}, ...]
            text: 原始文本
            text_mode: 编码模式
            mel_len: 该音频 mel 谱的真实帧数

        Returns:
            list of float: 每个音素对应的 mel 帧数
        """
        hop = audio_processor.HOP_LENGTH
        sr = audio_processor.SAMPLE_RATE

        if not alignment:
            # 回退：平均值（排除 <bos> <eos>）
            inner_len = max(text_seq_len - 2, 1)
            base = max(mel_len / inner_len, 0.5)
            durations = [1.0] + [base] * (text_seq_len - 2) + [1.0]
            return durations

        # 每个字的 mel 帧数（根据对齐时间戳）
        char_frames = []
        for a in alignment:
            start_f = int(a["start"] * sr / hop)
            end_f = int(a["end"] * sr / hop)
            char_frames.append(max(end_f - start_f, 1))

        # 每个字的音素数
        phoneme_counts = _count_phonemes_per_char(text, text_mode)

        # 长度校验：对齐字数与文本字数可能不一致
        if len(phoneme_counts) > len(char_frames):
            avg_char_frame = sum(char_frames) / max(len(char_frames), 1)
            char_frames.extend([avg_char_frame] * (len(phoneme_counts) - len(char_frames)))
        elif len(char_frames) > len(phoneme_counts):
            char_frames = char_frames[:len(phoneme_counts)]

        # 分配 duration：将每个字的 mel 帧数均分给其音素
        durations = [1.0]  # <bos>
        for cf, pc in zip(char_frames, phoneme_counts):
            per = max(cf / pc, 0.5)
            for _ in range(pc):
                durations.append(per)
        durations.append(1.0)  # <eos>

        # 补齐/截断到 text_seq_len
        if len(durations) > text_seq_len:
            durations = durations[:text_seq_len]
        while len(durations) < text_seq_len:
            durations.append(1.0)

        # 缩放使总和接近 mel_len（允许 <bos><eos> 占少量帧）
        total = sum(durations)
        if total > 0 and mel_len > 0:
            scale = mel_len / total
            durations = [d * scale for d in durations]

        return durations

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    """自定义 collate：padding 到同长度，返回 durations。"""
    text_seqs = [item["text_seq"] for item in batch]
    mels = [item["mel"] for item in batch]
    wavs = [item["wav"] for item in batch]
    durations_list = [item["durations"] for item in batch]
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

    max_dur_len = max(len(d) for d in durations_list)
    durations_padded = torch.zeros(len(batch), max_dur_len)
    for i, dur in enumerate(durations_list):
        durations_padded[i, :len(dur)] = dur

    max_wav_len = max(len(w) for w in wavs)
    wav_padded = torch.zeros(len(batch), max_wav_len)
    for i, wav in enumerate(wavs):
        wav_padded[i, :len(wav)] = wav

    return {
        "text_seq": text_padded,
        "mel": mel_padded,
        "wav": wav_padded,
        "durations": durations_padded,
        "text_len": torch.LongTensor(text_lens),
        "mel_len": torch.LongTensor(mel_lens),
    }


def create_dataloader(clips: list,
                      text_mode: str = "pinyin",
                      batch_size: int = 1) -> DataLoader:
    """创建训练 DataLoader。

    Args:
        clips: [{"path": str, "text": str, "emotion": str}, ...]
        text_mode: 文本编码模式
        batch_size: batch 大小（长音频被自动切分后，建议 batch=1 避免 OOM）

    Returns:
        DataLoader
    """
    dataset = VoiceDataset(clips, text_mode)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        drop_last=False,
    )
