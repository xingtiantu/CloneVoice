"""Whisper 强制对齐模块：实现音频-文本字级时间戳对齐。"""
import os
import json
import logging
import hashlib
from typing import List, Dict

logger = logging.getLogger(__name__)

_model = None


def get_whisper_model(model_size: str = "base"):
    """懒加载 Whisper 模型（只加载一次）。"""
    global _model
    if _model is None:
        import whisper
        logger.info(f"[Aligner] 加载 Whisper {model_size} 模型...")
        _model = whisper.load_model(model_size)
        logger.info("[Aligner] Whisper 模型加载完成")
    return _model


def _compute_cache_key(audio_path: str, text: str) -> str:
    """基于音频路径+文本内容计算缓存文件名。"""
    m = hashlib.md5()
    m.update(os.path.abspath(audio_path).encode("utf-8"))
    m.update(text.encode("utf-8"))
    return m.hexdigest() + ".json"


def align_audio_text(
    audio_path: str,
    text: str,
    cache_dir: str = "models/cache/alignments",
    model_size: str = "base",
) -> List[Dict]:
    """对音频和文本进行字级强制对齐。

    使用 Whisper 模型转录音频并获取 segment 级时间戳，
    然后通过 LCS（最长公共子序列）将时间戳映射到用户提供的真实文本。

    Args:
        audio_path: 音频文件路径
        text: 用户提供的真实朗读文本
        cache_dir: 对齐结果缓存目录
        model_size: Whisper 模型尺寸（base 约 74M，速度与精度平衡）

    Returns:
        list: 每个字的 {"char": str, "start": float, "end": float}
              start/end 单位为秒，-1 表示未匹配（会被插值修复）
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_key = _compute_cache_key(audio_path, text)
    cache_path = os.path.join(cache_dir, cache_key)

    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            logger.debug(f"[Aligner] 命中缓存: {cache_key}")
            return cached
        except Exception:
            pass

    if not text or not text.strip():
        logger.warning("[Aligner] 文本为空，跳过对齐")
        return _fallback_alignment(audio_path, text)

    try:
        model = get_whisper_model(model_size)
        logger.info(f"[Aligner] 开始对齐: {os.path.basename(audio_path)}")

        # 使用用户文本作为初始提示，提高识别准确率
        prompt = text[:224] if len(text) > 224 else text
        result = model.transcribe(
            audio_path,
            language="zh",
            condition_on_previous_text=False,
            initial_prompt=prompt,
        )

        segments = result.get("segments", [])
        if not segments:
            logger.warning("[Aligner] Whisper 未识别到语音，回退到时间比例")
            return _fallback_alignment(audio_path, text)

        # 1. 将 Whisper segment 拆分为字级时间戳（segment 内均匀分配）
        whisper_chars = []
        for seg in segments:
            seg_text = seg.get("text", "").strip().replace(" ", "")
            start = seg.get("start", 0.0)
            end = seg.get("end", 0.0)
            if not seg_text:
                continue
            dur = max(end - start, 0.05)
            char_dur = dur / len(seg_text)
            for idx, ch in enumerate(seg_text):
                whisper_chars.append({
                    "char": ch,
                    "start": start + idx * char_dur,
                    "end": start + (idx + 1) * char_dur,
                })

        # 2. 将用户文本与 Whisper 识别文本通过 LCS 对齐
        user_chars = list(text.strip().replace(" ", ""))
        aligned = _lcs_alignment(user_chars, whisper_chars)

        # 3. 对未匹配的字进行线性插值
        aligned = _interpolate_missing(aligned)

        logger.info(f"[Aligner] 对齐完成: {len(aligned)} 字")

        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(aligned, f, ensure_ascii=False)

        return aligned

    except Exception as e:
        logger.error(f"[Aligner] 对齐失败: {e}", exc_info=True)
        return _fallback_alignment(audio_path, text)


def _fallback_alignment(audio_path: str, text: str) -> List[Dict]:
    """回退方案：按字数均匀分配音频总时长。"""
    try:
        import librosa
        dur = librosa.get_duration(path=audio_path)
    except Exception:
        dur = 10.0

    chars = list(text.strip().replace(" ", ""))
    if not chars:
        return []

    char_dur = dur / len(chars)
    return [
        {"char": ch, "start": i * char_dur, "end": (i + 1) * char_dur}
        for i, ch in enumerate(chars)
    ]


def _lcs_alignment(user_chars: List[str], whisper_chars: List[Dict]) -> List[Dict]:
    """基于最长公共子序列将 Whisper 时间戳映射到用户文本。

    动态规划求解 LCS，然后回溯建立 user_char_idx → whisper_char_idx 的映射。
    """
    m, n = len(user_chars), len(whisper_chars)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if user_chars[i - 1] == whisper_chars[j - 1]["char"]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    # 回溯
    i, j = m, n
    matches = {}
    while i > 0 and j > 0:
        if user_chars[i - 1] == whisper_chars[j - 1]["char"]:
            matches[i - 1] = j - 1
            i -= 1
            j -= 1
        elif dp[i - 1][j] > dp[i][j - 1]:
            i -= 1
        else:
            j -= 1

    aligned = []
    for idx, ch in enumerate(user_chars):
        if idx in matches:
            w = whisper_chars[matches[idx]]
            aligned.append({"char": ch, "start": w["start"], "end": w["end"]})
        else:
            aligned.append({"char": ch, "start": -1.0, "end": -1.0})

    return aligned


def _interpolate_missing(aligned: List[Dict]) -> List[Dict]:
    """对 start=-1 的未匹配字进行线性插值，补全时间戳。"""
    n = len(aligned)
    if n == 0:
        return aligned

    # 找到第一个和最后一个有效时间戳
    first_valid = None
    last_valid = None
    for i, item in enumerate(aligned):
        if item["start"] >= 0:
            if first_valid is None:
                first_valid = i
            last_valid = i

    if first_valid is None:
        # 全部无效，均匀分配虚拟时间
        for i, item in enumerate(aligned):
            item["start"] = float(i)
            item["end"] = float(i + 1)
        return aligned

    # 头部未匹配：从 0 插值到第一个有效字
    if first_valid > 0:
        end_t = aligned[first_valid]["start"]
        step = end_t / first_valid
        for i in range(first_valid):
            aligned[i]["start"] = i * step
            aligned[i]["end"] = (i + 1) * step

    # 尾部未匹配：从最后一个有效字外推
    if last_valid is not None and last_valid < n - 1:
        start_t = aligned[last_valid]["end"]
        remaining = n - 1 - last_valid
        # 粗略估计：用前面平均字长外推
        avg_dur = _estimate_avg_char_dur(aligned)
        for i in range(last_valid + 1, n):
            aligned[i]["start"] = start_t + (i - last_valid - 1) * avg_dur
            aligned[i]["end"] = start_t + (i - last_valid) * avg_dur

    # 中间未匹配：在前后有效字之间插值
    i = 0
    while i < n:
        if aligned[i]["start"] < 0:
            j = i
            while j < n and aligned[j]["start"] < 0:
                j += 1
            prev_end = aligned[i - 1]["end"] if i > 0 else 0.0
            next_start = aligned[j]["start"] if j < n else prev_end + (j - i)
            step = (next_start - prev_end) / max(j - i, 1)
            for k in range(i, j):
                aligned[k]["start"] = prev_end + (k - i) * step
                aligned[k]["end"] = prev_end + (k - i + 1) * step
            i = j
        else:
            i += 1

    return aligned


def _estimate_avg_char_dur(aligned: List[Dict]) -> float:
    """从已对齐数据中估计平均字长（秒）。"""
    valid = [a for a in aligned if a["start"] >= 0]
    if len(valid) < 2:
        return 0.25
    total_dur = sum(v["end"] - v["start"] for v in valid)
    return total_dur / len(valid)
