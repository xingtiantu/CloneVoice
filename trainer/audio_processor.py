"""音频处理模块：加载、预处理、mel 提取、保存"""
import os
import logging
import numpy as np
import torch

logger = logging.getLogger(__name__)

# 全局常量
SAMPLE_RATE = 22050
N_FFT = 1024
HOP_LENGTH = 256
N_MELS = 80
FMIN = 0
FMAX = 11025


def load_audio(path: str, sr: int = SAMPLE_RATE) -> np.ndarray:
    """加载音频文件，返回单声道波形。

    支持 WAV/MP3/FLAC/OGG 等格式。
    优先使用 librosa，失败时回退到 torchaudio。

    Args:
        path: 音频文件路径
        sr: 目标采样率

    Returns:
        np.ndarray: 归一化后的单声道波形 (samples,)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"音频文件不存在: {path}")

    wav = None

    # 方法 1: librosa（依赖 ffmpeg/soundfile 后端）
    try:
        import librosa
        wav, _ = librosa.load(path, sr=sr, mono=True)
    except Exception as e1:
        logger.warning(f"[Audio] librosa 加载失败: {e1}")

        # 方法 2: torchaudio（支持更多格式）
        try:
            import torchaudio
            waveform, orig_sr = torchaudio.load(path)
            # 转单声道
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            waveform = waveform.squeeze(0).numpy()
            # 重采样
            if orig_sr != sr:
                import torchaudio.transforms as T
                resampler = T.Resample(orig_sr, sr)
                waveform = resampler(torch.FloatTensor(waveform)).numpy()
            wav = waveform
        except Exception as e2:
            logger.warning(f"[Audio] torchaudio 加载失败: {e2}")

            # 方法 3: soundfile（WAV 直读）
            try:
                import soundfile as sf
                data, file_sr = sf.read(path)
                if data.ndim > 1:
                    data = data.mean(axis=1)
                if file_sr != sr:
                    from scipy.signal import resample
                    num_samples = int(len(data) * sr / file_sr)
                    data = resample(data, num_samples)
                wav = data.astype(np.float32)
            except Exception as e3:
                raise RuntimeError(
                    f"所有音频加载方式均失败。\n"
                    f"  librosa: {e1}\n"
                    f"  torchaudio: {e2}\n"
                    f"  soundfile: {e3}\n"
                    f"提示：请确保安装了 ffmpeg，或上传标准 WAV 格式文件。"
                )

    # 音量归一化
    peak = np.max(np.abs(wav))
    if peak > 0:
        wav = wav / peak * 0.95

    # 静音裁剪
    try:
        import librosa
        wav, _ = librosa.effects.trim(wav, top_db=25)
    except Exception:
        pass  # trim 失败不影响整体

    logger.info(f"[Audio] 加载音频: {path}, 时长: {len(wav)/sr:.2f}s, 采样率: {sr}")
    return wav


def wav_to_mel(wav: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """提取 mel spectrogram。

    Args:
        wav: 波形数组 (samples,)
        sr: 采样率

    Returns:
        np.ndarray: mel spectrogram (n_mels, T)
    """
    import librosa

    mel = librosa.feature.melspectrogram(
        y=wav,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX,
        power=1.0,  # magnitude spectrogram
    )

    # log mel
    mel = np.log(np.clip(mel, a_min=1e-5, a_max=None))

    return mel


def mel_to_wav_griffinlim(mel: np.ndarray, sr: int = SAMPLE_RATE, n_iter: int = 32) -> np.ndarray:
    """使用 Griffin-Lim 算法从 mel 谱重建波形（快速预览用）。

    Args:
        mel: log mel spectrogram (n_mels, T)
        sr: 采样率
        n_iter: Griffin-Lim 迭代次数

    Returns:
        np.ndarray: 重建的波形 (samples,)
    """
    import librosa

    # 反 log
    mel_exp = np.exp(mel)

    # mel → linear（近似）
    mel_basis = librosa.filters.mel(
        sr=sr, n_fft=N_FFT, n_mels=N_MELS, fmin=FMIN, fmax=FMAX
    )
    # 伪逆
    mel_inv = np.linalg.pinv(mel_basis)
    spec = np.maximum(np.dot(mel_inv, mel_exp), 0)

    # Griffin-Lim
    wav = librosa.griffinlim(spec, n_iter=n_iter, hop_length=HOP_LENGTH, n_fft=N_FFT)

    # 归一化
    peak = np.max(np.abs(wav))
    if peak > 0:
        wav = wav / peak * 0.95

    return wav


def load_and_preprocess(path: str):
    """完整预处理流程：加载 → 归一化 → mel。

    Args:
        path: 音频文件路径

    Returns:
        tuple: (wav_tensor [1, samples], mel_tensor [1, n_mels, T])
    """
    wav = load_audio(path)
    mel = wav_to_mel(wav)

    wav_tensor = torch.FloatTensor(wav).unsqueeze(0)
    mel_tensor = torch.FloatTensor(mel).unsqueeze(0)

    return wav_tensor, mel_tensor


def save_wav(wav, path: str, sr: int = SAMPLE_RATE):
    """保存波形为 WAV 文件。

    Args:
        wav: numpy array 或 torch tensor
        path: 输出路径
        sr: 采样率
    """
    import soundfile as sf

    if isinstance(wav, torch.Tensor):
        wav = wav.cpu().numpy()

    # 确保是 1D
    wav = wav.squeeze()

    # 归一化
    peak = np.max(np.abs(wav))
    if peak > 0:
        wav = wav / peak * 0.95

    # 裁剪到 int16 范围
    wav = np.clip(wav, -1.0, 1.0)

    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    sf.write(path, wav, sr, subtype="PCM_16")
    logger.info(f"[Audio] 保存音频: {path}")


def get_audio_duration(path: str) -> float:
    """获取音频时长（秒）。"""
    import librosa
    return librosa.get_duration(filename=path)


def split_audio_by_silence(wav: np.ndarray, sr: int = SAMPLE_RATE,
                           min_segment_sec: float = 2.0,
                           max_segment_sec: float = 5.0,
                           top_db: int = 30) -> list:
    """基于静音检测将音频切分为小段。

    Args:
        wav: 波形数组
        sr: 采样率
        min_segment_sec: 最小段时长
        max_segment_sec: 最大段时长
        top_db: 静音阈值

    Returns:
        list: 切分后的波形片段列表
    """
    import librosa

    # 检测非静音区间
    intervals = librosa.effects.split(wav, top_db=top_db)

    segments = []
    min_samples = int(min_segment_sec * sr)
    max_samples = int(max_segment_sec * sr)

    for start, end in intervals:
        chunk = wav[start:end]
        # 如果太短，跳过
        if len(chunk) < min_samples:
            continue
        # 如果太长，进一步切分
        while len(chunk) > max_samples:
            segments.append(chunk[:max_samples])
            chunk = chunk[max_samples:]
        # 剩余部分
        if len(chunk) >= min_samples:
            segments.append(chunk)

    # 如果没有合适分段，直接用整段
    if not segments and len(wav) > sr * 0.5:
        # 即使较短也保留
        if len(wav) > max_samples:
            for i in range(0, len(wav), max_samples):
                seg = wav[i:i + max_samples]
                if len(seg) >= sr * 0.5:
                    segments.append(seg)
        else:
            segments.append(wav)

    logger.info(f"[Audio] 音频切分: {len(intervals)} 个非静音段 → {len(segments)} 个训练片段")
    return segments

def run_inference_onnx_fast(onnx_path, text_seq, providers=None):
    """ONNX 推理快捷入口（单文件模型）。"""
    from .export_onnx import run_inference_onnx
    return run_inference_onnx(onnx_path, text_seq, providers)


def run_inference_split_fast(model_dir, text_seq, providers=None):
    """分段 ONNX 推理快捷入口（encoder + NumPy expand + decoder + postnet）。"""
    from .export_onnx import run_inference_split
    return run_inference_split(model_dir, text_seq, providers)

