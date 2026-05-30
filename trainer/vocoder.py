"""预训练声码器封装：提供 mel -> wav 的统一推理接口。"""
import os
import logging
import torch

from .hifigan_lite import HiFiGANGenerator
from . import audio_processor

logger = logging.getLogger(__name__)

_vocoder = None
_vocoder_device = None


def get_vocoder(checkpoint_path: str = None, device: str = "cpu") -> torch.nn.Module:
    """获取声码器实例（懒加载 + 缓存）。

    优先加载用户模型目录下的声码器权重，其次尝试加载通用预训练权重。
    如果都失败，返回 None（推理时将回退到 Griffin-Lim）。

    Args:
        checkpoint_path: 用户训练好的 hifigan 权重路径（可选）
        device: 运行设备

    Returns:
        HiFiGANGenerator 实例，或 None
    """
    global _vocoder, _vocoder_device
    if _vocoder is not None and _vocoder_device == device:
        return _vocoder

    generator = HiFiGANGenerator().to(device)

    # 1. 尝试加载用户模型权重
    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            # 先加载到 CPU，避免 DirectML 在 torch.load 中的兼容性问题
            state = torch.load(checkpoint_path, map_location="cpu")
            # 优先提取 hifigan_state_dict，其次是 model_state_dict
            if isinstance(state, dict) and "hifigan_state_dict" in state:
                state = state["hifigan_state_dict"]
            elif isinstance(state, dict) and "model_state_dict" in state:
                state = state["model_state_dict"]
            generator.load_state_dict(state, strict=False)
            generator = generator.to(device)
            logger.info(f"[Vocoder] 加载用户声码器权重: {checkpoint_path}")
            generator.eval()
            _vocoder = generator
            _vocoder_device = device
            return _vocoder
        except Exception as e:
            logger.warning(f"[Vocoder] 用户权重加载失败: {e}")

    # 2. 尝试加载通用预训练权重（轻量版）
    pretrained_path = os.path.join("models", "pretrained", "hifigan_base.pt")
    if os.path.exists(pretrained_path):
        try:
            state = torch.load(pretrained_path, map_location=device)
            if isinstance(state, dict) and "model_state_dict" in state:
                state = state["model_state_dict"]
            generator.load_state_dict(state, strict=False)
            logger.info(f"[Vocoder] 加载预训练声码器: {pretrained_path}")
            generator.eval()
            _vocoder = generator
            _vocoder_device = device
            return _vocoder
        except Exception as e:
            logger.warning(f"[Vocoder] 预训练权重不匹配: {e}")

    logger.info("[Vocoder] 无可用声码器权重，推理将使用 Griffin-Lim 回退")
    return None


def mel_to_wav(
    mel: torch.Tensor,
    checkpoint_path: str = None,
    device: str = "cpu",
) -> torch.Tensor:
    """将 mel 谱转换为波形。

    Args:
        mel: (n_mels, T) 或 (1, n_mels, T) 的 mel 谱
        checkpoint_path: 声码器权重路径（可选）
        device: 运行设备

    Returns:
        wav: (T_wav,) 的一维波形张量
    """
    vocoder = get_vocoder(checkpoint_path, device)

    if vocoder is None:
        # Griffin-Lim 回退（质量较低但无需预训练权重）
        mel_np = mel.squeeze().cpu().numpy()
        wav = audio_processor.mel_to_wav_griffinlim(mel_np)
        return torch.FloatTensor(wav)

    if mel.dim() == 2:
        mel = mel.unsqueeze(0)  # (1, n_mels, T)

    with torch.no_grad():
        wav = vocoder(mel.to(device))

    return wav.squeeze(0).squeeze(0).cpu()


def unload_vocoder():
    """释放声码器显存缓存。"""
    global _vocoder, _vocoder_device
    _vocoder = None
    _vocoder_device = None
