"""GPT-SoVITS 推理封装：提供简洁的 TTS 接口。"""
import os
import sys
import logging
import tempfile
import numpy as np

logger = logging.getLogger(__name__)

# GPT-SoVITS 安装路径（D 盘）
GPT_SOVITS_DIR = r"D:\GPT-SoVITS"
MODELS_DIR = os.path.join(GPT_SOVITS_DIR, "GPT_SoVITS", "pretrained_models")

# 延迟导入，避免服务启动时 GPT-SoVITS 还没装好就报错
_tts_pipeline = None


def _init_pipeline():
    """初始化 GPT-SoVITS 推理管道。"""
    global _tts_pipeline
    if _tts_pipeline is not None:
        return _tts_pipeline

    if not os.path.exists(GPT_SOVITS_DIR):
        raise RuntimeError(
            f"GPT-SoVITS 未安装。请先运行：python setup_gpt_sovits.py\n"
            f"预期路径：{GPT_SOVITS_DIR}"
        )

    # 将 GPT-SoVITS 加入搜索路径
    if GPT_SOVITS_DIR not in sys.path:
        sys.path.insert(0, GPT_SOVITS_DIR)

    try:
        # GPT-SoVITS 新版 TTS 推理类
        from GPT_SoVITS.TTS_infer_pack.TTS import TTS, Config

        # 自动查找模型文件
        cnhubert_dir = os.path.join(MODELS_DIR, "chinese-hubert-base")
        bert_dir = os.path.join(MODELS_DIR, "Chinese-RoBERTa-wwm-ext-large")

        # 列出所有可能的模型文件名
        sovits_candidates = [
            "s2G488k.pth",
            "s2G2333k.pth",
            "gsv-v2-final.pth",
        ]
        gpt_candidates = [
            "s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt",
            "s1bert25hz-2kh-longer-epoch=82e-step=30232.ckpt",
        ]

        sovits_path = _find_first_existing(MODELS_DIR, sovits_candidates)
        gpt_path = _find_first_existing(MODELS_DIR, gpt_candidates)

        if not sovits_path or not gpt_path:
            raise RuntimeError(
                f"未找到 GPT-SoVITS 预训练模型。\n"
                f"请确认模型已下载到：{MODELS_DIR}"
            )

        config = Config(
            device="cuda" if _has_cuda() else "cpu",
            is_half=False,
            t2s_weights_path=gpt_path,
            vits_weights_path=sovits_path,
            bert_base_path=bert_dir,
            cnhuhbert_base_path=cnhubert_dir,
        )

        _tts_pipeline = TTS(config)
        logger.info("[GPT-SoVITS] 推理管道初始化成功")
        return _tts_pipeline

    except Exception as e:
        logger.error(f"[GPT-SoVITS] 初始化失败: {e}")
        raise


def _find_first_existing(base_dir, candidates):
    for name in candidates:
        path = os.path.join(base_dir, name)
        if os.path.exists(path):
            return path
    return None


def _has_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def synthesize(
    ref_audio_path: str,
    ref_text: str,
    target_text: str,
    output_path: str = None,
    speed: float = 1.0,
) -> str:
    """
    使用 GPT-SoVITS 合成语音。

    Args:
        ref_audio_path: 参考音频路径（提供音色）
        ref_text: 参考音频对应的文本（用于对齐）
        target_text: 要合成的目标文本
        output_path: 输出音频路径（默认生成临时文件）
        speed: 语速倍率

    Returns:
        输出音频的绝对路径
    """
    tts = _init_pipeline()

    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)

    # GPT-SoVITS 推理参数
    params = {
        "text": target_text,
        "text_lang": "zh",
        "ref_audio_path": ref_audio_path,
        "prompt_text": ref_text,
        "prompt_lang": "zh",
        "top_k": 5,
        "top_p": 1.0,
        "temperature": 1.0,
        "text_split_method": "cut5",
        "batch_size": 1,
        "speed_factor": speed,
        "ref_text_free": False,
        "split_bucket": True,
        "return_fragment": False,
        "fragment_interval": 0.3,
    }

    try:
        tts.run(params, output_path)
        logger.info(f"[GPT-SoVITS] 合成完成: {output_path}")
        return os.path.abspath(output_path)
    except Exception as e:
        logger.error(f"[GPT-SoVITS] 合成失败: {e}")
        raise


def is_available() -> bool:
    """检查 GPT-SoVITS 是否已安装并可用。"""
    return os.path.exists(GPT_SOVITS_DIR) and os.path.exists(MODELS_DIR)
