"""设备检测模块：CUDA → DirectML → CPU"""
import logging
import torch

logger = logging.getLogger(__name__)


def get_device():
    """自动检测最佳计算设备。

    检测顺序：CUDA (NVIDIA) → DirectML (AMD/Intel) → CPU

    Returns:
        tuple: (torch.device, str) - 设备对象和设备标签
    """
    # 1. NVIDIA CUDA
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"[Device] 使用 NVIDIA GPU: {gpu_name}")
        return device, f"CUDA: {gpu_name}"

    # 2. AMD/Intel DirectML
    try:
        import torch_directml
        if torch_directml.is_available():
            dml_device = torch_directml.device()
            device_name = torch_directml.device_name(0)
            logger.info(f"[Device] 使用 DirectML 设备: {device_name}")
            return dml_device, f"DirectML: {device_name}"
    except ImportError:
        logger.debug("[Device] torch-directml 未安装，跳过 DirectML 检测")
    except Exception as e:
        logger.debug(f"[Device] DirectML 检测失败: {e}")

    # 3. CPU fallback
    logger.info("[Device] 使用 CPU 训练")
    return torch.device("cpu"), "CPU"


def get_onnx_providers():
    """获取 onnxruntime 推理 providers（按优先级排序）。

    Returns:
        list: 可用的 ExecutionProvider 列表
    """
    try:
        import onnxruntime as ort
        available = ort.get_available_providers()

        # 按优先级排列
        preferred = [
            "CUDAExecutionProvider",
            "DmlExecutionProvider",
            "CPUExecutionProvider",
        ]
        providers = [p for p in preferred if p in available]
        if not providers:
            providers = ["CPUExecutionProvider"]

        logger.info(f"[ONNX] 可用 providers: {providers}")
        return providers
    except ImportError:
        logger.warning("[ONNX] onnxruntime 未安装")
        return ["CPUExecutionProvider"]
