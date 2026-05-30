"""设备检测模块：CUDA (NVIDIA) → DirectML (AMD) → CPU"""
import logging
import torch

logger = logging.getLogger(__name__)


def get_device():
    """自动检测最佳计算设备。

    检测顺序：CUDA (NVIDIA) → DirectML (AMD) → CPU

    Returns:
        tuple: (torch.device 或 DirectML device, str) - 设备对象和设备标签
    """
    # 1. NVIDIA CUDA
    if torch.cuda.is_available():
        device = torch.device("cuda")
        try:
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"[Device] 使用 NVIDIA GPU: {gpu_name}")
            return device, f"CUDA: {gpu_name}"
        except Exception:
            logger.info("[Device] 使用 CUDA")
            return device, "CUDA"

    # 2. AMD / Intel DirectML（优先选独显 / 非 UMA 设备）
    try:
        import torch_directml
        if torch_directml.is_available():
            best_idx = 0
            best_name = ""
            count = torch_directml.device_count()
            for i in range(count):
                name = torch_directml.device_name(i)
                upper = name.upper()
                # 优先选 RX / Pro / 非 Graphics 的独显
                if "RX" in upper or "PRO" in upper or "RADEON" in upper:
                    if "GRAPHICS" not in upper:  # 排除集显
                        best_idx = i
                        best_name = name
                        break
                # 兜底：至少记住最后一个可用设备
                best_name = name
            dml_device = torch_directml.device(best_idx)
            logger.info(f"[Device] 使用 DirectML: {best_name} (device {best_idx}/{count})")
            return dml_device, f"DirectML: {best_name}"
        else:
            logger.warning("[Device] torch-directml 已安装但 DirectML 不可用，回退 CPU")
            return torch.device("cpu"), "CPU (DML unavailable)"
    except ImportError:
        logger.debug("[Device] torch-directml 未安装，跳过 DirectML")
    except Exception as e:
        logger.warning(f"[Device] DirectML 检测异常: {e}")

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
        logger.debug(f"[ONNX] 全部可用 providers: {available}")

        preferred = [
            "CUDAExecutionProvider",
            "DmlExecutionProvider",
            "CPUExecutionProvider",
        ]
        providers = [p for p in preferred if p in available]
        if not providers:
            providers = ["CPUExecutionProvider"]

        logger.info(f"[ONNX] 选用 providers: {providers}")
        return providers
    except ImportError:
        logger.warning("[ONNX] onnxruntime 未安装")
        return ["CPUExecutionProvider"]


def get_model_device(for_export: bool = False):
    """获取训练/推理设备。

    Args:
        for_export: 是否用于 ONNX 导出。AMD DirectML 下导出需强制 CPU。

    Returns:
        tuple: (device, device_label)
    """
    device, label = get_device()

    # DirectML 下 ONNX 导出不稳定，强制走 CPU
    if for_export and "DirectML" in label:
        logger.info("[Device] DirectML 下 ONNX 导出强制使用 CPU")
        return torch.device("cpu"), "CPU (export)"

    return device, label
