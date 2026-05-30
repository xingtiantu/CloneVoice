"""声音克隆训练器 - 一键安装依赖脚本"""
import subprocess
import sys
import os
import platform
from datetime import datetime

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

log_path = os.path.join(LOG_DIR, f"install_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
PYTHON = sys.executable


def log(msg: str):
    print(msg)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def run(cmd: list, check: bool = False, timeout: int = 600) -> int:
    cmd_str = " ".join(str(c) for c in cmd)
    log(f"  $ {cmd_str}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        with open(log_path, "a", encoding="utf-8") as f:
            if result.stdout:
                f.write(result.stdout.strip() + "\n")
            if result.stderr:
                f.write(result.stderr.strip() + "\n")
        if check and result.returncode != 0:
            log(f"  FAILED (code={result.returncode})")
        return result.returncode
    except subprocess.TimeoutExpired:
        log(f"  TIMEOUT ({timeout}s)")
        return -1
    except Exception as e:
        log(f"  ERROR: {e}")
        return -1


def check_module(module: str) -> bool:
    return run([PYTHON, "-c", f"import {module}"], check=False) == 0


def detect_gpu():
    """检测 GPU：尽量不提前 import torch（可能还没装），用 wmic 先查。"""
    log("")
    log("[GPU] Detecting graphics card...")

    # 优先用 wmic（不依赖任何 Python 包）
    if platform.system() == "Windows":
        try:
            result = subprocess.run(
                'wmic path win32_videocontroller get name /value',
                capture_output=True, text=True, shell=True, timeout=10
            )
            for line in result.stdout.splitlines():
                if "=" in line:
                    name = line.split("=", 1)[1].strip()
                    if "AMD" in name or "Radeon" in name:
                        log(f"      Found AMD GPU: {name}")
                        return "AMD"
                    if "NVIDIA" in name:
                        log(f"      Found NVIDIA GPU: {name}")
                        return "NVIDIA"
        except Exception:
            pass

    # 如果 torch 已装，再确认一下
    try:
        import torch
        if torch.cuda.is_available():
            log(f"      NVIDIA CUDA available: {torch.cuda.get_device_name(0)}")
            return "NVIDIA"
    except ImportError:
        pass
    try:
        import torch_directml
        if torch_directml.is_available():
            log(f"      AMD DirectML available: {torch_directml.device_name(0)}")
            return "AMD"
    except ImportError:
        pass

    log("      No GPU detected, will use CPU")
    return "CPU"


def pip_install(packages: list):
    """用 pip 安装包列表。"""
    cmd = [PYTHON, "-m", "pip", "install"] + packages
    code = run(cmd)
    if code == 0:
        log("  OK")
    else:
        log(f"  FAILED - check {log_path}")
    return code


def main():
    print("=" * 45)
    print("  Voice Clone - 安装依赖")
    print("=" * 45)
    print(f"  Python: {PYTHON}")
    print(f"  Log:    {log_path}")
    print("")

    # 升级 pip
    log("[1/7] Upgrading pip...")
    run([PYTHON, "-m", "pip", "install", "--upgrade", "pip", "-q"])

    # 检测 GPU
    gpu_type = detect_gpu()

    # 2/7 Flask
    log("")
    log("[2/7] Flask (web framework)...")
    if not check_module("flask"):
        pip_install(["flask"])
    else:
        log("  Already installed, OK")

    # 3/7 PyTorch
    log("")
    log("[3/7] PyTorch...")
    if not check_module("torch"):
        if gpu_type == "AMD":
            log("  Installing torch-directml (AMD GPU)...")
            pip_install(["torch-directml", "torchaudio"])
        elif gpu_type == "NVIDIA":
            log("  Installing torch with CUDA 12.1...")
            pip_install(["torch", "torchaudio", "--index-url",
                         "https://download.pytorch.org/whl/cu121"])
        else:
            log("  Installing torch CPU version...")
            pip_install(["torch", "torchaudio", "--index-url",
                         "https://download.pytorch.org/whl/cpu"])
        if check_module("torch"):
            log("  OK")
        else:
            log("  FAILED - check log")
    else:
        log("  Already installed, OK")

    # 4/7 Audio libs
    log("")
    log("[4/7] Audio libs (librosa + soundfile + scipy)...")
    if not check_module("librosa"):
        pip_install(["librosa", "soundfile", "scipy"])
    else:
        log("  Already installed, OK")

    # 5/7 ONNX
    log("")
    log("[5/7] ONNX runtime...")
    if gpu_type == "AMD":
        pip_install(["onnxruntime-directml", "onnxruntime", "onnxscript"])
    elif gpu_type == "NVIDIA":
        pip_install(["onnxruntime-gpu", "onnxruntime", "onnxscript"])
    else:
        pip_install(["onnxruntime", "onnxscript"])
    if check_module("onnxruntime"):
        log("  OK")
    else:
        log("  FAILED")

    # 6/7 pypinyin
    log("")
    log("[6/7] Text processing (pypinyin)...")
    if not check_module("pypinyin"):
        pip_install(["pypinyin"])
    else:
        log("  Already installed, OK")

    # 7/7 Whisper
    log("")
    log("[7/7] Whisper (forced alignment)...")
    if not check_module("whisper"):
        log("  Installing openai-whisper (this may take a while)...")
        pip_install(["openai-whisper"])
        if check_module("whisper"):
            log("  OK")
        else:
            log("  FAILED - check log")
    else:
        log("  Already installed, OK")

    # === Verification ===
    log("")
    log("=" * 45)
    log("  Verification")
    log("=" * 45)
    log("")

    run([PYTHON, "-c", "import torch; print(f'  PyTorch: {torch.__version__}')"])
    run([PYTHON, "-c", "import torch; print(f'  CUDA: {torch.cuda.is_available()}')"])

    # DirectML check (single line)
    dml_code = run(
        [PYTHON, "-c",
         "import torch_directml as d;"
         "print('  DirectML:', d.is_available());"
         "print('  Device:', d.device_name(0) if d.is_available() else 'N/A')"],
        check=False
    )

    run([PYTHON, "-c",
         "import onnxruntime as ort;"
         "print(f'  Providers: {ort.get_available_providers()}')"])

    log("")
    log("=" * 45)
    log(f"  All done! GPU mode: {gpu_type}")
    log(f"  Run 'run.bat' to start.")
    log("=" * 45)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[FATAL] {e}")
        print(f"Check log: {log_path}")
    finally:
        input("\n按 Enter 键退出...")
