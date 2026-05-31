#!/usr/bin/env python3
"""GPT-SoVITS 一键安装脚本（自动装到 D 盘，不占 C 盘）"""
import os
import sys
import shutil
import subprocess

DRIVE = "D:"
INSTALL_DIR = os.path.join(DRIVE, "GPT-SoVITS")
MODEL_DIR = os.path.join(INSTALL_DIR, "GPT_SoVITS", "pretrained_models")


def check_space():
    if not os.path.exists(DRIVE):
        print(f"❌ 错误：{DRIVE} 盘不存在")
        sys.exit(1)
    total, used, free = shutil.disk_usage(DRIVE)
    gb_free = free / (1024 ** 3)
    print(f"📀 {DRIVE} 盘可用空间：{gb_free:.1f} GB")
    if gb_free < 8:
        print("❌ 错误：需要至少 8GB 可用空间")
        sys.exit(1)
    print("✅ 空间足够")


def run(cmd, **kwargs):
    print(f">>> {' '.join(cmd)}")
    subprocess.run(cmd, check=True, **kwargs)


def clone_repo():
    if os.path.exists(os.path.join(INSTALL_DIR, ".git")):
        print("📁 GPT-SoVITS 仓库已存在，跳过克隆")
        return
    os.makedirs(INSTALL_DIR, exist_ok=True)
    print("📥 正在克隆 GPT-SoVITS 仓库...")
    run(["git", "clone", "--depth", "1", "https://github.com/RVC-Boss/GPT-SoVITS.git", INSTALL_DIR])
    print("✅ 仓库克隆完成")


def install_deps():
    print("📦 安装核心依赖（跳过编译型/NVIDIA/日语包）...")

    # 纯 Python 包 + 推理必需，跳过 torch（你已有 torch-directml）
    core_packages = [
        "numpy<2.0", "scipy", "librosa==0.10.2", "numba",
        "onnxruntime", "tqdm", "cn2an", "pypinyin",
        "modelscope", "sentencepiece", "transformers>=4.43,<=4.50",
        "peft<0.18.0", "chardet", "PyYAML", "psutil",
        "jieba", "split-lang", "fast_langdetect>=0.3.1",
        "wordsegment", "rotary_embedding_torch",
        "g2p_en", "g2pk2", "ToJyutping",
        "torchmetrics<=1.5", "pydantic<=2.10.6",
        "ctranslate2>=4.0,<5", "av>=11",
    ]
    run([sys.executable, "-m", "pip", "install"] + core_packages)

    # 单独装 torchaudio CPU 版（避免覆盖 torch-directml）
    print("📦 安装 torchaudio...")
    run([sys.executable, "-m", "pip", "install", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cpu"])

    print("✅ 依赖安装完成")


def download_models():
    """使用 HuggingFace 国内镜像下载预训练模型"""
    os.makedirs(MODEL_DIR, exist_ok=True)
    print("📥 开始下载预训练模型（使用 hf-mirror.com 国内镜像）...")

    base_url = "https://hf-mirror.com/lj1995/GPT-SoVITS/resolve/main"
    files = [
        "s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt",
        "s2G488k.pth",
        "s2D488k.pth",
    ]

    for fname in files:
        out_path = os.path.join(MODEL_DIR, fname)
        if os.path.exists(out_path):
            print(f"  ✅ 已存在：{fname}")
            continue
        url = f"{base_url}/{fname}"
        print(f"  ⬇️  下载：{fname} ...")
        try:
            import urllib.request
            urllib.request.urlretrieve(url, out_path)
            print(f"  ✅ 完成：{fname}")
        except Exception as e:
            print(f"  ❌ 失败：{e}")
            print(f"     链接：{url}")

    # 同时下载 tokenizer 和配置文件
    config_files = ["config.json", "tokenizer.json"]
    for fname in config_files:
        url = f"{base_url}/{fname}"
        out_path = os.path.join(MODEL_DIR, fname)
        if os.path.exists(out_path):
            continue
        try:
            import urllib.request
            urllib.request.urlretrieve(url, out_path)
        except Exception:
            pass

    print("✅ 模型下载完成")


def main():
    print("=" * 50)
    print("  GPT-SoVITS 一键安装脚本")
    print(f"  安装位置：{INSTALL_DIR}")
    print("=" * 50)

    check_space()
    clone_repo()
    install_deps()
    download_models()

    print("\n" + "=" * 50)
    print("🎉 安装完成！")
    print(f"📁 仓库路径：{INSTALL_DIR}")
    print(f"📁 模型路径：{MODEL_DIR}")
    print("\n下一步：重启你的声音克隆服务，即可使用 GPT-SoVITS 推理")
    print("=" * 50)


if __name__ == "__main__":
    main()
