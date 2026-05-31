#!/usr/bin/env python3
"""手动导出已训练好的模型为 ONNX（修复 TransformerBlock 后重试）"""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from trainer.export_onnx import export_model
from trainer.device import get_model_device

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", help="模型目录，如 models/voice_20260531_1324_9738")
    args = parser.parse_args()

    model_dir = args.model_dir
    checkpoint_path = os.path.join(model_dir, "checkpoint.pt")
    config_path = os.path.join(model_dir, "voice_config.json")

    if not os.path.exists(checkpoint_path):
        print(f"错误: checkpoint 不存在: {checkpoint_path}")
        sys.exit(1)

    # 读取配置获取参考文本和音频
    import json
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        voice_name = config.get("name", "我的声音")
        reference_text = config.get("reference_text", "")
        model_type = config.get("model_type", "vits")
        text_mode = config.get("text_mode", "pinyin")
    else:
        voice_name = "我的声音"
        reference_text = ""
        model_type = "vits"
        text_mode = "pinyin"

    ref_audio = os.path.join(model_dir, "reference_audio.wav")
    if not os.path.exists(ref_audio):
        # 尝试从 uploads 找最新参考音频
        uploads = [f for f in os.listdir("uploads") if f.endswith(".wav")]
        if uploads:
            ref_audio = os.path.join("uploads", sorted(uploads)[-1])
        else:
            ref_audio = ""

    print(f"模型目录: {model_dir}")
    print(f"模型类型: {model_type}")
    print(f"参考音频: {ref_audio}")
    print("开始导出 ONNX...")

    export_device, _ = get_model_device(for_export=True)

    export_model(
        model_type=model_type,
        checkpoint_path=checkpoint_path,
        output_dir=model_dir,
        voice_name=voice_name,
        reference_text=reference_text,
        audio_path=ref_audio,
        text_mode=text_mode,
        device=export_device,
    )
    print(f"导出完成! 文件在: {model_dir}")

if __name__ == "__main__":
    main()
