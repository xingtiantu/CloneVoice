# 声音克隆训练器 (Voice Clone Trainer)

训练真人音色模型，导出 ONNX 格式，供 LiveBuddy 导入使用。

## 功能特性

- 🎤 **浏览器录制**：通过麦克风录制 5-15 秒参考音频，实时波形可视化
- 📝 **文本对齐**：支持拼音/字符两种文本编码模式
- 🧠 **双模型架构**：FastSpeech2-lite（快速）/ VITS-lite（高质量）
- 🖥️ **多设备支持**：自动检测 CPU / NVIDIA CUDA / AMD DirectML
- 📊 **实时监控**：训练进度、Loss 曲线实时更新
- 🎵 **试听合成**：训练完成后输入文字即时试听，支持语速/音调调节
- 📦 **一键导出**：导出 ZIP 包直接导入 LiveBuddy

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

AMD 显卡用户额外安装：
```bash
pip install torch-directml onnxruntime-directml
```

### 2. 启动服务

```bash
python server.py
```

打开浏览器访问 `http://127.0.0.1:5000`

### 3. 使用流程

1. **录制**：点击麦克风录制 5-15 秒清晰语音，或上传已有音频
2. **配置**：输入对应文本，选择模型类型和训练参数
3. **训练**：等待 15-60 分钟（视模型类型和设备而定）
4. **导出**：试听满意后导出 ZIP 包

## 导出格式

```
my_voice.zip
├── voice_model.onnx      # ONNX 推理模型
├── voice_config.json     # 配置信息
└── reference_audio.wav   # 参考音频（22050Hz, 16bit, mono）
```

### LiveBuddy 推理接口

```python
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("voice_model.onnx", providers=["CPUExecutionProvider"])
# 输入：文本序列 (1, seq_len) int64
# 输出：mel谱 (1, T_mel, 80) float32
# 采样率：22050 Hz
```

## 模型说明

| 模型 | 参数量 | 训练时间(CPU) | 适用场景 |
|------|--------|--------------|---------|
| FastSpeech2-lite | ~10M | 15-30 分钟 | 快速实验、日常使用 |
| VITS-lite | ~18M | 30-60 分钟 | 追求更高质量 |

## 技术架构

- **前端**：纯 HTML/CSS/JS 单页面应用
- **后端**：Python Flask
- **模型**：PyTorch 训练 → ONNX 导出
- **推理**：onnxruntime（支持 CPU/CUDA/DirectML）

## 许可证

MIT License
