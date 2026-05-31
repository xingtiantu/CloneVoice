# 声音克隆训练器 (Voice Clone Trainer)

**训练真人音色模型，导出 ONNX 格式，供下游项目（如 LiveBuddy）实时推理使用。**

> **🤝 寻求社区贡献**
> 
> 本项目已完成从数据采集、模型训练、ONNX 导出到 Web 服务的**完整技术链路**，但在**音质还原度**上遇到了瓶颈。
> 我们已系统性排查并修复了设备检测、ONNX 导出、声码器训练、推理兼容等十余个技术问题（详见 [调试记录](#调试记录)），但受限于轻量模型参数量（10M~18M）和训练数据规模，音色克隆效果仍未达到生产可用标准。
> 
> **如果你熟悉 TTS 预训练模型蒸馏、数据增强或 ONNX 优化，欢迎参与贡献！** 项目骨架完整，只需在模型和数据层面突破，即可达到商业级效果。
> 
> ---
> 
> **项目状态：架构成熟 / 音质待突破**
> 
> 当前版本在训练管线、ONNX 导出和推理流程上已修复多个技术问题，但**受限于模型架构和数据量，音质（音色还原度）仍未达到可用标准**。
> 详见下方 [已知问题与瓶颈](#已知问题与瓶颈) 和 [调试记录](#调试记录)。

---

## 产品设计与目的

本项目目标是构建一套**轻量级、可私有化部署的声音克隆训练管线**，核心产出是 **ONNX 模型包**，可直接嵌入下游应用（如 LiveBuddy）进行实时 TTS 推理。

### 设计原则

1. **数据真实性优先**：前端支持为每段录音绑定真实朗读文本，彻底取代硬编码故事文案，从根本上解决文本-音频不匹配问题。
2. **精确对齐替代估算**：集成 Whisper 字级强制对齐，替代"按时间比例分配文本"的粗糙假设，建立精确的音频-文本映射。
3. **预训练声码器替代从头训练**：引入通用 HiFi-GAN 声码器权重，解决原项目在参考音频上训练 1500 步易过拟合导致的金属音/爆音问题。
4. **轻量 ONNX 导出**：训练完成后自动导出 ONNX，不依赖 PyTorch 运行时，便于嵌入各类终端环境。

## 功能特性

- 🎤 **浏览器录制/上传**：支持麦克风录制或上传已有音频
- 📝 **真实文本绑定**：每段音频可输入真实朗读内容，取代硬编码训练文本
- 🔗 **Whisper 强制对齐**：基于 Whisper base 模型生成字级时间戳，精确切分训练样本
- 🧠 **双模型架构**：FastSpeech2-lite（轻量）/ VITS-lite（端到端）
- 🖥️ **多设备支持**：自动检测 CPU / NVIDIA CUDA / AMD DirectML，优先选择独显
- 📊 **实时监控**：训练进度、Loss 曲线、设备状态实时更新
- 🎵 **试听合成**：训练完成后输入文字即时试听，支持语速/音调调节
- 📦 **一键导出 ONNX**：导出 ZIP 包（ONNX + config + 参考音频）供下游项目导入
- 🚀 **GPT-SoVITS 对比试听（实验性）**：可调用 GPT-SoVITS 大模型进行 zero-shot 试听（仅供效果对比，**不可导出 ONNX**）

---

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
# 或双击 run.bat
```

打开浏览器访问 `http://127.0.0.1:5000`

### 3. 使用流程

1. **录制/上传**：点击麦克风录制 5~15 秒清晰语音，或上传已有音频（建议 5 分钟以上）
2. **配置**：输入对应文本，选择模型类型（推荐 VITS）和训练参数
3. **训练**：等待 15~60 分钟（视模型类型和设备而定）
4. **试听/导出**：试听满意后导出 ZIP 包

---

## 导出格式

```
my_voice.zip
├── voice_model.onnx      # ONNX 推理模型（VITS 为单文件，FastSpeech2 为分段模型）
├── voice_config.json     # 配置信息（文本模式、采样率、参考文本等）
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

---

## 模型说明

| 模型 | 参数量 | 训练时间(RX 9070 XT) | 声码器 | 适用场景 |
|------|--------|:-------------------:|:------:|---------|
| FastSpeech2-lite | ~10M | 15~30 分钟 | HiFi-GAN（训练）| 快速实验 |
| VITS-lite | ~18M | 30~60 分钟 | 端到端（mel）+ HiFi-GAN | 追求更好质量 |

---

## 已知问题与瓶颈

### 🔴 核心问题：音质不达标（电流声/失真/不像本人）

**表现**：训练完成后试听，输出有明显电流感或金属感，音色还原度差。

**根本原因（已确认）**：

| 瓶颈 | 说明 |
|------|------|
| **模型太小** | FastSpeech2-lite (~10M) 和 VITS-lite (~18M) 参数量远低于业界标准。商业级 TTS（GPT-SoVITS、XTTS 等）参数量在 **几亿到几十亿**，且经过数万小时音频预训练。 |
| **数据太少** | 本项目需要从头训练（无预训练），仅凭 **5 分钟~16 段样本** 无法让模型学会人类语音的复杂规律（发音、韵律、音色特征）。业界推荐至少 **几十分钟到几小时** 的高质量数据。 |
| **架构限制** | 轻量模型设计目标是在低资源设备上快速运行，而非高保真克隆。mel 谱预测精度有限，即使声码器（HiFi-GAN）正常，输入的 mel 本身就是失真的。 |

> **与商业软件的对比**：视频软件（ElevenLabs、HeyGen 等）使用的是预训练大模型。用户提供的 1 分钟音频仅用于"微调"或"条件化"，底层模型已经学过人类语音的普遍规律。本项目无预训练，必须从零学习，门槛完全不同。

### 🟡 技术矛盾

| 需求 | 可行性 |
|------|--------|
| 导出 ONNX 给 LiveBuddy | ✅ 本项目支持 |
| 高质量音色克隆 | ❌ 需要大模型（GPT-SoVITS 等） |
| **两者同时满足** | ❌ **不可行**。大模型（GPT-SoVITS）无法导出 ONNX；能导出 ONNX 的轻量模型质量不达标。 |

---

## 调试记录

### 已完成的修复与优化

| # | 问题 | 解决方案 | 状态 |
|---|------|---------|------|
| 1 | 默认使用 AMD 集显（512MB）而非 RX 9070 XT 独显（16GB） | 修改 `device.py`，遍历 DirectML 设备，自动跳过集显关键词（Graphics），优先选择 RX 系列独显 | ✅ 已解决 |
| 2 | `torch.onnx.export` 报错 `unexpected keyword argument 'external_data'` | 删除 `export_onnx.py` 中 4 处 `external_data=False` 参数 | ✅ 已解决 |
| 3 | 推理使用 Griffin-Lim（电流声），未加载用户训练的 HiFi-GAN | 1) 在 `train.py` 中 FastSpeech2/VITS 训练后追加 HiFi-GAN Generator 训练<br>2) `vocoder.py` 优先从 checkpoint 加载 `hifigan_state_dict`<br>3) `server.py` 推理时传入 `checkpoint_path` | ✅ 已解决 |
| 4 | `torch.load` 在 DirectML 设备上报错 `'>=' not supported` | `vocoder.py` 中 `torch.load` 改用 `map_location="cpu"`，再 `.to(device)` | ✅ 已解决 |
| 5 | ONNX 推理报错 `Reshape` 节点 shape 不匹配 `{226,1,256} != {32,4,64}` | 根本原因是 `nn.MultiheadAttention` 的 ONNX 导出 bug。最终方案：用纯手写的 `SimpleMHA` 完全替换 `nn.MultiheadAttention`，权重键名与 PyTorch 原生兼容，无需重新训练。 | ✅ 已解决 |
| 6 | Batch Size 选择困惑 | 建议：RX 9070 XT 16GB 显存，选 **2** 最安全（比 1 快近一倍，比 3/4 更稳） | ✅ 已指导 |

### 尝试过的方案（未解决根本问题）

| # | 方案 | 结果 | 原因 |
|---|------|------|------|
| 1 | 切换到 GPT-SoVITS 大模型 | ❌ 无法导出 ONNX | GPT-SoVITS 是预训练大模型 + 参考音频的架构，输出不能转为 LiveBuddy 可用的 ONNX |
| 2 | 用 GPT-SoVITS "造数据"再蒸馏到 VITS | ⏸️ 未实施 | 方案可行但复杂度高（需批量生成 1000+ 句合成音频，再训练），用户选择先用 VITS 直接训练评估 |
| 3 | 增加训练 epoch 到 500 | ⚠️ 有改善但不解决 | Loss 收敛到平台期后不再下降，模型容量是硬限制 |

---

## 技术架构

- **前端**：纯 HTML/CSS/JS 单页面应用
- **后端**：Python Flask
- **模型**：PyTorch 训练 → ONNX 导出
- **推理**：onnxruntime（支持 CPU/CUDA/DirectML）
- **新增模块**：
  - `trainer/gpt_sovits_wrapper.py`：GPT-SoVITS 推理封装
  - `setup_gpt_sovits.py`：GPT-SoVITS 一键安装脚本（安装到 D 盘）
  - `export_existing.py`：手动重新导出已训练模型的 ONNX
  - `cleanup.py`：清理 uploads 和 models 缓存

---

## 文件变更概览

### 修改的文件（git tracked）
- `index.html`：添加 GPT-SoVITS 试听按钮，默认模型改为 VITS
- `server.py`：添加 `/api/try-voice-gpt` 接口，推理传入 checkpoint_path
- `trainer/export_onnx.py`：修复 `external_data`，提高 opset_version 到 17
- `trainer/fastspeech2.py`：添加 `SimpleMHA`，替换 `nn.MultiheadAttention`
- `trainer/train.py`：FastSpeech2/VITS 训练后追加 HiFi-GAN 训练
- `trainer/vits_lite.py`：添加 `SimpleMHA`，替换 `nn.MultiheadAttention`

### 新增的文件（untracked）
- `cleanup.py`
- `export_existing.py`
- `setup_gpt_sovits.py`
- `trainer/gpt_sovits_wrapper.py`

---

## 下一步建议（如需继续）

### 路线 A：接受当前架构，优化数据
- **录 50~100 段**不同文本的音频（每段 5~15 秒），覆盖尽可能多的字词组合
- 使用 **VITS** 模型，Batch Size=2，Epoch=500
- 预期效果：**可用但仍有 AI 感**，不会像真人

### 路线 B：换架构，放弃 ONNX 导出
- 使用 **GPT-SoVITS** 或 **XTTS v2** 等预训练大模型
- 1~10 分钟音频即可达到高质量克隆
- 缺点：只能在特定环境中使用，**无法导出给 LiveBuddy**

### 路线 C：数据增强 + 蒸馏（最复杂但最有希望）
1. 用 GPT-SoVITS + 参考音频生成 **1000+ 句**不同文本的合成语音
2. 把这些"假数据"作为训练集，训练 VITS-lite
3. 导出 ONNX
4. 预期效果：**明显优于路线 A**，接近路线 B

---

## 许可证

MIT License
