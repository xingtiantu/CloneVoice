"""FastSpeech2-lite 模型：轻量级非自回归 TTS"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """正弦位置编码。"""

    def __init__(self, d_model, max_len=3000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        """x: (B, T, D)"""
        return x + self.pe[:, :x.size(1)]


class TransformerBlock(nn.Module):
    """单层 Transformer encoder block。"""

    def __init__(self, d_model=256, n_head=4, d_ff=1024, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Args:
            x: (B, T, D)
            mask: optional attention mask
        """
        # Self-attention
        residual = x
        x = self.norm1(x)
        attn_out, _ = self.self_attn(x, x, x, key_padding_mask=mask)
        x = residual + self.dropout(attn_out)

        # FFN
        residual = x
        x = self.norm2(x)
        x = residual + self.dropout(self.ff(x))
        return x


class VariancePredictor(nn.Module):
    """方差预测器（Duration/Pitch/Energy）。

    2 层 1D-CNN + ReLU + LayerNorm + Linear
    """

    def __init__(self, d_model=256, channels=256, kernel_size=3, dropout=0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(d_model, channels, kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.BatchNorm1d(channels),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.BatchNorm1d(channels),
            nn.Dropout(dropout),
        )
        self.linear = nn.Linear(channels, 1)

    def forward(self, x):
        """
        Args:
            x: (B, T, D)
        Returns:
            (B, T, 1) 预测值
        """
        x = x.transpose(1, 2)  # (B, D, T)
        x = self.conv(x)
        x = x.transpose(1, 2)  # (B, T, C)
        x = self.linear(x)
        return x


class LengthRegulator(nn.Module):
    """长度调节器：根据 duration 展开 hidden states。

    使用 torch.repeat_interleave 替代 Python 循环，兼容 torch.export / ONNX dynamo 导出。
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, duration):
        """
        Args:
            x: (B, T_text, D)
            duration: (B, T_text) 每个位置的帧数（整数或浮点）

        Returns:
            (B, T_mel, D)
        """
        B, T, D = x.shape
        duration = duration.long().clamp(min=1)

        # torch.repeat_interleave 兼容 torch.export / ONNX dynamo 导出
        outputs = []
        for b in range(B):
            expanded = torch.repeat_interleave(x[b], duration[b], dim=0)
            outputs.append(expanded)

        if B == 1:
            return outputs[0].unsqueeze(0)

        # Pad to same length (for B > 1)
        max_len = max(o.size(0) for o in outputs)
        padded = torch.zeros(B, max_len, D, device=x.device, dtype=x.dtype)
        for b, o in enumerate(outputs):
            padded[b, :o.size(0)] = o
        return padded


class PostNet(nn.Module):
    """后处理网络：5 层 1D-CNN 残差连接。"""

    def __init__(self, n_mels=80, channels=256, kernel_size=5, n_layers=5):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(nn.Sequential(
            nn.Conv1d(n_mels, channels, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(channels),
            nn.Tanh(),
            nn.Dropout(0.5),
        ))
        for _ in range(n_layers - 2):
            self.convs.append(nn.Sequential(
                nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(channels),
                nn.Tanh(),
                nn.Dropout(0.5),
            ))
        self.convs.append(nn.Sequential(
            nn.Conv1d(channels, n_mels, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(n_mels),
            nn.Dropout(0.5),
        ))

    def forward(self, mel):
        """
        Args:
            mel: (B, T, n_mels)
        Returns:
            (B, T, n_mels) 残差修正后的 mel
        """
        x = mel.transpose(1, 2)  # (B, n_mels, T)
        for conv in self.convs:
            x = conv(x)
        x = x.transpose(1, 2)  # (B, T, n_mels)
        return mel + x


class FastSpeech2Lite(nn.Module):
    """FastSpeech2-lite 轻量级非自回归 TTS 模型。

    架构：
    - Encoder: 2 层 Transformer (256-dim, 4-head)
    - Variance Adaptor: Duration + Pitch + Energy predictor + Length regulator
    - Decoder: 2 层 Transformer
    - Mel Linear + PostNet

    参数量约 8-12M
    """

    def __init__(self, vocab_size: int, d_model: int = 256, n_head: int = 4,
                 d_ff: int = 1024, n_encoder_layers: int = 2,
                 n_decoder_layers: int = 2, n_mels: int = 80,
                 max_seq_len: int = 3000, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_mels = n_mels

        # Embedding
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)

        # Encoder
        self.encoder = nn.ModuleList([
            TransformerBlock(d_model, n_head, d_ff, dropout)
            for _ in range(n_encoder_layers)
        ])

        # Variance Adaptor
        self.duration_predictor = VariancePredictor(d_model, d_model, 3, dropout)
        self.pitch_predictor = VariancePredictor(d_model, d_model, 3, dropout)
        self.energy_predictor = VariancePredictor(d_model, d_model, 3, dropout)

        self.pitch_embedding = nn.Embedding(256, d_model)  # pitch bins
        self.energy_embedding = nn.Embedding(256, d_model)  # energy bins

        self.length_regulator = LengthRegulator()

        # Decoder
        self.decoder = nn.ModuleList([
            TransformerBlock(d_model, n_head, d_ff, dropout)
            for _ in range(n_decoder_layers)
        ])

        # Output
        self.mel_linear = nn.Linear(d_model, n_mels)
        self.postnet = PostNet(n_mels, d_model)

    def forward(self, text_seq, durations=None, pitches=None, energies=None,
                src_mask=None, mel_len_target=None):
        """
        Args:
            text_seq: (B, T_text) 输入文本序列
            durations: (B, T_text) 真实 duration（训练时提供）
            pitches: (B, T_mel) 真实 pitch（训练时提供）
            energies: (B, T_mel) 真实 energy（训练时提供）
            src_mask: (B, T_text) padding mask
            mel_len_target: int，目标 mel 长度

        Returns:
            dict with keys:
                mel_pred: (B, T_mel, n_mels) 预测的 mel
                mel_postnet: (B, T_mel, n_mels) PostNet 修正后的 mel
                duration_pred: (B, T_text, 1) 预测的 duration
                pitch_pred: (B, T_text, 1) 预测的 pitch
                energy_pred: (B, T_text, 1) 预测的 energy
        """
        # Embedding + Position
        x = self.embedding(text_seq) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)

        # Encoder
        for layer in self.encoder:
            x = layer(x, mask=src_mask)

        encoder_out = x  # (B, T_text, D)

        # Duration prediction
        duration_pred = self.duration_predictor(encoder_out)  # (B, T_text, 1)

        if durations is not None:
            # 训练时：用真实 duration 做 length regulation
            duration_for_lr = durations.float().unsqueeze(-1) if durations.dim() == 2 else durations
            if duration_for_lr.dim() == 3:
                duration_for_lr = duration_for_lr.squeeze(-1)
            x = self.length_regulator(encoder_out, duration_for_lr)
        else:
            # 推理时：用预测的 duration
            duration_for_lr = torch.clamp(duration_pred.squeeze(-1).round(), min=1)
            x = self.length_regulator(encoder_out, duration_for_lr)

        # T_mel now = x.size(1)

        # Pitch & Energy prediction（在 encoder 输出上预测）
        pitch_pred = self.pitch_predictor(encoder_out)  # (B, T_text, 1)
        energy_pred = self.energy_predictor(encoder_out)  # (B, T_text, 1)

        # 如果有真实 pitch/energy，在 regulated 后的序列上添加
        if durations is not None and pitches is not None:
            # pitch/energy 需要扩展到 mel 长度
            pitches_expanded = pitches.unsqueeze(-1) if pitches.dim() == 2 else pitches
            energies_expanded = energies.unsqueeze(-1) if energies.dim() == 2 else energies
            # 量化到 bins
            pitch_bins = torch.clamp((pitches_expanded * 128 + 128).long(), 0, 255)
            energy_bins = torch.clamp((energies_expanded * 128 + 128).long(), 0, 255)
            # 确保长度匹配
            T_mel = x.size(1)
            if pitch_bins.size(1) > T_mel:
                pitch_bins = pitch_bins[:, :T_mel]
            elif pitch_bins.size(1) < T_mel:
                pitch_bins = F.pad(pitch_bins, (0, 0, 0, T_mel - pitch_bins.size(1)))
            if energy_bins.size(1) > T_mel:
                energy_bins = energy_bins[:, :T_mel]
            elif energy_bins.size(1) < T_mel:
                energy_bins = F.pad(energy_bins, (0, 0, 0, T_mel - energy_bins.size(1)))
            x = x + self.pitch_embedding(pitch_bins.squeeze(-1)) + self.energy_embedding(energy_bins.squeeze(-1))

        # Position encoding for decoder
        x = self.pos_encoding(x)

        # Decoder
        for layer in self.decoder:
            x = layer(x)

        # Mel output
        mel_pred = self.mel_linear(x)  # (B, T_mel, n_mels)
        mel_postnet = self.postnet(mel_pred)  # (B, T_mel, n_mels)

        return {
            "mel_pred": mel_pred,
            "mel_postnet": mel_postnet,
            "duration_pred": duration_pred,
            "pitch_pred": pitch_pred,
            "energy_pred": energy_pred,
        }


class FastSpeech2Inference(nn.Module):
    """FastSpeech2 ONNX 导出用推理包装器。

    简化接口：只接收 text_seq，输出 mel。
    将 encoder + variance adaptor + decoder + postnet 打包。
    """

    def __init__(self, model: FastSpeech2Lite):
        super().__init__()
        self.model = model

    def forward(self, text_seq):
        """
        Args:
            text_seq: (1, T_text) int64

        Returns:
            mel: (1, T_mel, n_mels) float32
        """
        result = self.model(text_seq)
        return result["mel_postnet"]

