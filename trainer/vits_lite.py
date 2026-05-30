"""简化 VITS 模型：端到端概率 TTS"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """正弦位置编码（支持动态扩展超长序列）。"""

    def __init__(self, d_model, max_len=3000):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        pe = self._build_pe(max_len)
        self.register_buffer("pe", pe.unsqueeze(0))

    def _build_pe(self, length):
        pe = torch.zeros(length, self.d_model)
        position = torch.arange(0, length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, x):
        T = x.size(1)
        if T > self.pe.size(1):
            new_pe = self._build_pe(T).to(x.device)
            self.pe = new_pe.unsqueeze(0)
        return x + self.pe[:, :T]


class TransformerBlock(nn.Module):
    def __init__(self, d_model=256, n_head=4, d_ff=1024, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        residual = x
        x = self.norm1(x)
        attn_out, _ = self.self_attn(x, x, x, key_padding_mask=mask)
        x = residual + self.dropout(attn_out)
        residual = x
        x = self.norm2(x)
        x = residual + self.dropout(self.ff(x))
        return x


class TextEncoder(nn.Module):
    """文本编码器：Embedding + 2 层 Transformer。"""

    def __init__(self, vocab_size, d_model=256, n_head=4, d_ff=1024,
                 n_layers=2, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_head, d_ff, dropout) for _ in range(n_layers)
        ])
        self.proj = nn.Linear(d_model, d_model * 2)  # mean + logvar

    def forward(self, x, mask=None):
        x = self.embedding(x) * math.sqrt(256)
        x = self.pos(x)
        for layer in self.layers:
            x = layer(x, mask)
        stats = self.proj(x)
        mean, logvar = stats.chunk(2, dim=-1)
        return x, mean, logvar


class PosteriorEncoder(nn.Module):
    """后验编码器：从 mel 提取 latent（训练时使用）。"""

    def __init__(self, n_mels=80, d_model=256, n_layers=3):
        super().__init__()
        self.pre = nn.Conv1d(n_mels, d_model, 1)
        self.convs = nn.ModuleList()
        for i in range(n_layers):
            dilation = 2 ** i
            self.convs.append(nn.Sequential(
                nn.Conv1d(d_model, d_model, 3, dilation=dilation, padding=dilation),
                nn.LeakyReLU(0.1),
            ))
        self.proj = nn.Conv1d(d_model, d_model * 2, 1)  # mean + logvar

    def forward(self, mel):
        """
        Args:
            mel: (B, n_mels, T_mel)
        Returns:
            z: (B, d_model, T_mel)
            mean: (B, d_model, T_mel)
            logvar: (B, d_model, T_mel)
        """
        x = self.pre(mel)
        for conv in self.convs:
            x = x + conv(x)
        stats = self.proj(x)
        mean, logvar = stats.chunk(2, dim=1)
        z = mean + torch.randn_like(mean) * torch.exp(0.5 * logvar)
        return z, mean, logvar


class AffineCouplingLayer(nn.Module):
    """仿射耦合层（flow 组件）。"""

    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels // 2, channels, kernel_size, padding=kernel_size // 2),
            nn.LeakyReLU(0.1),
            nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2),
            nn.LeakyReLU(0.1),
            nn.Conv1d(channels, channels, 1),
        )

    def forward(self, x, reverse=False):
        x1, x2 = x.chunk(2, dim=1)
        h = self.net(x1)
        log_scale, shift = h.chunk(2, dim=1)
        log_scale = torch.tanh(log_scale)

        if not reverse:
            x2 = x2 * torch.exp(log_scale) + shift
            log_det = log_scale.sum(dim=[1, 2])
        else:
            x2 = (x2 - shift) * torch.exp(-log_scale)
            log_det = -log_scale.sum(dim=[1, 2])

        return torch.cat([x1, x2], dim=1), log_det


class Flow(nn.Module):
    """简化 normalizing flow：2 层耦合层 + 1x1 Conv。"""

    def __init__(self, channels=256, n_layers=2):
        super().__init__()
        self.flows = nn.ModuleList()
        for _ in range(n_layers):
            self.flows.append(nn.Conv1d(channels, channels, 1))
            self.flows.append(AffineCouplingLayer(channels))

    def forward(self, z, reverse=False):
        log_det_total = 0
        if not reverse:
            for flow in self.flows:
                if isinstance(flow, AffineCouplingLayer):
                    z, log_det = flow(z, reverse=False)
                    log_det_total = log_det_total + log_det
                else:
                    z = flow(z)
        else:
            for flow in reversed(self.flows):
                if isinstance(flow, AffineCouplingLayer):
                    z, log_det = flow(z, reverse=True)
                else:
                    z = torch.linalg.solve(flow.weight.squeeze(-1).T.unsqueeze(0),
                                            z.transpose(1, 2)).transpose(1, 2)
        return z, log_det_total


class DurationPredictor(nn.Module):
    """简化时长预测器：2 层 CNN。"""

    def __init__(self, d_model=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(d_model, 1, 3, padding=1),
        )

    def forward(self, x):
        """x: (B, T, D) → (B, T, 1)"""
        return self.conv(x.transpose(1, 2)).transpose(1, 2)


class LengthRegulator(nn.Module):
    """长度调节器：使用 torch.repeat_interleave 兼容 torch.export。"""

    def forward(self, x, duration):
        B, T, D = x.shape
        duration = duration.long().clamp(min=1)
        outputs = []
        for b in range(B):
            expanded = torch.repeat_interleave(x[b], duration[b], dim=0)
            outputs.append(expanded)
        if B == 1:
            return outputs[0].unsqueeze(0)
        max_len = max(o.size(0) for o in outputs)
        padded = torch.zeros(B, max_len, D, device=x.device, dtype=x.dtype)
        for b, o in enumerate(outputs):
            padded[b, :o.size(0)] = o
        return padded


class VITSLite(nn.Module):
    """简化 VITS 模型。

    架构：
    - Text Encoder: 2 层 Transformer
    - Posterior Encoder: 3 层 dilated CNN
    - Flow: 2 层 Affine Coupling + 1x1 Conv
    - Duration Predictor: 2 层 CNN
    - Decoder: 线性投影 + HiFi-GAN (外部)

    参数量约 15-20M（不含 HiFi-GAN decoder）
    """

    def __init__(self, vocab_size: int, d_model: int = 256, n_head: int = 4,
                 d_ff: int = 1024, n_mels: int = 80, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_mels = n_mels

        self.text_encoder = TextEncoder(vocab_size, d_model, n_head, d_ff, 2, dropout)
        self.posterior_encoder = PosteriorEncoder(n_mels, d_model, 3)
        self.flow = Flow(d_model, 2)
        self.duration_predictor = DurationPredictor(d_model)
        self.length_regulator = LengthRegulator()

        # 从 latent 到 mel 的投影
        self.proj_to_mel = nn.Linear(d_model, n_mels)

    def forward(self, text_seq, mel_target=None, durations=None, src_mask=None):
        """
        Args:
            text_seq: (B, T_text)
            mel_target: (B, n_mels, T_mel) 训练时提供
            durations: (B, T_text) 训练时提供
            src_mask: padding mask

        Returns:
            dict
        """
        # Text encoding
        encoder_out, z_mean, z_logvar = self.text_encoder(text_seq, src_mask)

        # Duration
        duration_pred = self.duration_predictor(encoder_out)

        # Length regulation
        if durations is not None:
            dur = durations.float().unsqueeze(-1) if durations.dim() == 2 else durations
            if dur.dim() == 3:
                dur = dur.squeeze(-1)
            expanded = self.length_regulator(encoder_out, dur)
        else:
            dur_pred = torch.clamp(duration_pred.squeeze(-1).round(), min=1)
            expanded = self.length_regulator(encoder_out, dur_pred)

        # Flow: prior → z_p
        expanded_t = expanded.transpose(1, 2)  # (B, D, T)
        z_p, _ = self.flow(expanded_t, reverse=False)

        # Posterior (训练时)
        z_q, q_mean, q_logvar = None, None, None
        if mel_target is not None:
            z_q, q_mean, q_logvar = self.posterior_encoder(mel_target)

        # Project to mel
        mel_pred = self.proj_to_mel(expanded)  # (B, T_mel, n_mels)

        return {
            "mel_pred": mel_pred,
            "z_p": z_p,
            "z_q": z_q,
            "z_mean": z_mean,
            "z_logvar": z_logvar,
            "q_mean": q_mean,
            "q_logvar": q_logvar,
            "duration_pred": duration_pred,
        }


class VITSLiteInference(nn.Module):
    """VITS-lite ONNX 导出包装器。

    接口：text_seq → mel
    """

    def __init__(self, model: VITSLite):
        super().__init__()
        self.model = model

    def forward(self, text_seq):
        result = self.model(text_seq)
        return result["mel_pred"]
