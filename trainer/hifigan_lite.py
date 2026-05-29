"""轻量 HiFi-GAN vocoder：mel → 波形"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class ResBlock(nn.Module):
    """Multi-Reception Field Fusion 残差块。"""

    def __init__(self, channels, kernel_size=3, dilations=(1, 3, 5)):
        super().__init__()
        self.convs = nn.ModuleList()
        for d in dilations:
            self.convs.append(nn.Sequential(
                nn.LeakyReLU(0.1),
                nn.Conv1d(channels, channels, kernel_size,
                          dilation=d, padding=(kernel_size * d - d) // 2),
                nn.LeakyReLU(0.1),
                nn.Conv1d(channels, channels, kernel_size,
                          padding=kernel_size // 2),
            ))

    def forward(self, x):
        for conv in self.convs:
            x = x + conv(x)
        return x


class HiFiGANGenerator(nn.Module):
    """轻量 HiFi-GAN 生成器。

    将 mel spectrogram 上采样为波形。
    上采样率: 8 * 8 * 2 * 2 = 256 (对应 hop_length=256)
    """

    def __init__(self, n_mels: int = 80, upsample_initial_channel: int = 256,
                 upsample_rates=(8, 8, 2, 2),
                 upsample_kernel_sizes=(16, 16, 4, 4),
                 resblock_kernel_sizes=(3, 7, 11),
                 resblock_dilations=((1, 3), (1, 3), (1, 3))):
        super().__init__()

        self.num_upsamples = len(upsample_rates)

        # 输入卷积
        self.conv_pre = nn.Conv1d(n_mels, upsample_initial_channel, 7, padding=3)

        # 上采样层
        self.ups = nn.ModuleList()
        ch = upsample_initial_channel
        for i, (rate, k_size) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            ch_next = ch // 2
            self.ups.append(nn.Sequential(
                nn.LeakyReLU(0.1),
                nn.ConvTranspose1d(ch, ch_next, k_size, stride=rate,
                                    padding=(k_size - rate) // 2),
            ))
            ch = ch_next

        # MRF 残差块（每个上采样层后）
        self.resblocks = nn.ModuleList()
        for i in range(self.num_upsamples):
            ch_at_level = upsample_initial_channel // (2 ** (i + 1))
            for k, d in zip(resblock_kernel_sizes, resblock_dilations):
                self.resblocks.append(ResBlock(ch_at_level, k, d))

        # 输出卷积
        ch_final = upsample_initial_channel // (2 ** self.num_upsamples)
        self.conv_post = nn.Conv1d(ch_final, 1, 7, padding=3)

    def forward(self, mel):
        """
        Args:
            mel: (B, n_mels, T_mel) log mel spectrogram

        Returns:
            wav: (B, 1, T_wav) 波形
        """
        x = self.conv_pre(mel)

        resblock_idx = 0
        for i in range(self.num_upsamples):
            x = self.ups[i](x)
            # MRF: 所有残差块输出求和平均
            xs = None
            for j in range(len(self.resblock_kernel_sizes) if hasattr(self, 'resblock_kernel_sizes') else 3):
                if resblock_idx < len(self.resblocks):
                    if xs is None:
                        xs = self.resblocks[resblock_idx](x)
                    else:
                        xs = xs + self.resblocks[resblock_idx](x)
                    resblock_idx += 1
            if xs is not None:
                x = xs / 3.0

        x = F.leaky_relu(x, 0.1)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return x

    # 为了让 resblock_kernel_sizes 在 forward 中可访问
    resblock_kernel_sizes = (3, 7, 11)


class HiFiGANDiscriminator(nn.Module):
    """多周期判别器 (MPD) + 多尺度判别器 (MSD)。"""

    def __init__(self):
        super().__init__()
        self.mpd = MultiPeriodDiscriminator()
        self.msd = MultiScaleDiscriminator()

    def forward(self, real, fake):
        """
        Returns:
            (mpd_out_real, mpd_out_fake, msd_out_real, msd_out_fake,
             mpd_feats_real, mpd_feats_fake, msd_feats_real, msd_feats_fake)
        """
        mpd_r, mpd_fr = self.mpd(real)
        mpd_f, mpd_ff = self.mpd(fake)
        msd_r, msd_fr = self.msd(real)
        msd_f, msd_ff = self.msd(fake)
        return mpd_r, mpd_f, msd_r, msd_f, mpd_fr, mpd_ff, msd_fr, msd_ff


class PeriodDiscriminator(nn.Module):
    """单周期判别器。"""

    def __init__(self, period, kernel_size=5, stride=3):
        super().__init__()
        self.period = period
        self.convs = nn.ModuleList()
        ch = 1
        for out_ch in [32, 128, 512, 512]:
            self.convs.append(nn.Sequential(
                nn.Conv2d(ch, out_ch, (kernel_size, 1), (stride, 1),
                          padding=(kernel_size // 2, 0)),
                nn.LeakyReLU(0.1),
            ))
            ch = out_ch
        self.conv_post = nn.Conv2d(ch, 1, (3, 1), padding=(1, 0))

    def forward(self, x):
        fmap = []
        B, C, T = x.shape
        # Reshape to 2D
        if T % self.period != 0:
            pad_len = self.period - (T % self.period)
            x = F.pad(x, (0, pad_len), mode="reflect")
            T = T + pad_len
        x = x.view(B, C, T // self.period, self.period)

        for conv in self.convs:
            x = conv(x)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        return x, fmap


class MultiPeriodDiscriminator(nn.Module):
    """多周期判别器。"""

    def __init__(self, periods=(2, 3, 5)):
        super().__init__()
        self.discriminators = nn.ModuleList([PeriodDiscriminator(p) for p in periods])

    def forward(self, x):
        ret_real = []
        ret_fake_fmaps = []
        for d in self.discriminators:
            out, fmap = d(x)
            ret_real.append(out)
            ret_fake_fmaps.append(fmap)
        return ret_real, ret_fake_fmaps


class ScaleDiscriminator(nn.Module):
    """单尺度判别器。"""

    def __init__(self, use_spectral_norm=False):
        super().__init__()
        norm_fn = nn.utils.spectral_norm if use_spectral_norm else nn.utils.weight_norm
        self.convs = nn.ModuleList([
            norm_fn(nn.Conv1d(1, 64, 15, 1, padding=7)),
            norm_fn(nn.Conv1d(64, 128, 41, 2, groups=4, padding=20)),
            norm_fn(nn.Conv1d(128, 256, 41, 2, groups=16, padding=20)),
            norm_fn(nn.Conv1d(256, 512, 41, 4, groups=16, padding=20)),
            norm_fn(nn.Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
            norm_fn(nn.Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
            norm_fn(nn.Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = norm_fn(nn.Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []
        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, 0.1)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        return x, fmap


class MultiScaleDiscriminator(nn.Module):
    """多尺度判别器（简化为 2 个尺度）。"""

    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList([
            ScaleDiscriminator(use_spectral_norm=True),
            ScaleDiscriminator(),
        ])
        self.pooling = nn.ModuleList([
            nn.AvgPool1d(4, 2, padding=2),
        ])

    def forward(self, x):
        ret = []
        fmaps = []
        for i, d in enumerate(self.discriminators):
            if i > 0:
                x = self.pooling[i - 1](x)
            out, fmap = d(x)
            ret.append(out)
            fmaps.append(fmap)
        return ret, fmaps
