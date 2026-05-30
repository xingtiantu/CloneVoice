"""数据增强：SpecAugment（时域掩码 + 频域掩码）"""
import torch
import logging

logger = logging.getLogger(__name__)


class SpecAugment:
    """对 mel 谱进行随机掩码，提升模型泛化能力。

    参考：Park et al. "SpecAugment: A Simple Data Augmentation Method for ASR", 2019.
    """

    def __init__(
        self,
        freq_mask_param: int = 10,
        time_mask_param: int = 20,
        n_freq_masks: int = 1,
        n_time_masks: int = 1,
        p: float = 0.5,
    ):
        """
        Args:
            freq_mask_param: 频域掩码最大宽度（mel channel 数）
            time_mask_param: 时域掩码最大宽度（帧数）
            n_freq_masks: 每次应用频域掩码的数量
            n_time_masks: 每次应用时域掩码的数量
            p: 应用增强的概率
        """
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks
        self.p = p

    def __call__(self, mel: torch.Tensor) -> torch.Tensor:
        """对 mel 谱应用 SpecAugment。

        Args:
            mel: (n_mels, T) 或 (B, n_mels, T)

        Returns:
            增强后的 mel（与输入同 shape）
        """
        if torch.rand(1).item() > self.p:
            return mel

        squeeze = False
        if mel.dim() == 2:
            mel = mel.unsqueeze(0)
            squeeze = True

        B, n_mels, T = mel.shape

        # 频域掩码
        for _ in range(self.n_freq_masks):
            f = torch.randint(1, self.freq_mask_param + 1, (1,)).item()
            f = min(f, n_mels)
            f0 = torch.randint(0, n_mels - f + 1, (1,)).item()
            mel[:, f0 : f0 + f, :] = mel.mean()

        # 时域掩码
        for _ in range(self.n_time_masks):
            t = torch.randint(1, self.time_mask_param + 1, (1,)).item()
            t = min(t, T)
            t0 = torch.randint(0, T - t + 1, (1,)).item()
            mel[:, :, t0 : t0 + t] = mel.mean()

        if squeeze:
            mel = mel.squeeze(0)

        return mel
