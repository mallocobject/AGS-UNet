import torch


def compute_metrics(
    denoised: torch.Tensor, clean: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
) -> dict:
    """
    计算 ECG 信号去噪指标：
    RMSE、PRD、SNR(batch 维度求平均)
    """

    mean = mean.permute(0, 2, 1)  # (1, C, 1)
    std = std.permute(0, 2, 1)  # (1, C, 1)

    # === 保证形状匹配 ===
    # 输入通常是 (batch, channels, length)
    if clean.ndim == 2:
        clean = clean.unsqueeze(1)
        denoised = denoised.unsqueeze(1)

    # 反标准化
    denoised = denoised * std + mean

    # 保证浮点精度（避免半精度误差）
    clean = clean.float()
    denoised = denoised.float()

    # === 计算指标 ===
    diff = clean - denoised

    # RMSE（每样本取均值）
    rmse = torch.sqrt(torch.mean(diff**2, dim=[1, 2]))  # (batch,)
    rmse_mean = rmse.mean().item()

    # SNR（dB）
    noise_power = torch.mean(diff**2, dim=[1, 2])
    signal_power = torch.mean(clean**2, dim=[1, 2])
    snr = 10 * torch.log10(signal_power / (noise_power))
    snr_mean = snr.mean().item()

    return {"RMSE": rmse_mean, "SNR": snr_mean}
