import torch
import os
import json


def compute_metrics(
    denoised: torch.Tensor,
    clean: torch.Tensor,
) -> dict:
    """
    计算ECG信号去噪评估指标:
    RMSE, PRD, SNR
    """

    split_dir = "./data_split"

    # 加载分割信息
    split_path = os.path.join(split_dir, "split_info.json")
    if not os.path.exists(split_path):
        raise FileNotFoundError(f"Split file not found: {split_path}")

    with open(split_path, "r") as f:
        split_data = json.load(f)

    # 加载数据文件
    mean, std = split_data["clean_mean"], split_data["clean_std"]
    mean = torch.tensor(mean, device=clean.device, dtype=clean.dtype)
    std = torch.tensor(std, device=clean.device, dtype=clean.dtype)

    clean = clean.permute(0, 2, 1)  # (batch, window_size, channels)
    denoised = denoised.permute(0, 2, 1)  # (batch, window_size, channels)

    # 反标准化
    clean = clean * std + mean
    denoised = denoised * std + mean

    clean = clean.reshape(clean.shape[0], -1)  # (batch, window_size * channels)
    denoised = denoised.reshape(
        denoised.shape[0], -1
    )  # (batch, window_size * channels)

    # RMSE
    rmse = torch.sqrt(torch.mean((clean - denoised) ** 2, dim=1))  # (batch,)
    rmse_mean = torch.mean(rmse).item()

    # PRD
    prd = (
        torch.sqrt(torch.sum((clean - denoised) ** 2, dim=1))
        / (torch.sqrt(torch.sum(clean**2, dim=1)))
        * 100
    )  # (batch,)
    prd_mean = torch.mean(prd).item()

    # SNR
    noise_power = torch.mean((clean - denoised) ** 2, dim=1)
    signal_power = torch.mean(clean**2, dim=1)
    SNR = 10 * torch.log10((signal_power) / (noise_power))  # (batch,)
    SNR_mean = torch.mean(SNR).item()

    return {"RMSE": rmse_mean, "PRD": prd_mean, "SNR": SNR_mean}
