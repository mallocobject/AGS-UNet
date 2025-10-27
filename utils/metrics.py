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
    if clean.dim() == 1:
        clean = clean.unsqueeze(0)
        denoised = denoised.unsqueeze(0)
        noisy = noisy.unsqueeze(0)

    if clean.dim() == 3:
        clean = clean.squeeze(1)
        denoised = denoised.squeeze(1)
        noisy = noisy.squeeze(1)

    split_dir = "./data_split"

    # 加载分割信息
    split_path = os.path.join(split_dir, "split_info.json")
    if not os.path.exists(split_path):
        raise FileNotFoundError(f"Split file not found: {split_path}")

    with open(split_path, "r") as f:
        split_data = json.load(f)

    # 加载数据文件
    mean, std = split_data["clean_mean"], split_data["clean_std"]

    # 反标准化
    clean = clean * std + mean
    denoised = denoised * std + mean

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
