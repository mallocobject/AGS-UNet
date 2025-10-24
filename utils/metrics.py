import torch


def compute_metrics(
    denoised: torch.Tensor, clean: torch.Tensor, eps: float = 1e-8
) -> dict:
    """
    计算ECG信号去噪的评估指标, 包括MAE, PCC, and SNRI(Signal-to-Noise Ratio Improvement).
    """
    assert clean.shape == denoised.shape, "输入张量形状必须相同"

    if clean.dim() == 1:
        clean = clean.unsqueeze(0)
        denoised = denoised.unsqueeze(0)

    batch_size, length = clean.shape

    # MAE
    mae = torch.mean(torch.abs(clean - denoised), dim=-1)  # (batch,)
    mae = torch.mean(mae).item()  # 平均 MAE

    # PCC
    mean_clean = torch.mean(clean, dim=-1, keepdim=True)  # (batch, 1)
    mean_denoised = torch.mean(denoised, dim=-1, keepdim=True)  # (batch, 1)
    clean_centered = clean - mean_clean  # (batch, length)
    denoised_centered = denoised - mean_denoised  # (batch, length)

    numerator = torch.sum(clean_centered * denoised_centered, dim=-1)  # (batch,)
    denom_clean = torch.sqrt(torch.sum(clean_centered**2, dim=-1) + eps)  # (batch,)
    denom_denoised = torch.sqrt(
        torch.sum(denoised_centered**2, dim=-1) + eps
    )  # (batch,)
    pcc = numerator / (denom_clean * denom_denoised + eps)  # (batch,)
    pcc = torch.mean(pcc).item()  # 平均 PCC

    # SNRI
    signal_power = torch.sum(clean**2, dim=-1)  # (batch,)
    noise_power = torch.sum((clean - denoised) ** 2, dim=-1)  # (batch,)
    snri = 10 * torch.log10(signal_power / (noise_power + eps))  # (batch,)
    snri = torch.mean(snri).item()  # 平均 SNRI

    return {"MAE": mae, "PCC": pcc, "SNRI": snri}
