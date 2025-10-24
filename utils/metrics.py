import torch


def compute_metrics(
    denoised: torch.Tensor,
    clean: torch.Tensor,
    noisy: torch.Tensor,
    eps: float = 1e-8,
) -> dict:
    """
    计算ECG信号去噪评估指标:
    RMSE, PRD, SNRI
    """
    if clean.dim() == 1:
        clean = clean.unsqueeze(0)
        denoised = denoised.unsqueeze(0)
        noisy = noisy.unsqueeze(0)

    if clean.dim() == 3:
        clean = clean.squeeze(1)
        denoised = denoised.squeeze(1)
        noisy = noisy.squeeze(1)

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

    # SNRI
    noise_power = torch.sum((clean - noisy) ** 2, dim=1)
    residual_noise_power = torch.sum((clean - denoised) ** 2, dim=1)
    snri = 10 * torch.log10((noise_power) / (residual_noise_power))  # (batch,)
    snri_mean = torch.mean(snri).item()

    return {"RMSE": rmse_mean, "PRD": prd_mean, "SNRI": snri_mean}
