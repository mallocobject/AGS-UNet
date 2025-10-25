import torch
import torch.nn as nn
import torch.nn.functional as F


class TFTransUNet1DLoss(nn.Module):
    def __init__(self, alpha=1, beta=0.1):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, outputs, targets):
        time_loss = F.smooth_l1_loss(outputs, targets)
        outputs_fft = torch.fft.fft(outputs).abs()
        targets_fft = torch.fft.fft(targets).abs()
        freq_loss = F.mse_loss(outputs_fft, targets_fft)
        total_loss = self.alpha * time_loss + self.beta * freq_loss
        return total_loss
