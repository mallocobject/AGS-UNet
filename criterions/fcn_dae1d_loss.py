import torch
import torch.nn as nn
import torch.nn.functional as F


class FCNDAE1DLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, targets):
        return F.mse_loss(outputs, targets)
