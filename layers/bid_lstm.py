import torch
import torch.nn as nn
from typing import Optional


class BLSTM(nn.Module):
    def __init__(self, dim: int, layers: int = 2, bid: bool = True):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=dim,
            hidden_size=dim,
            num_layers=layers,
            batch_first=False,
            bidirectional=bid,
        )
        self.linear = None
        if bid:
            self.linear = nn.Linear(dim * 2, dim)

    def forward(self, x: torch.Tensor, hidden: Optional[torch.Tensor] = None):
        x, hidden = self.lstm(x, hidden)
        if self.linear is not None:
            x = self.linear(x)
        return x, hidden


if __name__ == "__main__":
    lstm = BLSTM(256)
    x = torch.randn(100, 256, 25)  # (B, C, T)
    x = x.permute(2, 0, 1)
    x, _ = lstm(x)
    x = x.permute(1, 2, 0)
    print(x.shape)
