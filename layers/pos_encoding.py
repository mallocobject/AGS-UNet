import torch
import torch.nn as nn


class AbsPositionalEncoding(nn.Module):
    """Absolute Position embedding for Transformer"""

    def __init__(self, num_hiddens, dropout=0.0, max_len=1000):
        super(AbsPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # generate a P matrix with shape of (1, max_len, num_hiddens)
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / torch.pow(
            10000, torch.arange(0, num_hiddens, 2, dtype=torch.float32) / num_hiddens
        )
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, : X.shape[1], :].to(X.device)
        return self.dropout(X)


if __name__ == "__main__":
    x = torch.randn(2, 256, 8)  # batch=2, seq_len=256, channels=8
    pos_enc = AbsPositionalEncoding(num_hiddens=8, dropout=0.1, max_len=1000)
    y = pos_enc(x)
    print(y.shape)  # torch.Size([2, 256, 8])
