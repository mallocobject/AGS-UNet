import torch
import torch.nn as nn


class DRNN(nn.Module):
    def __init__(self, num_relus: int):
        super().__init__()
        self.lstm = nn.LSTM(1, 256, num_layers=1, batch_first=True, bidirectional=False)
        self.relu_layers = nn.ModuleList(
            [nn.Sequential(nn.Linear(256, 256), nn.ReLU()) for _ in range(num_relus)]
        )
        self.fc3 = nn.Linear(256, 3600)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # 取最后一个时间步的输出
        for relu_layer in self.relu_layers:
            out = relu_layer(out)
        out = self.fc3(out)
        return out.squeeze(-1)


if __name__ == "__main__":
    model = DRNN(2)
    x = torch.randn(4, 3600)  # batch=4, 长度1024
    print(model)
    y = model(x)

    print(f"输入形状: {x.shape}")
    print(f"输出形状: {y.shape}")
