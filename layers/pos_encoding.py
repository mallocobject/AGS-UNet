import torch
import torch.nn as nn


class PositionalEncoding1D(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding1D, self).__init__()

        # 创建位置编码矩阵 [max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 注册为缓冲区，形状为 [1, max_len, d_model]
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):

        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]

        return x


# 测试代码
if __name__ == "__main__":
    d_model = 16
    max_len = 60
    pe = PositionalEncoding1D(d_model, max_len)

    # 测试 batch_first 格式
    x1 = torch.zeros(10, 60, d_model)  # (batch_size, seq_len, d_model)
    x1_with_pe = pe(x1)
    print(f"batch_first 格式输入形状: {x1.shape}")
    print(f"batch_first 格式输出形状: {x1_with_pe.shape}")

    # 验证位置编码是否正确添加
    print(f"位置编码形状: {pe.pe.shape}")
    print(f"第一个批次，第一个位置的编码: {x1_with_pe[0, 0, :4]}")  # 显示前4个值
    print(f"第二个批次，第一个位置的编码: {x1_with_pe[1, 0, :4]}")  # 应该相同
    print(f"第一个批次，第二个位置的编码: {x1_with_pe[0, 1, :4]}")  # 应该不同
