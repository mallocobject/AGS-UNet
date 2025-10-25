import torch
from torch.utils.data import Dataset
import os
import numpy as np
import json


class ECGDataset(Dataset):
    def __init__(self, split="train", split_dir="./data_split"):
        super().__init__()
        self.split = split
        self.split_dir = split_dir

        # 加载分割信息
        split_path = os.path.join(split_dir, "split.json")
        if not os.path.exists(split_path):
            raise FileNotFoundError(f"Split file not found: {split_path}")

        with open(split_path, "r") as f:
            self.split_data = json.load(f)

        # 获取当前分割的索引
        if split == "train":
            self.indices = self.split_data["train_indices"]
        elif split == "val":
            self.indices = self.split_data["val_indices"]
        elif split == "test":
            self.indices = self.split_data["test_indices"]
        else:
            raise ValueError(f"Unknown split: {split}")

        # 加载数据文件
        self.noisy_signals = np.load(os.path.join(split_dir, "noisy_signals.npy"))
        self.clean_signals = np.load(os.path.join(split_dir, "clean_signals.npy"))

        print(f"Loaded {split} dataset with {len(self.indices)} samples")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # 获取实际数据索引
        data_idx = self.indices[idx]

        # 获取带噪声的信号和干净信号
        noisy_signal = self.noisy_signals[data_idx]
        clean_signal = self.clean_signals[data_idx]

        # 转换为PyTorch张量
        noisy_tensor = torch.FloatTensor(noisy_signal)
        clean_tensor = torch.FloatTensor(clean_signal)

        return noisy_tensor, clean_tensor


# 测试代码
if __name__ == "__main__":
    # 测试数据集加载
    train_dataset = ECGDataset(split="train", split_dir="./data_split")
    val_dataset = ECGDataset(split="val", split_dir="./data_split")
    test_dataset = ECGDataset(split="test", split_dir="./data_split")

    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")

    # 测试一个样本
    noisy, clean = train_dataset[0]
    print(f"带噪声信号形状: {noisy.shape}")
    print(f"干净信号形状: {clean.shape}")
    print(f"带噪声信号范围: [{noisy.min():.4f}, {noisy.max():.4f}]")
    print(f"干净信号范围: [{clean.min():.4f}, {clean.max():.4f}]")
