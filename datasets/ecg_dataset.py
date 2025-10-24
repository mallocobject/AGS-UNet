import torch
from torch.utils.data import Dataset
import os
import numpy as np
import wfdb
import json

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import set_seed, save_split, load_split


class ECGDataset(Dataset):
    def __init__(
        self,
        mitdb_dir,
        nstdb_dir,
        record_names,
        window_size=1024,
        step_size=1024,
        split="train",
        seed=42,
        split_path="split.json",
    ):
        super().__init__()
        self.window_size = window_size
        self.step_size = step_size
        self.split = split
        self.split_path = split_path
        np.random.seed(seed)

        # 记录列表
        print(f"Loading ECGDataset for split: {split} with records: {record_names}")
        self.record_names = record_names
        self.mitdb_files = [os.path.join(mitdb_dir, f"{rid}") for rid in record_names]
        self.noise_types = ["bw", "ma", "em"]

        # 读取噪声
        self.noise_segments = []
        for nt in self.noise_types:
            rec = wfdb.rdrecord(os.path.join(nstdb_dir, nt))
            n_sig = rec.p_signal[:, 0]
            for start in range(0, len(n_sig) - window_size, step_size):
                seg = n_sig[start : start + window_size]
                if len(seg) == window_size:
                    self.noise_segments.append(seg)
        self.noise_segments = np.array(self.noise_segments, dtype=np.float32)

        # 读取干净 ECG
        self.clean_segments = []
        for rec in self.mitdb_files:
            record = wfdb.rdrecord(rec)
            sig = record.p_signal[:, 0]
            for start in range(0, len(sig) - window_size, step_size):
                seg = sig[start : start + window_size]
                if len(seg) == window_size:
                    self.clean_segments.append(seg)
        self.clean_segments = np.array(self.clean_segments, dtype=np.float32)

        # 预生成噪声索引和 SNR
        self.noise_indices = []
        self.snrs = []
        noise_split_path = f"noise_split_{split}.json"
        if os.path.exists(noise_split_path):
            with open(noise_split_path, "r") as f:
                noise_data = json.load(f)
                self.noise_indices = noise_data["noise_indices"]
                self.snrs = noise_data["snrs"]
        else:
            for _ in range(len(self.clean_segments)):
                bw_idx = int(np.random.randint(len(self.noise_segments)))
                ma_idx = int(np.random.randint(len(self.noise_segments)))
                em_idx = int(np.random.randint(len(self.noise_segments)))
                self.noise_indices.append([bw_idx, ma_idx, em_idx])
                snr_db = (
                    np.random.choice([-2.5, 0, 2.5, 5, 7.5])
                    if split in ["train", "val"]
                    else np.random.choice([-1, 3, 7])
                )
                self.snrs.append(float(snr_db))
            with open(noise_split_path, "w") as f:
                json.dump({"noise_indices": self.noise_indices, "snrs": self.snrs}, f)

    def __len__(self):
        return len(self.clean_segments)

    def __getitem__(self, idx):
        clean = self.clean_segments[idx].copy()
        bw_idx, ma_idx, em_idx = self.noise_indices[idx]
        snr_db = self.snrs[idx]

        # 组合噪声
        bw_noise = self.noise_segments[bw_idx].copy()
        ma_noise = self.noise_segments[ma_idx].copy()
        em_noise = self.noise_segments[em_idx].copy()
        noise = (bw_noise + ma_noise + em_noise) / 3

        # 应用 SNR
        clean_power = np.mean(clean**2)
        noise_power = np.mean(noise**2)
        desired_noise_power = clean_power / (10 ** (snr_db / 10))
        noise *= np.sqrt(desired_noise_power / (noise_power + 1e-12))

        noisy_orig = clean + noise

        # 归一化到 [0, 1]
        clean = (clean - clean.min()) / (clean.max() - clean.min() + 1e-8)
        noisy = (noisy_orig - noisy_orig.min()) / (
            noisy_orig.max() - noisy_orig.min() + 1e-8
        )

        return (
            torch.tensor(noisy, dtype=torch.float32),
            torch.tensor(clean, dtype=torch.float32),
        )


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    mitdb_dir = "./ECG-Data/mitdb"
    nstdb_dir = "./ECG-Data/nstdb"
    records = [f.split(".")[0] for f in os.listdir(mitdb_dir) if f.endswith(".dat")]

    set_seed(42)

    split_path = "split.json"
    if not os.path.exists(split_path):
        split = save_split(records, seed=42, split_path=split_path)
    else:
        split = load_split(split_path=split_path)

    train_set = ECGDataset(mitdb_dir, nstdb_dir, split["train"], split="train")
    val_set = ECGDataset(mitdb_dir, nstdb_dir, split["val"], split="val")
    test_set = ECGDataset(mitdb_dir, nstdb_dir, split["test"], split="test")

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
