import torch
from torch.utils.data import Dataset
import os
import numpy as np
import wfdb


class MITNSTDBDataset(Dataset):
    def __init__(
        self,
        mitdb_dir,
        nstdb_dir,
        record_names,
        window_size=3600,
        step_size=3600,
        snr_db=0,
        split="train",
        val_ratio=0.1,
        test_subjects=None,
        seed=42,
    ):
        super().__init__()
        self.window_size = window_size
        self.step_size = step_size
        self.snr_db = snr_db
        self.split = split
        np.random.seed(seed)

        # ========= 训练 / 测试划分 =========
        if split == "test":
            assert test_subjects is not None
            record_names = test_subjects
        elif split in ["train", "val"] and test_subjects is not None:
            record_names = [r for r in record_names if r not in test_subjects]

        # ========= 读取干净信号 =========
        self.clean_segments = []
        for rec in record_names:
            record = wfdb.rdrecord(os.path.join(mitdb_dir, rec))
            sig = record.p_signal[:, 0]  # 取MLII导联
            for start in range(0, len(sig) - window_size, step_size):
                seg = sig[start : start + window_size]
                self.clean_segments.append(seg)

        self.clean_segments = np.array(self.clean_segments, dtype=np.float32)

        # ========= 读取噪声信号 =========
        noise_types = ["bw", "em", "ma"]
        """
        baseline: 基线漂移噪声 (Baseline Wander)
        muscle: 肌电噪声 (EMG)
        motion: 电极移动伪迹 (Motion Artifact)
        """
        self.noises = []
        for nt in noise_types:
            path = os.path.join(nstdb_dir, nt)
            if not os.path.exists(path + ".hea"):
                raise FileNotFoundError(f"Missing file: {path}.hea or .dat")
            rec = wfdb.rdrecord(os.path.join(nstdb_dir, nt))
            n_sig = rec.p_signal[:, 0]
            self.noises.append(n_sig)

        # ========= 划分验证集 =========
        if split in ["train", "val"]:
            n_total = len(self.clean_segments)
            n_val = int(val_ratio * n_total)
            idxs = np.random.permutation(n_total)
            if split == "train":
                self.clean_segments = self.clean_segments[idxs[n_val:]]
            else:
                self.clean_segments = self.clean_segments[idxs[:n_val]]

    def __len__(self):
        return len(self.clean_segments)

    def __getitem__(self, idx):
        clean = self.clean_segments[idx].copy()

        # 随机选噪声类型 & 段
        ntype = np.random.randint(len(self.noises))
        noise = self.noises[ntype]
        start = np.random.randint(0, len(noise) - self.window_size)
        noise = noise[start : start + self.window_size].copy()

        # 调整噪声能量以匹配目标SNR
        clean_power = np.mean(clean**2)
        noise_power = np.mean(noise**2)
        desired_noise_power = clean_power / (10 ** (self.snr_db / 10))
        noise *= np.sqrt(desired_noise_power / (noise_power + 1e-12))

        noisy = clean + noise

        # Z-normalization
        clean = (clean - clean.mean()) / (clean.std() + 1e-8)
        noisy = (noisy - noisy.mean()) / (noisy.std() + 1e-8)

        return (
            torch.tensor(noisy, dtype=torch.float32),
            torch.tensor(clean, dtype=torch.float32),
        )


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    mitdb_dir = "./ECG-Data/mitdb"
    nstdb_dir = "./ECG-Data/nstdb"
    records = ["100", "101", "102", "103", "104"]
    test_subjects = ["105"]

    train_set = MITNSTDBDataset(
        mitdb_dir, nstdb_dir, records, split="train", test_subjects=test_subjects
    )
    val_set = MITNSTDBDataset(
        mitdb_dir, nstdb_dir, records, split="val", test_subjects=test_subjects
    )
    test_set = MITNSTDBDataset(
        mitdb_dir, nstdb_dir, records, split="test", test_subjects=test_subjects
    )

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
