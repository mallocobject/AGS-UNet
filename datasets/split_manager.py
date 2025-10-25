# split_manager.py
import os
import json
import numpy as np
from typing import Dict, List
import wfdb


class SplitManager:
    def __init__(
        self,
        mitdb_dir: str,
        nstdb_dir: str,
        window_size: int = 3600,
        step_size: int = 3600,
        seed: int = 42,
    ):
        self.mitdb_dir = mitdb_dir
        self.nstdb_dir = nstdb_dir
        self.window_size = window_size
        self.step_size = step_size
        self.seed = seed
        np.random.seed(seed)

        self.clean = (None, None)
        self.noisy = (None, None)

    def _get_all_records(self) -> List[str]:
        """获取所有记录文件名（不带扩展名）"""
        return [
            f.split(".")[0] for f in os.listdir(self.mitdb_dir) if f.endswith(".dat")
        ]

    def _load_noise_segments(self) -> Dict[str, np.ndarray]:
        """加载噪声片段，按类型分开存储"""
        nstdb_files = ["bw", "ma", "em"]
        noise_segments = {}

        for nt in nstdb_files:
            segments = []
            rec = wfdb.rdrecord(os.path.join(self.nstdb_dir, nt))
            n_sig = rec.p_signal[:, 0]
            for start in range(0, len(n_sig) - self.window_size, self.step_size):
                seg = n_sig[start : start + self.window_size]
                if len(seg) == self.window_size:
                    segments.append(seg)

            noise_segments[nt] = np.array(segments, dtype=np.float32)
            print(f"Loaded {len(segments)} segments for noise type: {nt}")

        return noise_segments

    def _load_clean_segments(self) -> np.ndarray:
        """加载干净信号片段"""
        records = self._get_all_records()
        clean_segments = []

        for rid in records:
            rec_path = os.path.join(self.mitdb_dir, rid)
            record = wfdb.rdrecord(rec_path)
            sig = record.p_signal[:, 0]
            for start in range(0, len(sig) - self.window_size, self.step_size):
                seg = sig[start : start + self.window_size]
                if len(seg) == self.window_size:
                    clean_segments.append(seg)

        return np.array(clean_segments, dtype=np.float32)

    def _normalize_signals(
        self, signals: np.ndarray, stats: tuple, method: str = "zscore"
    ) -> np.ndarray:
        """归一化信号

        Args:
            signals: 输入信号数组
            method: 归一化方法，可选 "zscore" (标准化) 或 "minmax" (最小最大归一化)

        Returns:
            归一化后的信号数组
        """
        if method == "zscore":
            normalized = (signals - stats[0]) / stats[1]
        elif method == "minmax":
            normalized = (signals - stats[0]) / (stats[1] - stats[0])
        else:
            raise ValueError(f"Unknown normalization method: {method}")

        return normalized

    def _create_composite_noise(
        self, noise_segments_dict: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """创建复合噪声，混合所有三种噪声类型"""
        composite_noise = []
        noise_types = list(noise_segments_dict.keys())

        # 找到最小长度，确保所有噪声类型都有足够的样本
        min_length = min(len(segments) for segments in noise_segments_dict.values())

        # 为每种噪声类型创建索引排列
        indices = {}
        for nt in noise_types:
            indices[nt] = np.random.permutation(len(noise_segments_dict[nt]))[
                :min_length
            ]

        # 创建复合噪声 (bw_noise + ma_noise + em_noise) / 3
        for i in range(min_length):
            mixed_noise = np.zeros(self.window_size, dtype=np.float32)

            for nt in noise_types:
                noise_segment = noise_segments_dict[nt][indices[nt][i]]
                mixed_noise += noise_segment

            # 平均所有噪声类型
            mixed_noise /= len(noise_types)
            composite_noise.append(mixed_noise)

        return np.array(composite_noise, dtype=np.float32)

    def _create_noisy_signals(
        self,
        clean_segments: np.ndarray,
        composite_noise: np.ndarray,
        snr_db: float = 0.0,
    ) -> np.ndarray:
        """创建带复合噪声的信号"""
        noisy_signals = []

        for i, clean_sig in enumerate(clean_segments):
            # 循环使用复合噪声样本
            noise_idx = i % len(composite_noise)
            noise = composite_noise[noise_idx]

            # 计算SNR并调整噪声水平
            clean_power = np.mean(clean_sig**2)
            noise_power = np.mean(noise**2)

            # 根据目标SNR调整噪声
            target_noise_power = clean_power / (10 ** (snr_db / 10))
            scale_factor = np.sqrt(target_noise_power / noise_power)
            noise = noise * scale_factor

            # 合成带噪声的信号
            noisy_sig = clean_sig + noise
            noisy_signals.append(noisy_sig)

        return np.array(noisy_signals, dtype=np.float32)

    def save_split(
        self,
        split_dir: str,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        snr_db: float = 0.0,
        normalization_method: str = "zscore",
    ) -> Dict[str, List[int]]:
        """保存数据集划分到 JSON 文件

        Args:
            split_dir: 分割文件保存目录
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            snr_db: 信噪比 (dB)
            normalization_method: 归一化方法，"zscore" 或 "minmax"
        """
        print("Loading noise segments...")
        noise_segments = self._load_noise_segments()

        print("Loading clean segments...")
        clean_segments = self._load_clean_segments()
        print(f"Loaded {len(clean_segments)} clean segments")

        # 创建复合噪声（混合所有三种噪声）
        print("Creating composite noise...")
        composite_noise = self._create_composite_noise(noise_segments)
        print(f"Created {len(composite_noise)} composite noise segments")

        # 创建带噪声的信号
        print("Creating noisy signals...")
        noisy_signals = self._create_noisy_signals(
            clean_segments, composite_noise, snr_db
        )

        # 划分索引
        n_total = len(clean_segments)
        indices = list(range(n_total))
        np.random.shuffle(indices)

        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        train_indices = indices[:n_train]
        val_indices = indices[n_train : n_train + n_val]
        test_indices = indices[n_train + n_val :]

        train_clean_data = noisy_signals[train_indices]
        train_noisy_data = clean_segments[train_indices]
        # 计算归一化参数
        if normalization_method == "zscore":
            self.clean = (np.mean(train_clean_data), np.std(train_clean_data))
            self.noisy = (np.mean(train_noisy_data), np.std(train_noisy_data))
        elif normalization_method == "minmax":
            self.clean = (np.min(train_clean_data), np.max(train_clean_data))
            self.noisy = (np.min(train_noisy_data), np.max(train_noisy_data))
        else:
            raise ValueError(f"Unknown normalization method: {normalization_method}")

        # 归一化数据
        print("Normalizing signals...")
        noisy_signals_normalized = self._normalize_signals(
            noisy_signals, self.noisy, method=normalization_method
        )

        clean_segments_normalized = self._normalize_signals(
            clean_segments, self.clean, method=normalization_method
        )

        split_indices = {
            "train": train_indices,
            "val": val_indices,
            "test": test_indices,
        }

        # 确保目录存在
        os.makedirs(split_dir, exist_ok=True)

        # 保存分割信息
        split_path = os.path.join(split_dir, "split.json")
        with open(split_path, "w") as f:
            json.dump(
                {
                    "train_indices": train_indices,
                    "val_indices": val_indices,
                    "test_indices": test_indices,
                    "total_samples": n_total,
                    "train_ratio": train_ratio,
                    "val_ratio": val_ratio,
                    "test_ratio": 1.0 - train_ratio - val_ratio,
                    "window_size": self.window_size,
                    "step_size": self.step_size,
                    "snr_db": snr_db,
                    "seed": self.seed,
                    "noise_types_used": list(noise_segments.keys()),
                    "normalization_method": normalization_method,
                    "original_stats": {
                        "clean(mean/min)": float(self.clean[0]),
                        "clean(std/max)": float(self.clean[1]),
                        "noisy(mean/min)": float(self.noisy[0]),
                        "noisy(std/max)": float(self.noisy[1]),
                    },
                },
                f,
                indent=2,
            )

        print(f"Saved split to {split_path}")
        print(f"Train: {len(train_indices)} samples")
        print(f"Val: {len(val_indices)} samples")
        print(f"Test: {len(test_indices)} samples")

        # 保存归一化后的数据文件
        np.save(os.path.join(split_dir, "noisy_signals.npy"), noisy_signals_normalized)
        np.save(os.path.join(split_dir, "clean_signals.npy"), clean_segments_normalized)
        np.save(os.path.join(split_dir, "composite_noise.npy"), composite_noise)

        return split_indices


if __name__ == "__main__":
    mitdb_dir = "./ECG-Data/mitdb"
    nstdb_dir = "./ECG-Data/nstdb"
    split_dir = "./data_split"

    manager = SplitManager(mitdb_dir, nstdb_dir)

    # 使用Z-score标准化
    manager.save_split(
        split_dir,
        train_ratio=0.8,
        val_ratio=0.1,
        snr_db=0.0,
        normalization_method="zscore",
    )

    # 或者使用最小最大归一化
    # manager.save_split(split_dir, train_ratio=0.8, val_ratio=0.1, snr_db=0.0, normalization_method="minmax")
