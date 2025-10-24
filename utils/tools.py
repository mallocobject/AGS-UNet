import torch
import numpy as np
from sklearn.model_selection import train_test_split
import json


def set_seed(seed: int = 42):
    """
    设置随机种子以确保实验可重复性
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_split(records, seed: int = 42, split_path: str = "split.json"):
    """
    将记录列表拆分为训练集和测试集
    """
    train_records, temp_records = train_test_split(
        records, test_size=0.2, random_state=seed
    )
    val_records, test_records = train_test_split(
        temp_records, test_size=0.5, random_state=seed
    )
    split = {
        "train": train_records,
        "val": val_records,
        "test": test_records,
    }
    with open(split_path, "w") as f:
        json.dump(split, f, indent=4)
    return split


def load_split(split_path: str = "split.json") -> dict:
    """
    加载记录拆分信息
    """
    with open(split_path, "r") as f:
        split = json.load(f)
    return split
