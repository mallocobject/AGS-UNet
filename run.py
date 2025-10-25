import argparse

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exps import ECGDenoisingExperiment


def parse_args():
    parser = argparse.ArgumentParser(description="ECG Denoising Experiment")

    # ====== 数据路径 ======
    parser.add_argument(
        "--split_dir",
        type=str,
        default="./data_split",
        help="Path to split directory containing data splits and files",
    )

    # ====== 模型与训练参数 ======
    parser.add_argument(
        "--model",
        type=str,
        default="DDUNet",
        choices=["DDUNet", "FCNDAE1D", "TFTransUNet1D"],
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--regularizer", type=float, default=1e-5)
    parser.add_argument("--betas", nargs=2, type=float, default=[0.9, 0.999])
    parser.add_argument(
        "--patience", type=int, default=10, help="Early stopping patience"
    )

    # ====== 模型保存与加载 ======
    parser.add_argument(
        "--save_checkpoint", type=str, default="./checkpoints/best_model.pth"
    )
    parser.add_argument(
        "--load_checkpoint", type=str, default="./checkpoints/best_model.pth"
    )

    # ====== 模式选择 ======
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"])

    return parser.parse_args()


def main():
    args = parse_args()
    exp = ECGDenoisingExperiment(args)

    if args.mode == "train":
        exp.train()
    elif args.mode == "test":
        exp.test()


if __name__ == "__main__":
    main()
