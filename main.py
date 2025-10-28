import numpy as np
import argparse
import os
import sys
import torch
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import fft_denoise, wavelet_denoise, compute_metrics
from datasets import ECGDataset
from models import *

model_dict = {
    "UNet": UNet,
    "ACDAE": ACDAE,
    "Seq2Seq2": Seq2Seq2,
    "ralenet": ralenet,
    "DTUNet": DTUNet,
}


def main(model_name: str = "ralenet"):
    dataset = ECGDataset(
        split="test",
        noise_type="bw",
        snr_db=-4,
        split_dir="./data_split",
    )
    idx = np.random.randint(0, len(dataset) - 1)
    clean, noisy = dataset[idx]

    model = model_dict[model_name]()
    model = torch.load("./checkpoints/best_ralenet_bw_snr_-4.pth", map_location="cpu")
    model.eval()

    with torch.no_grad():
        noisy_input = torch.tensor(noisy).unsqueeze(0)  # 添加batch维度
        denoised_output = model(noisy_input).squeeze(0).numpy()  # 去除batch维度

    metrics_res = compute_metrics(denoised_output, clean)
    print("DNN Denoising Metrics:", metrics_res)

    denoised_output = denoised_output.numpy()
    clean = clean.numpy()
    noisy = noisy.numpy()

    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(clean[0], label="Clean ECG")
    plt.plot(noisy[0], label="Noisy ECG")
    plt.plot(denoised_output[0], label="DNN Denoised ECG")
    plt.title("ECG Signal (Channel 0)")
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(clean[1], label="Clean ECG")
    plt.plot(noisy[1], label="Noisy ECG")
    plt.plot(denoised_output[1], label="DNN Denoised ECG")
    plt.title("ECG Signal (Channel 1)")
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
