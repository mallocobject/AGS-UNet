import os
import wfdb
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    import os
    import sys

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from datasets import MITNSTDBDataset

    mitdb_dir = "./ECG-Data/mitdb"
    nstdb_dir = "./ECG-Data/nstdb"
    records = ["100", "101", "102", "103", "104"]
    test_subjects = ["105"]

    train_set = MITNSTDBDataset(
        mitdb_dir, nstdb_dir, records, split="train", test_subjects=test_subjects
    )

    plt.figure(figsize=(12, 6))
    for i in range(3):
        noisy, clean = train_set[i]
        plt.subplot(3, 1, i + 1)
        plt.plot(noisy.numpy(), label="Noisy Signal", alpha=0.7)
        plt.plot(clean.numpy(), label="Clean Signal", alpha=0.7)
        plt.title(f"Sample {i + 1}")
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.legend()
    plt.tight_layout()
    plt.show()
