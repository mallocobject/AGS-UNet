import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchaudio.transforms as T
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import os
import sys


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import set_seed, save_split, load_split, compute_metrics
from datasets import ECGDataset
from models import DDUnet, FCNDAE1D

device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")


# def criterion(
#     denoised: torch.Tensor,
#     clean: torch.Tensor,
#     use_freq: bool = False,
#     sample_rate: int = 360,
# ):
#     """
#     计算 ECG 去噪损失(MSE + MAE + 可选频域损失)

#     Args:
#         clean (torch.Tensor): 干净 ECG 信号，形状 (batch, length)
#         denoised (torch.Tensor): 去噪信号，形状同 clean
#         use_freq (bool): 是否使用频域损失
#         sample_rate (int): 采样率(Hz), 用于 STFT

#     Returns:
#         torch.Tensor: 总损失
#     """
#     assert (
#         clean.shape == denoised.shape
#     ), "Clean and denoised signals must have same shape"

#     # MSE 损失
#     mse_loss = torch.mean((clean - denoised) ** 2)

#     # MAE 损失
#     mae_loss = torch.mean(torch.abs(clean - denoised))

#     # 频域损失（可选）
#     fft_loss = 0.0
#     if use_freq:
#         clean_fft = torch.fft.fft(clean).abs()  # (batch, length)
#         denoised_fft = torch.fft.fft(denoised).abs()
#         fft_loss = torch.mean((clean_fft - denoised_fft) ** 2)

#         # stft = T.Spectrogram(n_fft=12, hop_length=4, power=None).to(clean.device)
#         # clean_spec = stft(clean.unsqueeze(1)).abs()  # (batch, 1, freq, time)
#         # denoised_spec = stft(denoised.unsqueeze(1)).abs()
#         # freq_loss = torch.mean((clean_spec - denoised_spec) ** 2)

#     # 组合损失
#     alpha, beta, gamma = 0.7, 0.3, 0.05 if use_freq else 0.0
#     total_loss = alpha * mse_loss + beta * mae_loss + gamma * fft_loss

#     return total_loss


def validate_model(model: nn.Module, val_loader: DataLoader):
    model.eval()
    val_loss = 0.0
    metrics_sum = {"RMSE": 0.0, "PRD": 0.0, "SNR": 0.0}
    num_batches = 0

    criterion = nn.MSELoss()

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            # loss = criterion(outputs, labels, use_freq=False)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)

            # 计算指标
            metrics = compute_metrics(outputs, labels, inputs)
            for key in metrics_sum:
                metrics_sum[key] += metrics[key] * inputs.size(0)
            num_batches += inputs.size(0)

    val_loss /= num_batches
    metrics_avg = {key: value / num_batches for key, value in metrics_sum.items()}

    return {"loss": val_loss, **metrics_avg}


def test_model(
    model: nn.Module,
    test_loader: DataLoader,
    checkpoint_path: str = "checkpoints/best_ddunet.pth",
):
    model.load_state_dict(torch.load(checkpoint_path, weights_only=False))
    test_metrics = validate_model(model, test_loader)
    print(
        f"Test Results - Loss: {test_metrics['loss']:.4f}, RMSE: {test_metrics['RMSE']:.4f}, "
        f"PRD: {test_metrics['PRD']:.4f}, SNR: {test_metrics['SNR']:.4f}"
    )


def train_model(
    model,
    train_loader,
    val_loader,
    num_epochs=25,
    learning_rate=0.001,
    patience=100,
    save_path: str = None,
):
    # optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer, T_max=num_epochs, eta_min=1e-6
    # )
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.99))
    criterion = nn.MSELoss()
    best_val_loss = float("inf")

    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False
        ):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            # loss = criterion(outputs, labels, use_freq=False)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        val_metrics = validate_model(model, val_loader)
        val_loss = val_metrics["loss"]

        print(
            f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, RMSE: {val_metrics['RMSE']:.4f}, "
            f"PRD: {val_metrics['PRD']:.4f}, SNR: {val_metrics['SNR']:.4f}"
        )

        # scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model at epoch {epoch+1}")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epochs")
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    test_model(model, test_loader, checkpoint_path=save_path)


if __name__ == "__main__":

    mitdb_dir = "./ECG-Data/mitdb"
    nstdb_dir = "./ECG-Data/nstdb"
    # 动态加载所有 MIT-BIH 记录
    records = [f.split(".")[0] for f in os.listdir(mitdb_dir) if f.endswith(".dat")]

    split_path = "split.json"
    set_seed(42)
    if not os.path.exists(split_path):
        split = save_split(records, seed=42, split_path=split_path)
    else:
        split = load_split(split_path=split_path)

    train_set = ECGDataset(mitdb_dir, nstdb_dir, split["train"], split="train")
    val_set = ECGDataset(mitdb_dir, nstdb_dir, split["val"], split="val")
    test_set = ECGDataset(mitdb_dir, nstdb_dir, split["test"], split="test")

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

    # 初始化模型
    # model = DDUnet(
    #     chin=1,
    #     chout=1,
    #     hidden=32,
    #     depth=4,
    #     kernel_size=3,
    #     stride=2,
    #     causal=False,
    #     growth=1.5,
    #     glu=False,
    # )

    model = FCNDAE1D()

    # 训练模型
    train_model(
        model,
        train_loader,
        val_loader,
        num_epochs=100,
        learning_rate=0.001,
        save_path="checkpoints/best_fcndae.pth",
    )
