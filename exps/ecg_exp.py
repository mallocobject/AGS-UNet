import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.rich import tqdm
from accelerate import Accelerator
import torch.distributed as dist
import numpy as np
import time
import warnings
from tqdm import TqdmExperimentalWarning

warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import set_seed, save_split, load_split, compute_metrics
from datasets import ECGDataset
from models import DDUNet, FCNDAE1D, TFTransUNet1D
from criterions import DDUNetLoss, TFTransUNet1DLoss, FCNDAE1DLoss


class ECGDenoisingExperiment:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.accelerator = Accelerator()

        self.model_dict = {
            "DDUNet": DDUNet,
            "FCNDAE1D": FCNDAE1D,
        }

    def _build_model(self):
        if self.args.model == "DDUNet":
            model = DDUNet()
        elif self.args.model == "FCNDAE1D":
            model = FCNDAE1D()
        elif self.args.model == "TFTransUNet1D":
            model = TFTransUNet1D()
        else:
            raise ValueError(f"Unknown model type: {self.args.model}")
        return model

    def _get_dataloader(self, split: str):
        dataset = ECGDataset(split=split, split_dir=self.args.split_dir)
        dataloader = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=(split == "train"),
            num_workers=4,
        )
        return dataloader

    def _select_criterion(self):
        if self.args.model == "DDUNet":
            criterion = DDUNetLoss()
        elif self.args.model == "FCNDAE1D":
            criterion = FCNDAE1DLoss()
        elif self.args.model == "TFTransUNet1D":
            criterion = TFTransUNet1DLoss()
        else:
            raise ValueError(f"Unknown model type: {self.args.model}")
        return criterion

    def _select_optimizer(self, model: nn.Module):
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.regularizer,
            betas=tuple(self.args.betas),
        )
        return optimizer

    def _select_scheduler(self, optimizer: optim.Optimizer):
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.args.epochs,
            eta_min=1e-6,
        )
        return scheduler

    def train(self):
        train_dataloader = self._get_dataloader("train")
        valid_dataloader = self._get_dataloader("val")

        model = self._build_model()

        criterion = self._select_criterion()
        best_vali_loss = float("inf")
        epochs_no_improve = 0

        optimizer = self._select_optimizer(model)
        scheduler = self._select_scheduler(optimizer)

        model, criterion, optimizer, train_dataloader, valid_dataloader = (
            self.accelerator.prepare(
                model, criterion, optimizer, train_dataloader, valid_dataloader
            )
        )

        self.accelerator.print("🚀 Starting training...")
        self.accelerator.print(f"Model: {self.args.model}")
        self.accelerator.print(f"Accelerator state: {self.accelerator.state}")

        # 用于跨进程同步的变量
        should_stop = torch.tensor(0, device=self.accelerator.device)

        for epoch in range(self.args.epochs):
            # 检查是否需要提前停止
            if should_stop.item() == 1:
                break

            model.train()
            progress_bar = tqdm(
                train_dataloader,
                desc=f"[bold cyan]Training Epoch {epoch+1}",
                unit="batch",
                colour="magenta",
                disable=not self.accelerator.is_local_main_process,
            )
            for x, label in progress_bar:
                x, label = x.to(self.accelerator.device), label.to(
                    self.accelerator.device
                )

                optimizer.zero_grad()
                outputs = model(x)
                loss = criterion(outputs, label)
                self.accelerator.backward(loss)
                optimizer.step()

            scheduler.step()

            val_metrics = self.validate(model, valid_dataloader, criterion)
            val_loss = val_metrics["loss"]

            # 在主进程中检查是否需要早停
            if self.accelerator.is_main_process:
                if val_loss < best_vali_loss:
                    best_vali_loss = val_loss
                    epochs_no_improve = 0
                    # 保存最佳模型
                    torch.save(
                        self.accelerator.get_state_dict(model),
                        self.args.save_checkpoint,
                    )
                    self.accelerator.print(
                        f"✅ Saved best model at epoch {epoch+1} with Val Loss: {val_loss:.4f}"
                    )
                else:
                    epochs_no_improve += 1
                    self.accelerator.print(
                        f"⚠️ No improvement for {epochs_no_improve} epochs"
                    )
                    if epochs_no_improve >= self.args.patience:
                        self.accelerator.print(f"⏸️ Early stopping at epoch {epoch+1}")
                        should_stop.fill_(1)  # 设置停止标志

            # 同步所有进程的停止标志
            if torch.distributed.is_initialized():
                torch.distributed.broadcast(should_stop, src=0)

            self.accelerator.print(
                f"Epoch {epoch+1}/{self.args.epochs}, Val Loss: {val_loss:.4f}, "
                f"RMSE: {val_metrics['RMSE']:.4f}, "
                f"PRD: {val_metrics['PRD']:.4f}, SNRI: {val_metrics['SNRI']:.4f}"
            )

        self.accelerator.print("🏁 Training completed!")

    def validate(self, model, val_dataloader, criterion):
        model.eval()
        total_loss = []
        metrics = {"RMSE": [], "PRD": [], "SNRI": []}

        with torch.no_grad():
            for x, label in val_dataloader:
                x, label = x.to(self.accelerator.device), label.to(
                    self.accelerator.device
                )

                outputs = model(x)
                loss = criterion(outputs, label)
                total_loss.append(loss.item())

                metrics_res = compute_metrics(outputs, label, x)
                for key in metrics:
                    metrics[key].append(metrics_res[key])

        # 收集所有进程的指标
        total_loss = self.accelerator.gather(
            torch.tensor(total_loss, device=self.accelerator.device)
        )
        for key in metrics:
            metrics[key] = self.accelerator.gather(
                torch.tensor(metrics[key], device=self.accelerator.device)
            )

        # 只在主进程上计算平均值
        if self.accelerator.is_main_process:
            vali_loss = total_loss.mean().item()
            for key in metrics:
                metrics[key] = metrics[key].mean().item()
        else:
            vali_loss = 0.0
            for key in metrics:
                metrics[key] = 0.0

        # 广播结果到所有进程
        if torch.distributed.is_initialized():
            vali_loss_tensor = torch.tensor(vali_loss, device=self.accelerator.device)
            torch.distributed.broadcast(vali_loss_tensor, src=0)
            vali_loss = vali_loss_tensor.item()

            for key in metrics:
                metric_tensor = torch.tensor(
                    metrics[key], device=self.accelerator.device
                )
                torch.distributed.broadcast(metric_tensor, src=0)
                metrics[key] = metric_tensor.item()

        return {"loss": vali_loss, **metrics}

    def test(self):
        test_dataloader = self._get_dataloader("test")

        model = self._build_model()

        # ====== 加载 checkpoint ======
        model.load_state_dict(
            torch.load(self.args.load_checkpoint, weights_only=True, map_location="cpu")
        )

        # prepare（保证设备、DDP兼容）
        model, test_dataloader = self.accelerator.prepare(model, test_dataloader)

        # ====== 测试阶段 ======
        model.eval()
        metrics = {"RMSE": [], "PRD": [], "SNRI": []}

        with torch.no_grad():
            for x, label in tqdm(
                test_dataloader,
                desc="[bold cyan]Testing",
                disable=not self.accelerator.is_local_main_process,
            ):
                x, label = x.to(self.accelerator.device), label.to(
                    self.accelerator.device
                )

                outputs = model(x)
                metrics_res = compute_metrics(outputs, label, x)
                for key in metrics:
                    metrics[key].append(metrics_res[key])

        # 收集所有进程的指标
        for key in metrics:
            metrics[key] = self.accelerator.gather(
                torch.tensor(metrics[key], device=self.accelerator.device)
            )

        # 只在主进程上计算平均值
        if self.accelerator.is_main_process:
            for key in metrics:
                metrics[key] = metrics[key].mean().item()

            self.accelerator.print("🚀 Test Results:")
            self.accelerator.print(
                f"RMSE: {metrics['RMSE']:.4f}, PRD: {metrics['PRD']:.4f}, SNRI: {metrics['SNRI']:.4f}"
            )
