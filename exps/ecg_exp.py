import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.rich import tqdm
from accelerate import Accelerator, DistributedDataParallelKwargs
import numpy as np
import time
import warnings
from tqdm import TqdmExperimentalWarning

warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import compute_metrics
from datasets import ECGDataset
from models import *


class ECGDenoisingExperiment:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        # ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        # self.accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
        self.accelerator = Accelerator()

        self.model_dict = {
            "UNet": UNet,
            "ACDAE": ACDAE,
            "Seq2Seq2": Seq2Seq2,
            "ralenet": ralenet,
            "DTUNet": DTUNet,
        }

        self.checkpoint = os.path.join(
            self.args.checkpoint_dir,
            f"best_{self.args.model}_{self.args.noise_type}_snr_{self.args.snr_db}.pth",
        )

    def _build_model(self):
        if self.args.model not in self.model_dict:
            raise ValueError(f"Unknown model type: {self.args.model}")
        model = self.model_dict[self.args.model]()
        return model

    def _get_dataloader(self, split: str):
        dataset = ECGDataset(
            split=split,
            noise_type=self.args.noise_type,
            snr_db=self.args.snr_db,
            split_dir=self.args.split_dir,
        )
        self.mean, self.std = dataset.get_stats()
        self.mean = self.mean.to(self.accelerator.device)
        self.std = self.std.to(self.accelerator.device)
        dataloader = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=(split == "train"),
            num_workers=2,
        )
        return dataloader

    def _select_criterion(self):
        # if self.args.model == "DDUNet":
        #     criterion = DDUNetLoss()
        # elif self.args.model == "FCNDAE1D":
        #     criterion = FCNDAE1DLoss()
        # elif self.args.model == "TFTransUNet1D":
        #     criterion = TFTransUNet1DLoss()
        # else:
        #     raise ValueError(f"Unknown model type: {self.args.model}")
        # return criterion
        return nn.MSELoss()

    def _select_optimizer(self, model: nn.Module):
        optimizer = optim.Adam(model.parameters(), lr=self.args.lr)
        return optimizer

    def _select_scheduler(self, optimizer: optim.Optimizer):
        def lr_lambda(epoch):
            if epoch < 40:
                return 1.0  # 保持初始学习率
            else:
                return 0.1

        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.args.epochs, eta_min=1e-5
        )
        # scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        return scheduler

    def train(self):
        dataloader = self._get_dataloader("train")

        model = self._build_model()

        criterion = self._select_criterion()

        optimizer = self._select_optimizer(model)
        scheduler = self._select_scheduler(optimizer)

        model, criterion, optimizer, scheduler, dataloader = self.accelerator.prepare(
            model, criterion, optimizer, scheduler, dataloader
        )

        self.accelerator.print("🚀 Starting training...")
        self.accelerator.print(f"Model: {self.args.model}")
        self.accelerator.print(f"Accelerator state: {self.accelerator.state}")

        for epoch in range(self.args.epochs):

            model.train()
            progress_bar = tqdm(
                dataloader,
                desc=f"[bold cyan]Training Epoch {epoch+1}",
                unit="batch",
                colour="magenta",
                disable=not self.accelerator.is_local_main_process,
            )
            losses = []
            for x, label in progress_bar:
                x, label = x.to(self.accelerator.device), label.to(
                    self.accelerator.device
                )

                optimizer.zero_grad()
                # print(x.shape, label.shape)
                outputs = model(x)
                loss = criterion(outputs, label)
                losses.append(loss.item())
                self.accelerator.backward(loss)
                optimizer.step()

            scheduler.step()

            avg_loss = np.mean(
                self.accelerator.gather(
                    torch.tensor(losses, device=self.accelerator.device)
                )
                .cpu()
                .numpy()
            )

            self.accelerator.print(
                f"Epoch {epoch+1}/{self.args.epochs}, Train Loss: {avg_loss:.4f}"
            )

            self.test(model=model)

        torch.save(self.accelerator.get_state_dict(model), self.checkpoint)
        self.accelerator.print(f"✅ Model saved to {self.checkpoint}")
        self.accelerator.print("🏁 Training completed!")

    # def validate(self, model, val_dataloader, criterion):
    #     model.eval()
    #     total_loss = []
    #     metrics = {"RMSE": [], "PRD": [], "SNR": []}

    #     with torch.no_grad():
    #         for x, label in val_dataloader:
    #             x, label = x.to(self.accelerator.device), label.to(
    #                 self.accelerator.device
    #             )

    #             outputs = model(x)
    #             loss = criterion(outputs, label)
    #             total_loss.append(loss.item())

    #             metrics_res = compute_metrics(outputs, label, x)
    #             for key in metrics:
    #                 metrics[key].append(metrics_res[key])

    #     # 收集所有进程的指标
    #     total_loss = self.accelerator.gather(
    #         torch.tensor(total_loss, device=self.accelerator.device)
    #     )
    #     for key in metrics:
    #         metrics[key] = self.accelerator.gather(
    #             torch.tensor(metrics[key], device=self.accelerator.device)
    #         )

    #     # 只在主进程上计算平均值
    #     if self.accelerator.is_main_process:
    #         vali_loss = total_loss.mean().item()
    #         for key in metrics:
    #             metrics[key] = metrics[key].mean().item()
    #     else:
    #         vali_loss = 0.0
    #         for key in metrics:
    #             metrics[key] = 0.0

    #     # 广播结果到所有进程
    #     if torch.distributed.is_initialized():
    #         vali_loss_tensor = torch.tensor(vali_loss, device=self.accelerator.device)
    #         torch.distributed.broadcast(vali_loss_tensor, src=0)
    #         vali_loss = vali_loss_tensor.item()

    #         for key in metrics:
    #             metric_tensor = torch.tensor(
    #                 metrics[key], device=self.accelerator.device
    #             )
    #             torch.distributed.broadcast(metric_tensor, src=0)
    #             metrics[key] = metric_tensor.item()

    #     return {"loss": vali_loss, **metrics}

    def test(self, model: nn.Module = None):
        test_dataloader = self._get_dataloader("test")

        if model is None:
            model = self._build_model()
            model.load_state_dict(
                torch.load(self.checkpoint, weights_only=True, map_location="cpu")
            )

        # prepare（保证设备、DDP兼容）
        model, test_dataloader = self.accelerator.prepare(model, test_dataloader)

        # ====== 测试阶段 ======
        model.eval()
        metrics = {"RMSE": [], "SNR": []}

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
                metrics_res = compute_metrics(outputs, label, self.mean, self.std)
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
                f"RMSE: {metrics['RMSE']:.2f}, SNR: {metrics['SNR']:.2f}"
            )
