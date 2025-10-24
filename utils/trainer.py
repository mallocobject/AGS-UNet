"""Training utilities for DDU-Net"""

import torch
import torch.nn as nn
from tqdm import tqdm
from .metrics import dice_coefficient, iou_score


class Trainer:
    """
    Trainer class for DDU-Net
    
    Args:
        model: DDU-Net model
        optimizer: Optimizer
        criterion: Loss function
        device: Device to train on
        metrics: List of metrics to compute
    """
    
    def __init__(self, model, optimizer, criterion=None, device='cuda', metrics=None):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.model.to(device)
        
        if criterion is None:
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = criterion
        
        if metrics is None:
            self.metrics = {'dice': dice_coefficient, 'iou': iou_score}
        else:
            self.metrics = metrics
    
    def train_epoch(self, dataloader):
        """
        Train for one epoch
        
        Args:
            dataloader: Training data loader
        
        Returns:
            Dictionary of average metrics
        """
        self.model.train()
        total_loss = 0
        metrics_sum = {name: 0 for name in self.metrics.keys()}
        
        pbar = tqdm(dataloader, desc='Training')
        for images, masks in pbar:
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            for name, metric_fn in self.metrics.items():
                metrics_sum[name] += metric_fn(outputs, masks).item()
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
        
        # Calculate averages
        num_batches = len(dataloader)
        avg_loss = total_loss / num_batches
        avg_metrics = {name: value / num_batches for name, value in metrics_sum.items()}
        avg_metrics['loss'] = avg_loss
        
        return avg_metrics
    
    def validate(self, dataloader):
        """
        Validate the model
        
        Args:
            dataloader: Validation data loader
        
        Returns:
            Dictionary of average metrics
        """
        self.model.eval()
        total_loss = 0
        metrics_sum = {name: 0 for name in self.metrics.keys()}
        
        with torch.no_grad():
            pbar = tqdm(dataloader, desc='Validation')
            for images, masks in pbar:
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                
                # Update metrics
                total_loss += loss.item()
                for name, metric_fn in self.metrics.items():
                    metrics_sum[name] += metric_fn(outputs, masks).item()
                
                # Update progress bar
                pbar.set_postfix({'loss': loss.item()})
        
        # Calculate averages
        num_batches = len(dataloader)
        avg_loss = total_loss / num_batches
        avg_metrics = {name: value / num_batches for name, value in metrics_sum.items()}
        avg_metrics['loss'] = avg_loss
        
        return avg_metrics
    
    def save_checkpoint(self, path, epoch, best_metric=None):
        """
        Save model checkpoint
        
        Args:
            path: Path to save checkpoint
            epoch: Current epoch
            best_metric: Best metric value
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_metric': best_metric
        }
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path):
        """
        Load model checkpoint
        
        Args:
            path: Path to checkpoint
        
        Returns:
            Dictionary with epoch and best_metric
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return {
            'epoch': checkpoint['epoch'],
            'best_metric': checkpoint.get('best_metric', None)
        }
