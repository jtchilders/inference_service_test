#!/usr/bin/env python3
"""
Trainer class for CIFAR-100 model training.
Handles training loop, validation, and logging.
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Any, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm


class Trainer:
   """Trainer class for CIFAR-100 classification models."""
   
   def __init__(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                test_loader: DataLoader, learning_rate: float = 0.001, weight_decay: float = 1e-4,
                num_epochs: int = 50, output_dir: Path = None, job_id: str = None):
      
      self.model = model
      self.train_loader = train_loader
      self.val_loader = val_loader
      self.test_loader = test_loader
      self.learning_rate = learning_rate
      self.weight_decay = weight_decay
      self.num_epochs = num_epochs
      self.output_dir = Path(output_dir) if output_dir else Path("outputs")
      self.job_id = job_id
      
      self.logger = logging.getLogger(__name__)
      
      # Setup device
      self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      self.model.to(self.device)
      
      # Setup optimizer and scheduler
      self.optimizer = optim.Adam(
         self.model.parameters(),
         lr=learning_rate,
         weight_decay=weight_decay
      )
      
      self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
         self.optimizer,
         mode='max',
         factor=0.5,
         patience=5,
         verbose=True
      )
      
      # Setup loss function
      self.criterion = nn.CrossEntropyLoss()
      
      # Setup tensorboard writer
      self.writer = SummaryWriter(log_dir=str(self.output_dir / "tensorboard"))
      
      # Training history
      self.training_history = []
      
      # Best model tracking
      self.best_val_accuracy = 0.0
      self.best_model_path = self.output_dir / "best_model.pth"
      
      # Create output directory
      self.output_dir.mkdir(parents=True, exist_ok=True)
   
   def train(self) -> List[Dict[str, Any]]:
      """Train the model."""
      
      self.logger.info(f"Starting training for {self.num_epochs} epochs")
      self.logger.info(f"Device: {self.device}")
      self.logger.info(f"Learning rate: {self.learning_rate}")
      self.logger.info(f"Weight decay: {self.weight_decay}")
      
      for epoch in range(self.num_epochs):
         epoch_start_time = time.time()
         
         # Training phase
         train_loss, train_accuracy = self._train_epoch(epoch)
         
         # Validation phase
         val_loss, val_accuracy = self._validate_epoch(epoch)
         
         # Update learning rate
         self.scheduler.step(val_accuracy)
         
         # Record epoch results
         epoch_time = time.time() - epoch_start_time
         epoch_results = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "learning_rate": self.optimizer.param_groups[0]['lr'],
            "epoch_time": epoch_time
         }
         
         self.training_history.append(epoch_results)
         
         # Log results
         self._log_epoch_results(epoch_results)
         
         # Save best model
         if val_accuracy > self.best_val_accuracy:
            self.best_val_accuracy = val_accuracy
            self._save_best_model()
            self.logger.info(f"New best validation accuracy: {val_accuracy:.4f}")
         
         # Save checkpoint
         if (epoch + 1) % 10 == 0:
            self._save_checkpoint(epoch + 1)
      
      # Close tensorboard writer
      self.writer.close()
      
      self.logger.info("Training completed!")
      return self.training_history
   
   def evaluate(self) -> float:
      """Evaluate the model on the test set."""
      
      self.logger.info("Evaluating on test set...")
      
      # Load best model
      if self.best_model_path.exists():
         self.model.load_state_dict(torch.load(self.best_model_path, map_location=self.device))
         self.logger.info("Loaded best model for evaluation")
      
      self.model.eval()
      test_loss = 0.0
      correct = 0
      total = 0
      
      with torch.no_grad():
         for batch_idx, (data, target) in enumerate(tqdm(self.test_loader, desc="Testing")):
            data, target = data.to(self.device), target.to(self.device)
            
            output = self.model(data)
            loss = self.criterion(output, target)
            
            test_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
      
      test_accuracy = 100.0 * correct / total
      test_loss /= len(self.test_loader)
      
      self.logger.info(f"Test Loss: {test_loss:.4f}")
      self.logger.info(f"Test Accuracy: {test_accuracy:.2f}%")
      
      return test_accuracy / 100.0  # Return as fraction
   
   def _train_epoch(self, epoch: int) -> tuple:
      """Train for one epoch."""
      
      self.model.train()
      train_loss = 0.0
      correct = 0
      total = 0
      
      pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs} [Train]")
      
      for batch_idx, (data, target) in enumerate(pbar):
         data, target = data.to(self.device), target.to(self.device)
         
         self.optimizer.zero_grad()
         output = self.model(data)
         loss = self.criterion(output, target)
         loss.backward()
         self.optimizer.step()
         
         train_loss += loss.item()
         _, predicted = output.max(1)
         total += target.size(0)
         correct += predicted.eq(target).sum().item()
         
         # Update progress bar
         accuracy = 100.0 * correct / total
         pbar.set_postfix({
            'Loss': f'{train_loss/(batch_idx+1):.4f}',
            'Acc': f'{accuracy:.2f}%'
         })
      
      train_loss /= len(self.train_loader)
      train_accuracy = 100.0 * correct / total
      
      return train_loss, train_accuracy / 100.0  # Return accuracy as fraction
   
   def _validate_epoch(self, epoch: int) -> tuple:
      """Validate for one epoch."""
      
      self.model.eval()
      val_loss = 0.0
      correct = 0
      total = 0
      
      with torch.no_grad():
         pbar = tqdm(self.val_loader, desc=f"Epoch {epoch+1}/{self.num_epochs} [Val]")
         
         for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            output = self.model(data)
            loss = self.criterion(output, target)
            
            val_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            # Update progress bar
            accuracy = 100.0 * correct / total
            pbar.set_postfix({
               'Loss': f'{val_loss/(batch_idx+1):.4f}',
               'Acc': f'{accuracy:.2f}%'
            })
      
      val_loss /= len(self.val_loader)
      val_accuracy = 100.0 * correct / total
      
      return val_loss, val_accuracy / 100.0  # Return accuracy as fraction
   
   def _log_epoch_results(self, results: Dict[str, Any]) -> None:
      """Log epoch results to tensorboard and console."""
      
      epoch = results["epoch"]
      
      # Log to tensorboard
      self.writer.add_scalar('Loss/Train', results["train_loss"], epoch)
      self.writer.add_scalar('Loss/Validation', results["val_loss"], epoch)
      self.writer.add_scalar('Accuracy/Train', results["train_accuracy"], epoch)
      self.writer.add_scalar('Accuracy/Validation', results["val_accuracy"], epoch)
      self.writer.add_scalar('Learning_Rate', results["learning_rate"], epoch)
      self.writer.add_scalar('Time/Epoch', results["epoch_time"], epoch)
      
      # Log to console
      self.logger.info(
         f"Epoch {epoch:3d}/{self.num_epochs} | "
         f"Train Loss: {results['train_loss']:.4f} | "
         f"Train Acc: {results['train_accuracy']:.4f} | "
         f"Val Loss: {results['val_loss']:.4f} | "
         f"Val Acc: {results['val_accuracy']:.4f} | "
         f"LR: {results['learning_rate']:.6f} | "
         f"Time: {results['epoch_time']:.2f}s"
      )
   
   def _save_best_model(self) -> None:
      """Save the best model based on validation accuracy."""
      
      torch.save(self.model.state_dict(), self.best_model_path)
      self.logger.debug(f"Saved best model to {self.best_model_path}")
   
   def _save_checkpoint(self, epoch: int) -> None:
      """Save a training checkpoint."""
      
      checkpoint = {
         'epoch': epoch,
         'model_state_dict': self.model.state_dict(),
         'optimizer_state_dict': self.optimizer.state_dict(),
         'scheduler_state_dict': self.scheduler.state_dict(),
         'best_val_accuracy': self.best_val_accuracy,
         'training_history': self.training_history,
         'hyperparameters': {
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'num_epochs': self.num_epochs
         }
      }
      
      checkpoint_path = self.output_dir / f"checkpoint_epoch_{epoch}.pth"
      torch.save(checkpoint, checkpoint_path)
      self.logger.info(f"Saved checkpoint to {checkpoint_path}")
   
   def load_checkpoint(self, checkpoint_path: str) -> int:
      """Load a training checkpoint."""
      
      checkpoint = torch.load(checkpoint_path, map_location=self.device)
      
      self.model.load_state_dict(checkpoint['model_state_dict'])
      self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
      self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
      self.best_val_accuracy = checkpoint['best_val_accuracy']
      self.training_history = checkpoint['training_history']
      
      self.logger.info(f"Loaded checkpoint from {checkpoint_path}")
      return checkpoint['epoch']
   
   def get_training_summary(self) -> Dict[str, Any]:
      """Get a summary of the training results."""
      
      if not self.training_history:
         return {}
      
      # Calculate statistics
      train_accuracies = [epoch["train_accuracy"] for epoch in self.training_history]
      val_accuracies = [epoch["val_accuracy"] for epoch in self.training_history]
      train_losses = [epoch["train_loss"] for epoch in self.training_history]
      val_losses = [epoch["val_loss"] for epoch in self.training_history]
      
      summary = {
         "best_val_accuracy": self.best_val_accuracy,
         "final_train_accuracy": train_accuracies[-1],
         "final_val_accuracy": val_accuracies[-1],
         "max_train_accuracy": max(train_accuracies),
         "max_val_accuracy": max(val_accuracies),
         "min_train_loss": min(train_losses),
         "min_val_loss": min(val_losses),
         "total_training_time": sum(epoch["epoch_time"] for epoch in self.training_history),
         "num_epochs": len(self.training_history),
         "convergence_epoch": self._find_convergence_epoch(val_accuracies)
      }
      
      return summary
   
   def _find_convergence_epoch(self, accuracies: List[float], threshold: float = 0.001) -> int:
      """Find the epoch where validation accuracy converges."""
      
      if len(accuracies) < 5:
         return len(accuracies)
      
      # Look for plateau in the last 5 epochs
      recent_accuracies = accuracies[-5:]
      max_acc = max(recent_accuracies)
      min_acc = min(recent_accuracies)
      
      if max_acc - min_acc < threshold:
         # Find the first epoch where accuracy is close to the plateau
         for i, acc in enumerate(accuracies):
            if abs(acc - max_acc) < threshold:
               return i + 1
      
      return len(accuracies)
   
   def plot_training_curves(self, save_path: Optional[str] = None) -> None:
      """Plot training curves."""
      
      try:
         import matplotlib.pyplot as plt
         
         epochs = [epoch["epoch"] for epoch in self.training_history]
         train_acc = [epoch["train_accuracy"] for epoch in self.training_history]
         val_acc = [epoch["val_accuracy"] for epoch in self.training_history]
         train_loss = [epoch["train_loss"] for epoch in self.training_history]
         val_loss = [epoch["val_loss"] for epoch in self.training_history]
         
         fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
         
         # Accuracy plot
         ax1.plot(epochs, train_acc, label='Train Accuracy', marker='o')
         ax1.plot(epochs, val_acc, label='Validation Accuracy', marker='s')
         ax1.set_xlabel('Epoch')
         ax1.set_ylabel('Accuracy')
         ax1.set_title('Training and Validation Accuracy')
         ax1.legend()
         ax1.grid(True)
         
         # Loss plot
         ax2.plot(epochs, train_loss, label='Train Loss', marker='o')
         ax2.plot(epochs, val_loss, label='Validation Loss', marker='s')
         ax2.set_xlabel('Epoch')
         ax2.set_ylabel('Loss')
         ax2.set_title('Training and Validation Loss')
         ax2.legend()
         ax2.grid(True)
         
         plt.tight_layout()
         
         if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Training curves saved to {save_path}")
         else:
            plt.show()
         
         plt.close()
         
      except ImportError:
         self.logger.warning("Matplotlib not available, skipping training curves plot") 