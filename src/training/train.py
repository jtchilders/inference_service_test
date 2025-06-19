#!/usr/bin/env python3
"""
Main training script for CIFAR-100 classification.
Entry point that loads hyperparameters and kicks off a single training run.
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from training.trainer import Trainer
from training.model import get_model
from utils.data_utils import get_cifar100_dataloaders
from utils.metrics import calculate_auc


def setup_logging(log_level: str = "INFO") -> None:
   """Setup logging configuration."""
   log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
   date_format = "%d-%m %H:%M"
   
   logging.basicConfig(
      level=getattr(logging, log_level.upper()),
      format=log_format,
      datefmt=date_format,
      handlers=[
         logging.StreamHandler(sys.stdout)
      ]
   )


def parse_arguments() -> argparse.Namespace:
   """Parse command line arguments."""
   parser = argparse.ArgumentParser(description="CIFAR-100 Training Script")
   
   # Model parameters
   parser.add_argument("--model_type", type=str, default="resnet18",
                      choices=["resnet18", "resnet34", "resnet50", "vgg16", "densenet121"],
                      help="Type of model to train")
   parser.add_argument("--hidden_size", type=int, default=1024,
                      help="Hidden layer size for custom models")
   parser.add_argument("--num_layers", type=int, default=3,
                      help="Number of layers for custom models")
   parser.add_argument("--dropout_rate", type=float, default=0.2,
                      help="Dropout rate")
   
   # Training parameters
   parser.add_argument("--learning_rate", type=float, default=0.001,
                      help="Learning rate")
   parser.add_argument("--batch_size", type=int, default=128,
                      help="Batch size")
   parser.add_argument("--num_epochs", type=int, default=50,
                      help="Number of training epochs")
   parser.add_argument("--weight_decay", type=float, default=1e-4,
                      help="Weight decay")
   
   # Data parameters
   parser.add_argument("--data_dir", type=str, default="data/cifar-100-python",
                      help="Path to CIFAR-100 data directory")
   parser.add_argument("--num_workers", type=int, default=4,
                      help="Number of data loading workers")
   
   # Output parameters
   parser.add_argument("--output_dir", type=str, required=True,
                      help="Output directory for results")
   parser.add_argument("--job_id", type=str, required=True,
                      help="Job ID for tracking")
   
   # Logging
   parser.add_argument("--log_level", type=str, default="INFO",
                      choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                      help="Logging level")
   
   return parser.parse_args()


def save_results(results: Dict[str, Any], output_dir: str, job_id: str) -> None:
   """Save training results to JSON file."""
   
   output_path = Path(output_dir) / "results.json"
   
   # Add metadata
   results["job_id"] = job_id
   results["timestamp"] = datetime.now().isoformat()
   results["output_dir"] = output_dir
   
   # Save results
   with open(output_path, 'w') as f:
      json.dump(results, f, indent=3)
   
   logging.info(f"Results saved to: {output_path}")


def main():
   """Main training function."""
   
   # Parse arguments
   args = parse_arguments()
   
   # Setup logging
   setup_logging(args.log_level)
   logger = logging.getLogger(__name__)
   
   # Create output directory
   output_dir = Path(args.output_dir)
   output_dir.mkdir(parents=True, exist_ok=True)
   
   logger.info(f"Starting CIFAR-100 training job: {args.job_id}")
   logger.info(f"Output directory: {output_dir}")
   
   # Log hyperparameters
   hyperparams = {
      "model_type": args.model_type,
      "hidden_size": args.hidden_size,
      "num_layers": args.num_layers,
      "dropout_rate": args.dropout_rate,
      "learning_rate": args.learning_rate,
      "batch_size": args.batch_size,
      "num_epochs": args.num_epochs,
      "weight_decay": args.weight_decay,
      "data_dir": args.data_dir,
      "num_workers": args.num_workers
   }
   
   logger.info(f"Hyperparameters: {hyperparams}")
   
   try:
      # Check if data directory exists
      if not Path(args.data_dir).exists():
         raise FileNotFoundError(f"Data directory not found: {args.data_dir}")
      
      # Get data loaders
      logger.info("Loading CIFAR-100 dataset...")
      train_loader, val_loader, test_loader = get_cifar100_dataloaders(
         data_dir=args.data_dir,
         batch_size=args.batch_size,
         num_workers=args.num_workers
      )
      
      logger.info(f"Dataset loaded - Train: {len(train_loader.dataset)}, "
                 f"Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")
      
      # Create model
      logger.info(f"Creating {args.model_type} model...")
      model = get_model(
         model_type=args.model_type,
         num_classes=100,  # CIFAR-100 has 100 classes
         hidden_size=args.hidden_size,
         num_layers=args.num_layers,
         dropout_rate=args.dropout_rate
      )
      
      # Count parameters
      total_params = sum(p.numel() for p in model.parameters())
      trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
      logger.info(f"Model created - Total params: {total_params:,}, "
                 f"Trainable params: {trainable_params:,}")
      
      # Create trainer
      trainer = Trainer(
         model=model,
         train_loader=train_loader,
         val_loader=val_loader,
         test_loader=test_loader,
         learning_rate=args.learning_rate,
         weight_decay=args.weight_decay,
         num_epochs=args.num_epochs,
         output_dir=output_dir,
         job_id=args.job_id
      )
      
      # Start training
      logger.info("Starting training...")
      start_time = time.time()
      
      training_history = trainer.train()
      
      training_time = time.time() - start_time
      logger.info(f"Training completed in {training_time:.2f} seconds")
      
      # Evaluate on test set
      logger.info("Evaluating on test set...")
      test_accuracy = trainer.evaluate()
      
      # Calculate AUC of accuracy curve
      train_accuracies = [epoch["train_accuracy"] for epoch in training_history]
      val_accuracies = [epoch["val_accuracy"] for epoch in training_history]
      
      train_auc = calculate_auc(train_accuracies)
      val_auc = calculate_auc(val_accuracies)
      
      # Prepare results
      results = {
         "hyperparameters": hyperparams,
         "final_accuracy": test_accuracy,
         "train_auc": train_auc,
         "val_auc": val_auc,
         "training_time": training_time,
         "training_history": training_history,
         "model_info": {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "model_type": args.model_type
         }
      }
      
      # Save results
      save_results(results, str(output_dir), args.job_id)
      
      logger.info(f"Training completed successfully!")
      logger.info(f"Final test accuracy: {test_accuracy:.4f}")
      logger.info(f"Training AUC: {train_auc:.4f}")
      logger.info(f"Validation AUC: {val_auc:.4f}")
      
   except Exception as e:
      logger.error(f"Training failed: {e}")
      
      # Save error information
      error_results = {
         "job_id": args.job_id,
         "timestamp": datetime.now().isoformat(),
         "error": str(e),
         "hyperparameters": hyperparams
      }
      
      save_results(error_results, str(output_dir), args.job_id)
      
      sys.exit(1)


if __name__ == "__main__":
   main() 