#!/usr/bin/env python3
"""
Consolidated training function for Globus Compute submission.
This function uses the actual training code directly instead of subprocess calls.
"""

import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any


def cifar100_training_function(*args, **kwargs) -> Dict[str, Any]:
   """
   Training function that runs CIFAR-100 training on Aurora via Globus Compute.
   
   Args:
      job_config: Configuration dictionary containing:
         - job_id: Unique identifier for the job
         - model_type: Type of model to train
         - learning_rate: Learning rate
         - batch_size: Batch size
         - num_epochs: Number of epochs
         - data_dir: Path to CIFAR-100 data
         - output_dir: Output directory for results
         - And other hyperparameters
      
      Or can be called via Globus Compute with function_args=[job_config]
         
   Returns:
      Dict containing training results including AUC
   """
   
   # Handle different calling patterns for Globus Compute compatibility
   if 'function_args' in kwargs:
      # Called via Globus Compute with function_args keyword
      function_args = kwargs['function_args']
      if isinstance(function_args, list) and len(function_args) > 0:
         job_config = function_args[0]
      else:
         raise ValueError("function_args must be a non-empty list containing job_config")
   elif len(args) > 0:
      # Called directly with positional argument
      job_config = args[0]
   elif 'job_config' in kwargs:
      # Called with job_config as keyword argument
      job_config = kwargs['job_config']
   else:
      raise ValueError("No job_config provided. Function must be called with job_config as positional argument, keyword argument, or via function_args list")
   
   # Validate job_config is a dictionary
   if not isinstance(job_config, dict):
      raise TypeError(f"job_config must be a dictionary, got {type(job_config)}")
   
   # Setup logging
   logging.basicConfig(
      level=logging.INFO,
      format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
      datefmt="%d-%m %H:%M"
   )
   logger = logging.getLogger(__name__)
   
   job_id = job_config["job_id"]
   output_dir = job_config["output_dir"]
   
   logger.info(f"Starting training job: {job_id}")
   
   # Record start time
   start_time = datetime.now()
   
   try:
      # Import necessary modules (done inside function for serialization)
      import torch
      import torch.nn as nn
      import torch.optim as optim
      from torch.utils.data import DataLoader
      import socket
      
      # Import Intel PyTorch Extension if available
      try:
         import intel_extension_for_pytorch as ipex
         IPEX_AVAILABLE = True
         logger.info("Intel PyTorch Extension (IPEX) imported successfully")
      except ImportError:
         IPEX_AVAILABLE = False
         logger.warning("Intel PyTorch Extension (IPEX) not available")
      
      # Add project path to sys.path for imports
      repo_path = job_config.get("repo_path", "/lus/flare/projects/datascience/parton/inference_service_test")
      if repo_path not in sys.path:
         sys.path.insert(0, repo_path)
      
      # Import project modules
      from src.training.trainer import Trainer
      from src.training.model import get_model
      from src.utils.data_utils import get_cifar100_dataloaders
      from src.utils.metrics import calculate_auc
      
      # Create output directory
      os.makedirs(output_dir, exist_ok=True)
      
      # Setup Aurora environment
      if 'OMP_NUM_THREADS' not in os.environ:
         os.environ['OMP_NUM_THREADS'] = '1'
      if 'OMP_PLACES' not in os.environ:
         os.environ['OMP_PLACES'] = 'cores'
      if 'OMP_PROC_BIND' not in os.environ:
         os.environ['OMP_PROC_BIND'] = 'close'
      
      # Log system information
      hostname = socket.gethostname()
      logger.info(f"Running on host: {hostname}")
      logger.info(f"PyTorch version: {torch.__version__}")
      logger.info(f"Intel PyTorch Extension available: {IPEX_AVAILABLE}")
      logger.info(f"Intel GPU support: {hasattr(torch, 'xpu')}")
      
      # Extract hyperparameters
      model_type = job_config.get("model_type", "resnet18")
      hidden_size = job_config.get("hidden_size", 1024)
      num_layers = job_config.get("num_layers", 3)
      dropout_rate = job_config.get("dropout_rate", 0.2)
      learning_rate = job_config.get("learning_rate", 0.001)
      batch_size = job_config.get("batch_size", 128)
      num_epochs = job_config.get("num_epochs", 10)
      weight_decay = job_config.get("weight_decay", 1e-4)
      num_workers = job_config.get("num_workers", 4)
      data_dir = job_config.get("data_dir", "data/cifar-100-python")
      
      logger.info(f"Hyperparameters: model={model_type}, lr={learning_rate}, "
                 f"batch={batch_size}, epochs={num_epochs}")
      
      # Check if data directory exists
      if not Path(data_dir).exists():
         raise FileNotFoundError(f"Data directory not found: {data_dir}")
      
      # Get data loaders
      logger.info("Loading CIFAR-100 dataset...")
      train_loader, val_loader, test_loader = get_cifar100_dataloaders(
         data_dir=data_dir,
         batch_size=batch_size,
         num_workers=num_workers
      )
      
      logger.info(f"Dataset loaded - Train: {len(train_loader.dataset)}, "
                 f"Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")
      
      # Create model
      logger.info(f"Creating {model_type} model...")
      model = get_model(
         model_type=model_type,
         num_classes=100,  # CIFAR-100 has 100 classes
         hidden_size=hidden_size,
         num_layers=num_layers,
         dropout_rate=dropout_rate
      )
      
      # Count parameters
      total_params = sum(p.numel() for p in model.parameters())
      trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
      logger.info(f"Model created - Total params: {total_params:,}, "
                 f"Trainable params: {trainable_params:,}")
      
      # Create trainer with device auto-detection
      trainer = Trainer(
         model=model,
         train_loader=train_loader,
         val_loader=val_loader,
         test_loader=test_loader,
         learning_rate=learning_rate,
         weight_decay=weight_decay,
         num_epochs=num_epochs,
         output_dir=Path(output_dir),
         job_id=job_id,
         device_type="auto",  # Let trainer auto-detect best device
         use_ipex=IPEX_AVAILABLE
      )
      
      # Start training
      logger.info("Starting training...")
      training_start = time.time()
      
      training_history = trainer.train()
      
      training_time = time.time() - training_start
      logger.info(f"Training completed in {training_time:.2f} seconds")
      
      # Evaluate on test set
      logger.info("Evaluating on test set...")
      test_accuracy = trainer.evaluate()
      
      # Calculate AUC of accuracy curves
      train_accuracies = [epoch["train_accuracy"] for epoch in training_history]
      val_accuracies = [epoch["val_accuracy"] for epoch in training_history]
      
      train_auc = calculate_auc(train_accuracies)
      val_auc = calculate_auc(val_accuracies)
      
      # Prepare results
      end_time = datetime.now()
      execution_time = (end_time - start_time).total_seconds()
      
      results = {
         "job_id": job_id,
         "status": "completed",
         "hostname": hostname,
         "execution_time": execution_time,
         "training_time": training_time,
         "start_time": start_time.isoformat(),
         "end_time": end_time.isoformat(),
         "hyperparameters": {
            "model_type": model_type,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "dropout_rate": dropout_rate,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "weight_decay": weight_decay,
            "data_dir": data_dir
         },
         "training_results": {
            "final_accuracy": test_accuracy,
            "train_auc": train_auc,
            "val_auc": val_auc,
            "auc": val_auc,  # Primary metric for validation
            "accuracy": test_accuracy  # Alias for compatibility
         },
         "model_info": {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "model_type": model_type
         },
         "system_info": {
            "hostname": hostname,
            "pytorch_version": torch.__version__,
            "intel_pytorch_extension": IPEX_AVAILABLE,
            "intel_gpu_support": hasattr(torch, 'xpu'),
            "device_used": str(trainer.device) if hasattr(trainer, 'device') else 'unknown'
         },
         "training_history": training_history
      }
      
      # Save results to file
      results_file = Path(output_dir) / "results.json"
      with open(results_file, 'w') as f:
         json.dump(results, f, indent=3)
      
      logger.info(f"Training completed successfully!")
      logger.info(f"Final test accuracy: {test_accuracy:.4f}")
      logger.info(f"Training AUC: {train_auc:.4f}")
      logger.info(f"Validation AUC: {val_auc:.4f}")
      logger.info(f"Results saved to: {results_file}")
      
      return results
      
   except Exception as e:
      # Handle errors
      end_time = datetime.now()
      execution_time = (end_time - start_time).total_seconds()
      
      error_results = {
         "job_id": job_id,
         "status": "failed",
         "hostname": socket.gethostname() if 'socket' in locals() else 'unknown',
         "execution_time": execution_time,
         "start_time": start_time.isoformat(),
         "end_time": end_time.isoformat(),
         "error": str(e),
         "error_type": type(e).__name__,
         "hyperparameters": job_config,
         "training_results": {
            "auc": 0.0,  # Failed jobs have 0 AUC
            "accuracy": 0.0
         }
      }
      
      # Save error results
      try:
         error_file = Path(output_dir) / "error_results.json"
         with open(error_file, 'w') as f:
            json.dump(error_results, f, indent=3)
      except:
         pass
      
      logger.error(f"Training failed: {e}")
      return error_results 