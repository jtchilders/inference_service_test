#!/usr/bin/env python3
"""
Data utilities for CIFAR-100 dataset loading and processing.
"""

import json
import logging
import os
from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import numpy as np


def get_cifar100_dataloaders(data_dir: str, batch_size: int = 128, 
                           num_workers: int = 4, val_split: float = 0.1,
                           augment: bool = True) -> Tuple[DataLoader, DataLoader, DataLoader]:
   """
   Create train, validation, and test dataloaders for CIFAR-100.
   
   Args:
      data_dir: Path to CIFAR-100 data directory
      batch_size: Batch size for dataloaders
      num_workers: Number of workers for data loading
      val_split: Fraction of training data to use for validation
      augment: Whether to apply data augmentation to training data
   
   Returns:
      Tuple of (train_loader, val_loader, test_loader)
   """
   
   logger = logging.getLogger(__name__)
   
   # Define transforms
   if augment:
      train_transform = transforms.Compose([
         transforms.RandomCrop(32, padding=4),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
      ])
   else:
      train_transform = transforms.Compose([
         transforms.ToTensor(),
         transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
      ])
   
   test_transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
   ])
   
   # Load datasets
   logger.info(f"Loading CIFAR-100 from {data_dir}")
   
   try:
      train_dataset = torchvision.datasets.CIFAR100(
         root=data_dir,
         train=True,
         download=False,  # Assume data is already downloaded
         transform=train_transform
      )
      
      test_dataset = torchvision.datasets.CIFAR100(
         root=data_dir,
         train=False,
         download=False,
         transform=test_transform
      )
      
   except Exception as e:
      logger.error(f"Error loading CIFAR-100 dataset: {e}")
      raise
   
   # Split training data into train and validation
   train_size = int((1 - val_split) * len(train_dataset))
   val_size = len(train_dataset) - train_size
   
   train_dataset, val_dataset = random_split(
      train_dataset, [train_size, val_size],
      generator=torch.Generator().manual_seed(42)
   )
   
   # Create dataloaders
   train_loader = DataLoader(
      train_dataset,
      batch_size=batch_size,
      shuffle=True,
      num_workers=num_workers,
      pin_memory=True
   )
   
   val_loader = DataLoader(
      val_dataset,
      batch_size=batch_size,
      shuffle=False,
      num_workers=num_workers,
      pin_memory=True
   )
   
   test_loader = DataLoader(
      test_dataset,
      batch_size=batch_size,
      shuffle=False,
      num_workers=num_workers,
      pin_memory=True
   )
   
   logger.info(f"Created dataloaders - Train: {len(train_dataset)}, "
              f"Val: {len(val_dataset)}, Test: {len(test_dataset)}")
   
   return train_loader, val_loader, test_loader


def load_training_results(results_dir: str) -> List[Dict[str, Any]]:
   """
   Load training results from multiple job directories.
   
   Args:
      results_dir: Directory containing job result directories
   
   Returns:
      List of training result dictionaries
   """
   
   logger = logging.getLogger(__name__)
   results = []
   
   results_path = Path(results_dir)
   if not results_path.exists():
      logger.warning(f"Results directory does not exist: {results_dir}")
      return results
   
   # Find all job directories
   job_dirs = [d for d in results_path.iterdir() if d.is_dir() and d.name.startswith("job_")]
   
   for job_dir in job_dirs:
      results_file = job_dir / "results.json"
      
      if results_file.exists():
         try:
            with open(results_file, 'r') as f:
               job_results = json.load(f)
               results.append(job_results)
               logger.debug(f"Loaded results from {job_dir}")
         except Exception as e:
            logger.error(f"Error loading results from {job_dir}: {e}")
      else:
         logger.warning(f"No results.json found in {job_dir}")
   
   logger.info(f"Loaded {len(results)} training results")
   return results


def parse_hyperparameters(results: List[Dict[str, Any]]) -> Dict[str, List]:
   """
   Parse hyperparameters from training results.
   
   Args:
      results: List of training result dictionaries
   
   Returns:
      Dictionary mapping hyperparameter names to lists of values
   """
   
   hyperparams = {}
   
   for result in results:
      if "hyperparameters" in result:
         for key, value in result["hyperparameters"].items():
            if key not in hyperparams:
               hyperparams[key] = []
            hyperparams[key].append(value)
   
   return hyperparams


def extract_metrics(results: List[Dict[str, Any]]) -> Dict[str, List]:
   """
   Extract metrics from training results.
   
   Args:
      results: List of training result dictionaries
   
   Returns:
      Dictionary mapping metric names to lists of values
   """
   
   metrics = {
      "final_accuracy": [],
      "train_auc": [],
      "val_auc": [],
      "training_time": [],
      "total_params": [],
      "trainable_params": []
   }
   
   for result in results:
      if "final_accuracy" in result:
         metrics["final_accuracy"].append(result["final_accuracy"])
      
      if "train_auc" in result:
         metrics["train_auc"].append(result["train_auc"])
      
      if "val_auc" in result:
         metrics["val_auc"].append(result["val_auc"])
      
      if "training_time" in result:
         metrics["training_time"].append(result["training_time"])
      
      if "model_info" in result:
         model_info = result["model_info"]
         if "total_params" in model_info:
            metrics["total_params"].append(model_info["total_params"])
         if "trainable_params" in model_info:
            metrics["trainable_params"].append(model_info["trainable_params"])
   
   return metrics


def get_best_configuration(results: List[Dict[str, Any]], 
                          metric: str = "final_accuracy") -> Optional[Dict[str, Any]]:
   """
   Find the best configuration based on a metric.
   
   Args:
      results: List of training result dictionaries
      metric: Metric to optimize (default: "final_accuracy")
   
   Returns:
      Best configuration dictionary or None if no results
   """
   
   if not results:
      return None
   
   # Find the result with the best metric
   best_result = max(results, key=lambda x: x.get(metric, 0))
   
   return {
      "hyperparameters": best_result.get("hyperparameters", {}),
      "metrics": {
         "final_accuracy": best_result.get("final_accuracy", 0),
         "train_auc": best_result.get("train_auc", 0),
         "val_auc": best_result.get("val_auc", 0),
         "training_time": best_result.get("training_time", 0)
      },
      "job_id": best_result.get("job_id", "unknown")
   }


def analyze_training_history(results: List[Dict[str, Any]]) -> Dict[str, Any]:
   """
   Analyze training history across multiple runs.
   
   Args:
      results: List of training result dictionaries
   
   Returns:
      Analysis dictionary with statistics
   """
   
   if not results:
      return {}
   
   # Extract metrics
   metrics = extract_metrics(results)
   
   # Calculate statistics
   analysis = {}
   
   for metric_name, values in metrics.items():
      if values:
         analysis[metric_name] = {
            "mean": np.mean(values),
            "std": np.std(values),
            "min": np.min(values),
            "max": np.max(values),
            "median": np.median(values)
         }
   
   # Find best and worst configurations
   if "final_accuracy" in metrics and metrics["final_accuracy"]:
      best_idx = np.argmax(metrics["final_accuracy"])
      worst_idx = np.argmin(metrics["final_accuracy"])
      
      analysis["best_configuration"] = results[best_idx]
      analysis["worst_configuration"] = results[worst_idx]
   
   return analysis


def save_analysis_report(analysis: Dict[str, Any], output_path: str) -> None:
   """
   Save analysis report to a JSON file.
   
   Args:
      analysis: Analysis dictionary
      output_path: Path to save the report
   """
   
   logger = logging.getLogger(__name__)
   
   try:
      with open(output_path, 'w') as f:
         json.dump(analysis, f, indent=3, default=str)
      
      logger.info(f"Analysis report saved to {output_path}")
      
   except Exception as e:
      logger.error(f"Error saving analysis report: {e}")


def get_dataset_info(data_dir: str) -> Dict[str, Any]:
   """
   Get information about the CIFAR-100 dataset.
   
   Args:
      data_dir: Path to CIFAR-100 data directory
   
   Returns:
      Dictionary with dataset information
   """
   
   logger = logging.getLogger(__name__)
   
   try:
      # Load a small sample to get dataset info
      transform = transforms.ToTensor()
      dataset = torchvision.datasets.CIFAR100(
         root=data_dir,
         train=True,
         download=False,
         transform=transform
      )
      
      # Get sample data
      sample_data, sample_label = dataset[0]
      
      info = {
         "num_classes": 100,
         "image_size": (32, 32),
         "channels": 3,
         "train_size": len(dataset),
         "sample_shape": sample_data.shape,
         "sample_label": sample_label,
         "class_names": dataset.classes[:10] + ["..."] + dataset.classes[-10:]  # Show first and last 10
      }
      
      # Get test dataset size
      test_dataset = torchvision.datasets.CIFAR100(
         root=data_dir,
         train=False,
         download=False,
         transform=transform
      )
      info["test_size"] = len(test_dataset)
      
      logger.info(f"Dataset info: {info}")
      return info
      
   except Exception as e:
      logger.error(f"Error getting dataset info: {e}")
      return {}


def create_data_summary(results_dir: str, output_path: str) -> None:
   """
   Create a comprehensive summary of all training results.
   
   Args:
      results_dir: Directory containing training results
      output_path: Path to save the summary
   """
   
   logger = logging.getLogger(__name__)
   
   # Load all results
   results = load_training_results(results_dir)
   
   if not results:
      logger.warning("No results found to summarize")
      return
   
   # Create summary
   summary = {
      "total_runs": len(results),
      "analysis": analyze_training_history(results),
      "best_configuration": get_best_configuration(results),
      "hyperparameter_ranges": parse_hyperparameters(results),
      "timestamp": str(Path().cwd() / results_dir)
   }
   
   # Save summary
   save_analysis_report(summary, output_path) 