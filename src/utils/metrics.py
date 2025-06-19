#!/usr/bin/env python3
"""
Metrics utilities for evaluating model performance.
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import torch


def calculate_auc(accuracies: List[float], normalize: bool = True) -> float:
   """
   Calculate the Area Under the Curve (AUC) of accuracy progression.
   
   Args:
      accuracies: List of accuracy values over epochs
      normalize: Whether to normalize the AUC by the number of epochs
   
   Returns:
      AUC value (normalized if normalize=True)
   """
   
   if not accuracies:
      return 0.0
   
   # Convert to numpy array
   accuracies = np.array(accuracies)
   
   # Calculate AUC using trapezoidal rule
   x = np.arange(len(accuracies))
   auc_value = np.trapz(accuracies, x)
   
   # Normalize by number of epochs if requested
   if normalize and len(accuracies) > 1:
      auc_value /= (len(accuracies) - 1)
   
   return float(auc_value)


def calculate_learning_curve_metrics(accuracies: List[float]) -> Dict[str, float]:
   """
   Calculate various metrics from learning curves.
   
   Args:
      accuracies: List of accuracy values over epochs
   
   Returns:
      Dictionary of learning curve metrics
   """
   
   if not accuracies:
      return {}
   
   accuracies = np.array(accuracies)
   
   metrics = {
      "final_accuracy": float(accuracies[-1]),
      "max_accuracy": float(np.max(accuracies)),
      "min_accuracy": float(np.min(accuracies)),
      "mean_accuracy": float(np.mean(accuracies)),
      "std_accuracy": float(np.std(accuracies)),
      "auc": calculate_auc(accuracies),
      "convergence_epoch": _find_convergence_epoch(accuracies),
      "improvement_rate": _calculate_improvement_rate(accuracies)
   }
   
   return metrics


def _find_convergence_epoch(accuracies: np.ndarray, threshold: float = 0.001, 
                           window_size: int = 5) -> int:
   """
   Find the epoch where accuracy converges.
   
   Args:
      accuracies: Array of accuracy values
      threshold: Threshold for considering convergence
      window_size: Size of window to check for convergence
   
   Returns:
      Epoch number where convergence occurs
   """
   
   if len(accuracies) < window_size:
      return len(accuracies)
   
   # Check for convergence in sliding windows
   for i in range(window_size, len(accuracies)):
      window = accuracies[i-window_size:i]
      if np.max(window) - np.min(window) < threshold:
         return i - window_size + 1
   
   return len(accuracies)


def _calculate_improvement_rate(accuracies: np.ndarray) -> float:
   """
   Calculate the rate of improvement in accuracy.
   
   Args:
      accuracies: Array of accuracy values
   
   Returns:
      Average improvement per epoch
   """
   
   if len(accuracies) < 2:
      return 0.0
   
   # Calculate differences between consecutive epochs
   differences = np.diff(accuracies)
   
   # Return mean improvement rate
   return float(np.mean(differences))


def calculate_classification_metrics(predictions: torch.Tensor, targets: torch.Tensor,
                                   num_classes: int = 100) -> Dict[str, float]:
   """
   Calculate comprehensive classification metrics.
   
   Args:
      predictions: Model predictions (logits or probabilities)
      targets: Ground truth labels
      num_classes: Number of classes
   
   Returns:
      Dictionary of classification metrics
   """
   
   # Convert to numpy if needed
   if isinstance(predictions, torch.Tensor):
      predictions = predictions.detach().cpu().numpy()
   if isinstance(targets, torch.Tensor):
      targets = targets.detach().cpu().numpy()
   
   # Get predicted classes
   if predictions.ndim > 1:
      predicted_classes = np.argmax(predictions, axis=1)
   else:
      predicted_classes = predictions
   
   # Calculate basic metrics
   accuracy = np.mean(predicted_classes == targets)
   
   # Calculate per-class accuracy
   per_class_accuracy = []
   for class_idx in range(num_classes):
      class_mask = targets == class_idx
      if np.sum(class_mask) > 0:
         class_acc = np.mean(predicted_classes[class_mask] == targets[class_mask])
         per_class_accuracy.append(class_acc)
      else:
         per_class_accuracy.append(0.0)
   
   metrics = {
      "accuracy": float(accuracy),
      "mean_per_class_accuracy": float(np.mean(per_class_accuracy)),
      "std_per_class_accuracy": float(np.std(per_class_accuracy)),
      "min_per_class_accuracy": float(np.min(per_class_accuracy)),
      "max_per_class_accuracy": float(np.max(per_class_accuracy))
   }
   
   # Calculate confusion matrix if needed
   if num_classes <= 20:  # Only for reasonable number of classes
      confusion_matrix = _calculate_confusion_matrix(predicted_classes, targets, num_classes)
      metrics["confusion_matrix"] = confusion_matrix.tolist()
   
   return metrics


def _calculate_confusion_matrix(predictions: np.ndarray, targets: np.ndarray, 
                               num_classes: int) -> np.ndarray:
   """
   Calculate confusion matrix.
   
   Args:
      predictions: Predicted class labels
      targets: True class labels
      num_classes: Number of classes
   
   Returns:
      Confusion matrix
   """
   
   confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
   
   for pred, target in zip(predictions, targets):
      confusion_matrix[target, pred] += 1
   
   return confusion_matrix


def calculate_roc_auc(predictions: torch.Tensor, targets: torch.Tensor,
                     num_classes: int = 100) -> Dict[str, float]:
   """
   Calculate ROC AUC scores for multi-class classification.
   
   Args:
      predictions: Model predictions (logits or probabilities)
      targets: Ground truth labels
      num_classes: Number of classes
   
   Returns:
      Dictionary of ROC AUC metrics
   """
   
   # Convert to numpy if needed
   if isinstance(predictions, torch.Tensor):
      predictions = predictions.detach().cpu().numpy()
   if isinstance(targets, torch.Tensor):
      targets = targets.detach().cpu().numpy()
   
   # Ensure predictions are probabilities
   if predictions.ndim == 1:
      # Binary case
      try:
         roc_auc = roc_auc_score(targets, predictions)
         return {"roc_auc": float(roc_auc)}
      except ValueError:
         return {"roc_auc": 0.0}
   
   # Multi-class case
   try:
      # One-vs-rest ROC AUC
      roc_auc_ovr = roc_auc_score(targets, predictions, multi_class='ovr', average='macro')
      
      # One-vs-one ROC AUC
      roc_auc_ovo = roc_auc_score(targets, predictions, multi_class='ovo', average='macro')
      
      # Per-class ROC AUC
      roc_auc_per_class = roc_auc_score(targets, predictions, multi_class='ovr', average=None)
      
      return {
         "roc_auc_ovr": float(roc_auc_ovr),
         "roc_auc_ovo": float(roc_auc_ovo),
         "roc_auc_per_class": roc_auc_per_class.tolist(),
         "mean_roc_auc_per_class": float(np.mean(roc_auc_per_class)),
         "std_roc_auc_per_class": float(np.std(roc_auc_per_class))
      }
   
   except ValueError:
      return {
         "roc_auc_ovr": 0.0,
         "roc_auc_ovo": 0.0,
         "roc_auc_per_class": [0.0] * num_classes,
         "mean_roc_auc_per_class": 0.0,
         "std_roc_auc_per_class": 0.0
      }


def calculate_precision_recall_auc(predictions: torch.Tensor, targets: torch.Tensor,
                                  num_classes: int = 100) -> Dict[str, float]:
   """
   Calculate Precision-Recall AUC scores.
   
   Args:
      predictions: Model predictions (logits or probabilities)
      targets: Ground truth labels
      num_classes: Number of classes
   
   Returns:
      Dictionary of PR AUC metrics
   """
   
   # Convert to numpy if needed
   if isinstance(predictions, torch.Tensor):
      predictions = predictions.detach().cpu().numpy()
   if isinstance(targets, torch.Tensor):
      targets = targets.detach().cpu().numpy()
   
   if predictions.ndim == 1:
      # Binary case
      try:
         precision, recall, _ = precision_recall_curve(targets, predictions)
         pr_auc = auc(recall, precision)
         return {"pr_auc": float(pr_auc)}
      except ValueError:
         return {"pr_auc": 0.0}
   
   # Multi-class case
   pr_auc_scores = []
   
   for class_idx in range(num_classes):
      try:
         # Create binary targets for this class
         binary_targets = (targets == class_idx).astype(int)
         class_predictions = predictions[:, class_idx]
         
         precision, recall, _ = precision_recall_curve(binary_targets, class_predictions)
         pr_auc = auc(recall, precision)
         pr_auc_scores.append(pr_auc)
      except ValueError:
         pr_auc_scores.append(0.0)
   
   return {
      "pr_auc_per_class": pr_auc_scores,
      "mean_pr_auc": float(np.mean(pr_auc_scores)),
      "std_pr_auc": float(np.std(pr_auc_scores))
   }


def calculate_training_efficiency_metrics(training_history: List[Dict[str, Any]]) -> Dict[str, float]:
   """
   Calculate training efficiency metrics.
   
   Args:
      training_history: List of epoch results from training
   
   Returns:
      Dictionary of efficiency metrics
   """
   
   if not training_history:
      return {}
   
   # Extract metrics
   train_accuracies = [epoch["train_accuracy"] for epoch in training_history]
   val_accuracies = [epoch["val_accuracy"] for epoch in training_history]
   epoch_times = [epoch.get("epoch_time", 0) for epoch in training_history]
   
   # Calculate efficiency metrics
   total_time = sum(epoch_times)
   final_accuracy = val_accuracies[-1] if val_accuracies else 0.0
   
   # Accuracy per time unit
   accuracy_per_second = final_accuracy / total_time if total_time > 0 else 0.0
   
   # Convergence speed
   convergence_epoch = _find_convergence_epoch(np.array(val_accuracies))
   convergence_time = sum(epoch_times[:convergence_epoch]) if convergence_epoch <= len(epoch_times) else total_time
   
   # Learning efficiency (improvement per epoch)
   if len(val_accuracies) > 1:
      learning_efficiency = (val_accuracies[-1] - val_accuracies[0]) / len(val_accuracies)
   else:
      learning_efficiency = 0.0
   
   return {
      "total_training_time": float(total_time),
      "final_accuracy": float(final_accuracy),
      "accuracy_per_second": float(accuracy_per_second),
      "convergence_epoch": int(convergence_epoch),
      "convergence_time": float(convergence_time),
      "learning_efficiency": float(learning_efficiency),
      "mean_epoch_time": float(np.mean(epoch_times)) if epoch_times else 0.0,
      "std_epoch_time": float(np.std(epoch_times)) if epoch_times else 0.0
   }


def calculate_overfitting_metrics(train_accuracies: List[float], 
                                 val_accuracies: List[float]) -> Dict[str, float]:
   """
   Calculate overfitting metrics.
   
   Args:
      train_accuracies: Training accuracy values
      val_accuracies: Validation accuracy values
   
   Returns:
      Dictionary of overfitting metrics
   """
   
   if not train_accuracies or not val_accuracies:
      return {}
   
   train_acc = np.array(train_accuracies)
   val_acc = np.array(val_accuracies)
   
   # Calculate overfitting metrics
   max_gap = np.max(train_acc - val_acc)
   final_gap = train_acc[-1] - val_acc[-1]
   mean_gap = np.mean(train_acc - val_acc)
   
   # Overfitting severity (higher values indicate more overfitting)
   overfitting_severity = max_gap / np.mean(val_acc) if np.mean(val_acc) > 0 else 0.0
   
   # Stability metric (lower values indicate more stable training)
   train_stability = np.std(train_acc)
   val_stability = np.std(val_acc)
   
   return {
      "max_accuracy_gap": float(max_gap),
      "final_accuracy_gap": float(final_gap),
      "mean_accuracy_gap": float(mean_gap),
      "overfitting_severity": float(overfitting_severity),
      "train_stability": float(train_stability),
      "val_stability": float(val_stability),
      "stability_ratio": float(train_stability / val_stability) if val_stability > 0 else 0.0
   }


def aggregate_metrics(metrics_list: List[Dict[str, Any]]) -> Dict[str, Any]:
   """
   Aggregate metrics across multiple runs.
   
   Args:
      metrics_list: List of metric dictionaries from different runs
   
   Returns:
      Aggregated metrics with statistics
   """
   
   if not metrics_list:
      return {}
   
   # Collect all metric names
   all_metrics = set()
   for metrics in metrics_list:
      all_metrics.update(metrics.keys())
   
   aggregated = {}
   
   for metric_name in all_metrics:
      values = []
      for metrics in metrics_list:
         if metric_name in metrics:
            value = metrics[metric_name]
            if isinstance(value, (int, float)):
               values.append(value)
      
      if values:
         aggregated[metric_name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "median": float(np.median(values)),
            "count": len(values)
         } 