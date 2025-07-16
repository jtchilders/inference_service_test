#!/usr/bin/env python3
"""
Test script for validating Globus Compute training pipeline.
Submits parallel training jobs with random hyperparameters to Aurora via Globus Compute.
"""

import argparse
import json
import logging
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import numpy as np

# Add project path for imports
sys.path.append(str(Path(__file__).parent.parent))

try:
   from globus_compute_sdk import Client
   GLOBUS_COMPUTE_AVAILABLE = True
except ImportError:
   GLOBUS_COMPUTE_AVAILABLE = False
   Client = None

from src.training.training_function import cifar100_training_function


def setup_logging(log_level: str = "INFO") -> None:
   """Setup logging configuration."""
   log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
   date_format = "%d-%m %H:%M"
   
   logging.basicConfig(
      level=getattr(logging, log_level.upper()),
      format=log_format,
      datefmt=date_format,
      handlers=[logging.StreamHandler(sys.stdout)]
   )


def generate_random_hyperparameters(job_id: str, base_data_dir: str = "/lus/flare/projects/datascience/parton/data") -> Dict[str, Any]:
   """Generate random hyperparameters for training."""
   
   # Learning rate: log scale between 1e-4 and 1e-2
   learning_rate = 10 ** random.uniform(-4, -2)
   
   # Batch size: choose from common values
   batch_size = random.choice([32, 64, 128])
   
   # Model type: choose from available models
   model_type = random.choice(["resnet18", "custom_cnn"])
   
   # Optimizer: choose from available optimizers
   optimizer = random.choice(["adam", "sgd"])
   
   # Dropout rate: uniform between 0.1 and 0.5
   dropout_rate = random.uniform(0.1, 0.5)
   
   # Weight decay: log scale between 1e-5 and 1e-3
   weight_decay = 10 ** random.uniform(-5, -3)
   
   # Hidden size for custom models
   hidden_size = random.choice([512, 1024, 2048])
   
   # Number of layers for custom models
   num_layers = random.choice([2, 3, 4])
   
   return {
      "job_id": job_id,
      "model_type": model_type,
      "learning_rate": round(learning_rate, 6),
      "batch_size": batch_size,
      "optimizer": optimizer,
      "dropout_rate": round(dropout_rate, 3),
      "weight_decay": weight_decay,
      "hidden_size": hidden_size,
      "num_layers": num_layers,
      "num_workers": 4,
      "data_dir": base_data_dir
   }


def create_job_config(job_id: str, hparams: Dict[str, Any], num_epochs: int, 
                     base_output_dir: str, repo_path: str) -> Dict[str, Any]:
   """Create complete job configuration."""
   
   # Create unique output directory for this job
   timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
   output_dir = f"{base_output_dir}/test_globus_{timestamp}/job_{job_id}"
   
   job_config = {
      "job_id": job_id,
      "output_dir": output_dir,
      "repo_path": repo_path,
      "num_epochs": num_epochs,
      **hparams
   }
   
   return job_config


def print_hyperparameters_table(job_configs: List[Dict[str, Any]]) -> None:
   """Print a formatted table of hyperparameters."""
   logger = logging.getLogger(__name__)
   
   logger.info("Generated hyperparameters for parallel training jobs:")
   logger.info("-" * 100)
   logger.info(f"{'Job ID':<15} {'Model':<12} {'LR':<10} {'Batch':<7} {'Dropout':<8} {'Epochs':<7}")
   logger.info("-" * 100)
   
   for config in job_configs:
      logger.info(f"{config['job_id']:<15} {config['model_type']:<12} "
                 f"{config['learning_rate']:<10.6f} {config['batch_size']:<7} "
                 f"{config['dropout_rate']:<8.3f} {config['num_epochs']:<7}")
   
   logger.info("-" * 100)


def print_results_table(results: List[Dict[str, Any]]) -> None:
   """Print a formatted table of results."""
   logger = logging.getLogger(__name__)
   
   logger.info("\nTraining Results Summary:")
   logger.info("=" * 120)
   logger.info(f"{'Job ID':<15} {'Status':<10} {'Duration':<10} {'AUC':<8} {'Accuracy':<10} {'Host':<20}")
   logger.info("=" * 120)
   
   successful_jobs = 0
   total_auc = 0.0
   total_accuracy = 0.0
   
   for result in results:
      status = result.get('status', 'unknown')
      duration = result.get('execution_time', 0)
      auc = result.get('training_results', {}).get('auc', 0.0)
      accuracy = result.get('training_results', {}).get('accuracy', 0.0)
      hostname = result.get('hostname', 'unknown')
      job_id = result.get('job_id', 'unknown')
      
      logger.info(f"{job_id:<15} {status:<10} {duration:<10.1f}s {auc:<8.4f} "
                 f"{accuracy:<10.4f} {hostname:<20}")
      
      if status == 'completed':
         successful_jobs += 1
         total_auc += auc
         total_accuracy += accuracy
   
   logger.info("=" * 120)
   
   # Summary statistics
   if successful_jobs > 0:
      avg_auc = total_auc / successful_jobs
      avg_accuracy = total_accuracy / successful_jobs
      success_rate = (successful_jobs / len(results)) * 100
      
      logger.info(f"\nSummary Statistics:")
      logger.info(f"  Successful Jobs: {successful_jobs}/{len(results)} ({success_rate:.1f}%)")
      logger.info(f"  Average AUC: {avg_auc:.4f}")
      logger.info(f"  Average Accuracy: {avg_accuracy:.4f}")
      
      # Validation criteria
      min_expected_auc = 0.01  # Very low threshold for short training
      if avg_auc >= min_expected_auc and success_rate >= 80:
         logger.info(f"✅ VALIDATION PASSED: Training pipeline is working correctly!")
      else:
         logger.warning(f"⚠️  VALIDATION CONCERNS: Success rate or AUC below expectations")
   else:
      logger.error(f"❌ VALIDATION FAILED: No jobs completed successfully")


def test_globus_training_pipeline(endpoint_id: str, num_jobs: int, num_epochs: int, 
                                base_output_dir: str, repo_path: str, data_dir: str,
                                max_wait_minutes: int = 30) -> bool:
   """Test the complete Globus Compute training pipeline."""
   
   logger = logging.getLogger(__name__)
   
   if not GLOBUS_COMPUTE_AVAILABLE:
      logger.error("globus-compute-sdk not available. Please install with: pip install globus-compute-sdk")
      return False
   
   logger.info(f"Starting Globus Compute training validation test")
   logger.info(f"Endpoint ID: {endpoint_id}")
   logger.info(f"Number of parallel jobs: {num_jobs}")
   logger.info(f"Epochs per job: {num_epochs}")
   logger.info(f"Maximum wait time: {max_wait_minutes} minutes")
   
   try:
      # Create Globus Compute client
      client = Client()
      
      # Register the training function
      logger.info("Registering training function...")
      function_id = client.register_function(cifar100_training_function)
      logger.info(f"Training function registered with ID: {function_id}")
      
      # Generate random hyperparameters for all jobs
      logger.info(f"Generating {num_jobs} sets of random hyperparameters...")
      job_configs = []
      
      for i in range(num_jobs):
         job_id = f"test_{i+1:03d}"
         hparams = generate_random_hyperparameters(job_id, data_dir)
         job_config = create_job_config(job_id, hparams, num_epochs, base_output_dir, repo_path)
         job_configs.append(job_config)
      
      # Print hyperparameters table
      print_hyperparameters_table(job_configs)
      
      # Submit all jobs
      logger.info(f"Submitting {num_jobs} parallel training jobs...")
      task_ids = []
      
      for config in job_configs:
         task_id = client.run(function_id=function_id, endpoint_id=endpoint_id, function_args=[config])
         task_ids.append(task_id)
         logger.info(f"Submitted job {config['job_id']} with task ID: {task_id}")
      
      # Monitor job progress
      logger.info("Monitoring job progress...")
      completed_results = []
      max_wait_seconds = max_wait_minutes * 60
      wait_interval = 30  # Check every 30 seconds
      elapsed_time = 0
      
      while elapsed_time < max_wait_seconds and len(completed_results) < len(task_ids):
         logger.info(f"Checking job status... ({len(completed_results)}/{len(task_ids)} completed, "
                    f"{elapsed_time//60:.0f}/{max_wait_minutes} minutes elapsed)")
         
         # Check each task
         for i, task_id in enumerate(task_ids):
            # Skip if already completed
            if any(r.get('task_id') == task_id for r in completed_results):
               continue
            
            try:
               # Check if task is done (will raise exception if not ready)
               result = client.get_result(task_id)
               result['task_id'] = task_id
               completed_results.append(result)
               
               job_id = job_configs[i]['job_id']
               status = result.get('status', 'unknown')
               logger.info(f"✅ Job {job_id} completed with status: {status}")
               
            except Exception:
               # Task not ready yet, continue
               pass
         
         # Wait before next check
         if len(completed_results) < len(task_ids):
            time.sleep(wait_interval)
            elapsed_time += wait_interval
      
      # Check for any remaining incomplete jobs
      if len(completed_results) < len(task_ids):
         incomplete_count = len(task_ids) - len(completed_results)
         logger.warning(f"⚠️  {incomplete_count} jobs did not complete within {max_wait_minutes} minutes")
         
         # Try to get status of incomplete jobs
         for i, task_id in enumerate(task_ids):
            if not any(r.get('task_id') == task_id for r in completed_results):
               job_id = job_configs[i]['job_id']
               try:
                  # Try one more time
                  result = client.get_result(task_id)
                  result['task_id'] = task_id
                  completed_results.append(result)
                  logger.info(f"✅ Job {job_id} completed (late)")
               except Exception as e:
                  logger.warning(f"⏱️  Job {job_id} still running or failed: {e}")
      
      # Display results
      if completed_results:
         print_results_table(completed_results)
         
         # Determine overall success
         successful_jobs = sum(1 for r in completed_results if r.get('status') == 'completed')
         success_rate = (successful_jobs / len(task_ids)) * 100
         
         return success_rate >= 80  # Consider test passed if 80% of jobs succeed
      else:
         logger.error("❌ No jobs completed successfully")
         return False
      
   except Exception as e:
      logger.error(f"Error in training pipeline test: {e}")
      return False


def main():
   """Main function."""
   parser = argparse.ArgumentParser(description="Test Globus Compute training pipeline")
   
   parser.add_argument("--endpoint-id", "-e", required=True,
                      help="Globus Compute endpoint ID for Aurora")
   parser.add_argument("--num-jobs", "-n", type=int, default=5,
                      help="Number of parallel training jobs to submit (default: 5)")
   parser.add_argument("--epochs", type=int, default=3,
                      help="Number of training epochs per job (default: 3)")
   parser.add_argument("--output-dir", default="/tmp/globus_training_test",
                      help="Base output directory for results")
   parser.add_argument("--repo-path", 
                      default="/lus/flare/projects/datascience/parton/inference_service_test",
                      help="Path to repository on Aurora")
   parser.add_argument("--data-dir",
                      default="/lus/flare/projects/datascience/parton/data",
                      help="Path to CIFAR-100 data parent directory on Aurora (torchvision expects cifar-100-python subdirectory here)")
   parser.add_argument("--max-wait-minutes", type=int, default=30,
                      help="Maximum time to wait for jobs to complete (default: 30)")
   parser.add_argument("--log-level", default="INFO",
                      choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                      help="Logging level")
   parser.add_argument("--seed", type=int, default=42,
                      help="Random seed for reproducible hyperparameters")
   
   args = parser.parse_args()
   
   # Setup logging
   setup_logging(args.log_level)
   logger = logging.getLogger(__name__)
   
   # Set random seed for reproducible tests
   random.seed(args.seed)
   np.random.seed(args.seed)
   
   # Check dependencies
   if not GLOBUS_COMPUTE_AVAILABLE:
      logger.error("globus-compute-sdk not available. Please install with:")
      logger.error("  pip install globus-compute-sdk")
      return 1
   
   # Validate arguments
   if args.num_jobs < 1:
      logger.error("Number of jobs must be at least 1")
      return 1
   
   if args.epochs < 1:
      logger.error("Number of epochs must be at least 1")
      return 1
   
   # Run the test
   logger.info("=" * 80)
   logger.info("Globus Compute Training Pipeline Validation Test")
   logger.info("=" * 80)
   
   success = test_globus_training_pipeline(
      endpoint_id=args.endpoint_id,
      num_jobs=args.num_jobs,
      num_epochs=args.epochs,
      base_output_dir=args.output_dir,
      repo_path=args.repo_path,
      data_dir=args.data_dir,
      max_wait_minutes=args.max_wait_minutes
   )
   
   if success:
      logger.info("🎉 Training pipeline validation PASSED!")
      logger.info("Your Globus Compute setup is ready for the agentic workflow.")
      return 0
   else:
      logger.error("💥 Training pipeline validation FAILED!")
      logger.error("Please check your Globus Compute setup and try again.")
      return 1


if __name__ == "__main__":
   sys.exit(main()) 