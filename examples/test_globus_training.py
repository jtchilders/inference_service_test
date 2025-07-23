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
import traceback
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
   """
   Generate random hyperparameters for training.
   
   Supported model types (from src/training/model.py get_model function):
   - "custom": CustomCNN with configurable hidden_size and num_layers
   - "resnet18", "resnet34", "resnet50": ResNet variants adapted for CIFAR-100
   - "vgg16": VGG16 adapted for CIFAR-100  
   - "densenet121": DenseNet121 adapted for CIFAR-100
   
   Note: Trainer class currently only supports Adam optimizer regardless of this parameter.
   """
   
   # Learning rate: log scale between 1e-4 and 1e-2
   learning_rate = 10 ** random.uniform(-4, -2)
   
   # Batch size: choose from common values
   batch_size = random.choice([32, 64, 128])  # Keeping smaller sizes for faster testing
   
   # Model type: choose from available models (must match get_model() function)
   model_type = random.choice(["resnet18", "custom", "resnet34", "resnet50", "vgg16", "densenet121"])
   
   # Note: optimizer is hardcoded as Adam in Trainer class, so this parameter is not used
   # Keeping for potential future support
   optimizer = "adam"  # Only Adam is currently supported
   
   # Dropout rate: uniform between 0.1 and 0.5 (within validation range [0.0, 0.8])
   dropout_rate = random.uniform(0.1, 0.5)
   
   # Weight decay: log scale between 1e-5 and 1e-3
   weight_decay = 10 ** random.uniform(-5, -3)
   
   # Hidden size for custom models (only used by "custom" model type)
   hidden_size = random.choice([256, 512, 1024, 2048])  # Matches validation choices
   
   # Number of layers for custom models (only used by "custom" model type)  
   num_layers = random.choice([2, 3, 4])  # Within validation range [1, 10]
   
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
   logger.info("=" * 140)
   logger.info(f"{'Job ID':<15} {'Status':<12} {'Duration':<10} {'AUC':<8} {'Accuracy':<10} {'Host':<20} {'Error Category':<20}")
   logger.info("=" * 140)
   
   successful_jobs = 0
   failed_jobs = 0
   total_auc = 0.0
   total_accuracy = 0.0
   error_summary = {}
   
   for result in results:
      status = result.get('status', 'unknown')
      duration = result.get('execution_time', 0)
      auc = result.get('training_results', {}).get('auc', 0.0)
      accuracy = result.get('training_results', {}).get('accuracy', 0.0)
      hostname = result.get('hostname', 'unknown')
      job_id = result.get('job_id', 'unknown')
      
      # Extract error information
      error_info = result.get('error_info', {})
      if error_info:
         error_category = error_info.get('error_category', 'unknown')
         error_severity = error_info.get('error_severity', 'unknown')
         error_stage = error_info.get('execution_stage', 'unknown')
         
         # Track error statistics
         if error_category not in error_summary:
            error_summary[error_category] = []
         error_summary[error_category].append({
            'job_id': job_id,
            'severity': error_severity,
            'stage': error_stage,
            'description': error_info.get('error_description', 'No description')
         })
         
         error_display = f"{error_category} ({error_severity})"
      else:
         error_display = "None"
      
      logger.info(f"{job_id:<15} {status:<12} {duration:<10.1f}s {auc:<8.4f} "
                 f"{accuracy:<10.4f} {hostname:<20} {error_display:<20}")
      
      if status == 'completed':
         successful_jobs += 1
         total_auc += auc
         total_accuracy += accuracy
      else:
         failed_jobs += 1
   
   logger.info("=" * 140)
   
   # Summary statistics
   total_jobs = len(results)
   if successful_jobs > 0:
      avg_auc = total_auc / successful_jobs
      avg_accuracy = total_accuracy / successful_jobs
      success_rate = (successful_jobs / total_jobs) * 100
      
      logger.info(f"\nSummary Statistics:")
      logger.info(f"  Successful Jobs: {successful_jobs}/{total_jobs} ({success_rate:.1f}%)")
      logger.info(f"  Failed Jobs: {failed_jobs}/{total_jobs} ({(failed_jobs/total_jobs)*100:.1f}%)")
      logger.info(f"  Average AUC (successful): {avg_auc:.4f}")
      logger.info(f"  Average Accuracy (successful): {avg_accuracy:.4f}")
      
      # Validation criteria
      min_expected_auc = 0.01  # Very low threshold for short training
      if avg_auc >= min_expected_auc and success_rate >= 80:
         logger.info(f"✅ VALIDATION PASSED: Training pipeline is working correctly!")
      else:
         logger.warning(f"⚠️  VALIDATION CONCERNS: Success rate or AUC below expectations")
   else:
      logger.error(f"❌ VALIDATION FAILED: No jobs completed successfully")
   
   # Detailed error analysis
   if error_summary:
      logger.info(f"\n📊 Error Analysis:")
      logger.info("=" * 80)
      
      for category, errors in error_summary.items():
         count = len(errors)
         severity_counts = {}
         stage_counts = {}
         
         for error in errors:
            severity = error['severity']
            stage = error['stage']
            
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
            stage_counts[stage] = stage_counts.get(stage, 0) + 1
         
         logger.info(f"\n🔍 {category.upper().replace('_', ' ')} ({count} jobs):")
         
         # Show severity breakdown
         severity_str = ", ".join([f"{sev}: {cnt}" for sev, cnt in severity_counts.items()])
         logger.info(f"   Severity breakdown: {severity_str}")
         
         # Show stage breakdown
         stage_str = ", ".join([f"{stage}: {cnt}" for stage, cnt in stage_counts.items()])
         logger.info(f"   Failure stages: {stage_str}")
         
         # Show specific examples (up to 3)
         logger.info(f"   Examples:")
         for i, error in enumerate(errors[:3]):
            logger.info(f"     • Job {error['job_id']}: {error['description']}")
         if len(errors) > 3:
            logger.info(f"     ... and {len(errors) - 3} more")
      
      # Provide diagnostic recommendations
      logger.info(f"\n💡 Diagnostic Recommendations:")
      logger.info("-" * 50)
      
      if 'environment_error' in error_summary:
         logger.info("🔧 Environment Errors:")
         logger.info("   • Check that all required packages are installed on Aurora")
         logger.info("   • Verify the virtual environment or module loading")
         logger.info("   • Ensure correct Python path and import statements")
      
      if 'data_access_error' in error_summary:
         logger.info("📁 Data Access Errors:")
         logger.info("   • Verify CIFAR-100 data is available at the specified path")
         logger.info("   • Check file permissions and storage access on Aurora")
         logger.info(f"   • Consider setting CIFAR100_DATA_DIR environment variable")
      
      if 'resource_error' in error_summary:
         logger.info("💾 Resource Errors:")
         logger.info("   • Reduce batch size or model complexity")
         logger.info("   • Check memory usage and available GPU/CPU resources")
         logger.info("   • Consider requesting more resources in PBS job")
      
      if 'configuration_error' in error_summary:
         logger.info("⚙️  Configuration Errors:")
         logger.info("   • Validate hyperparameter ranges and data types")
         logger.info("   • Check model compatibility with specified parameters")
         logger.info("   • Review job configuration structure")
      
      if 'training_error' in error_summary:
         logger.info("🎯 Training Errors:")
         logger.info("   • These may be hyperparameter-related (not system issues)")
         logger.info("   • Check for numerical instability (learning rate too high)")
         logger.info("   • Verify model architecture compatibility")
      
      if 'network_error' in error_summary:
         logger.info("🌐 Network Errors:")
         logger.info("   • Check Globus Compute endpoint connectivity")
         logger.info("   • Verify Aurora network access and firewall settings")
         logger.info("   • Consider retry mechanisms for transient issues")


def print_detailed_error_report(results: List[Dict[str, Any]], max_errors_per_category: int = 5) -> None:
   """Print detailed error information for debugging."""
   logger = logging.getLogger(__name__)
   
   failed_results = [r for r in results if r.get('status') == 'failed']
   if not failed_results:
      return
   
   logger.info(f"\n🔍 Detailed Error Report ({len(failed_results)} failed jobs):")
   logger.info("=" * 120)
   
   # Group by error category
   error_categories = {}
   for result in failed_results:
      error_info = result.get('error_info', {})
      category = error_info.get('error_category', 'unknown')
      
      if category not in error_categories:
         error_categories[category] = []
      error_categories[category].append(result)
   
   for category, category_results in error_categories.items():
      logger.info(f"\n📋 {category.upper().replace('_', ' ')} ERRORS ({len(category_results)} jobs):")
      logger.info("-" * 80)
      
      # Show detailed information for first few errors in category
      for i, result in enumerate(category_results[:max_errors_per_category]):
         error_info = result.get('error_info', {})
         job_id = result.get('job_id', 'unknown')
         hostname = result.get('hostname', 'unknown')
         stage = error_info.get('execution_stage', 'unknown')
         severity = error_info.get('error_severity', 'unknown')
         description = error_info.get('error_description', 'No description')
         error_msg = error_info.get('error_message', 'No message')
         
         logger.info(f"\n  Job {job_id} (Host: {hostname}):")
         logger.info(f"    Stage: {stage}")
         logger.info(f"    Severity: {severity}")
         logger.info(f"    Description: {description}")
         logger.info(f"    Error Message: {error_msg}")
         
         # Show traceback for critical errors or if requested
         if severity == 'critical' or logger.level <= logging.DEBUG:
            traceback_lines = error_info.get('full_traceback', '').strip().split('\n')
            if traceback_lines and len(traceback_lines) > 1:
               logger.info(f"    Traceback (last 5 lines):")
               for line in traceback_lines[-5:]:
                  logger.info(f"      {line}")
      
      if len(category_results) > max_errors_per_category:
         remaining = len(category_results) - max_errors_per_category
         logger.info(f"\n  ... and {remaining} more {category} errors (use --log-level DEBUG for full details)")


def validate_hyperparameters(hparams: Dict[str, Any]) -> bool:
   """
   Validate that generated hyperparameters are compatible with the training pipeline.
   
   Returns:
      True if all parameters are valid, False otherwise
   """
   logger = logging.getLogger(__name__)
   
   # Check model type is supported
   supported_models = ["custom", "resnet18", "resnet34", "resnet50", "vgg16", "densenet121"]
   if hparams.get("model_type") not in supported_models:
      logger.error(f"Unsupported model type: {hparams.get('model_type')}. Supported: {supported_models}")
      return False
   
   # Check reasonable parameter ranges
   lr = hparams.get("learning_rate", 0)
   if not (1e-6 <= lr <= 1):
      logger.error(f"Learning rate {lr} outside reasonable range [1e-6, 1]")
      return False
   
   batch_size = hparams.get("batch_size", 0)
   if batch_size not in [16, 32, 64, 128, 256]:
      logger.error(f"Batch size {batch_size} not in common values [16, 32, 64, 128, 256]")
      return False
   
   dropout = hparams.get("dropout_rate", 0)
   if not (0.0 <= dropout <= 0.8):
      logger.error(f"Dropout rate {dropout} outside reasonable range [0.0, 0.8]")
      return False
   
   # Validate custom model parameters if using custom model
   if hparams.get("model_type") == "custom":
      hidden_size = hparams.get("hidden_size", 0)
      if hidden_size not in [256, 512, 1024, 2048, 4096]:
         logger.error(f"Hidden size {hidden_size} not in common values for custom model")
         return False
      
      num_layers = hparams.get("num_layers", 0)
      if not (1 <= num_layers <= 10):
         logger.error(f"Number of layers {num_layers} outside reasonable range [1, 10]")
         return False
   
   logger.debug("Hyperparameters validation passed")
   return True


def test_globus_training_pipeline(endpoint_id: str, num_jobs: int, num_epochs: int, 
                                base_output_dir: str, repo_path: str, data_dir: str,
                                max_wait_minutes: int = 30, show_detailed_errors: bool = False) -> bool:
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
         
         # Validate hyperparameters before proceeding
         if not validate_hyperparameters(hparams):
            logger.error(f"Invalid hyperparameters generated for job {job_id}: {hparams}")
            return False
         
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
               
               if status == 'completed':
                  auc = result.get('training_results', {}).get('auc', 0.0)
                  logger.info(f"✅ Job {job_id} completed successfully (AUC: {auc:.4f})")
               else:
                  error_info = result.get('error_info', {})
                  error_category = error_info.get('error_category', 'unknown')
                  error_severity = error_info.get('error_severity', 'unknown')
                  logger.warning(f"❌ Job {job_id} failed: {error_category} ({error_severity})")
               
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
                  status = result.get('status', 'unknown')
                  logger.info(f"✅ Job {job_id} completed (late): {status}")
               except Exception as e:
                  logger.warning(f"⏱️  Job {job_id} still running or failed: {e}")
      
      # Display results
      if completed_results:
         print_results_table(completed_results)
         
         # Show detailed error report if there are failures and requested
         failed_count = sum(1 for r in completed_results if r.get('status') == 'failed')
         if failed_count > 0 and show_detailed_errors:
            print_detailed_error_report(completed_results)
         
         # Determine overall success
         successful_jobs = sum(1 for r in completed_results if r.get('status') == 'completed')
         success_rate = (successful_jobs / len(task_ids)) * 100
         
         return success_rate >= 80  # Consider test passed if 80% of jobs succeed
      else:
         logger.error("❌ No jobs completed successfully")
         return False
      
   except Exception as e:
      logger.error(f"Error in training pipeline test: {e}")
      logger.error(f"Full traceback: {traceback.format_exc()}")
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
   parser.add_argument("--show-detailed-errors", action="store_true",
                      help="Show detailed error information for failed jobs")
   
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
      max_wait_minutes=args.max_wait_minutes,
      show_detailed_errors=args.show_detailed_errors
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