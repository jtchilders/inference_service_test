#!/usr/bin/env python3
"""
Example script demonstrating how to use Globus Compute for job submission.
This script shows how to submit training jobs using the GlobusJobScheduler.
"""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.agents.globus_job_scheduler import GlobusJobScheduler
from src.utils.working_dirs import WorkingDirManager


def setup_logging():
   """Setup logging configuration."""
   logging.basicConfig(
      level=logging.INFO,
      format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
      datefmt="%d-%m %H:%M"
   )


def create_sample_config() -> Dict[str, Any]:
   """Create a sample configuration for testing."""
   return {
      "globus_compute": {
         "enabled": True,
         "endpoint_id": "your-endpoint-id-here",  # Replace with actual endpoint ID
         "auth_method": "native_client",
         "function_timeout": 3600,
         "max_retries": 2
      },
      "working_dirs": {
         "aurora_base": "/lus/flare/projects/datascience/parton/workflows",
         "local_base": "results",
         "launch_iteration_pattern": "{timestamp}_{experiment_id}",
         "job_pattern": "job_{job_id}"
      },
      "data": {
         "dir": "data/cifar-100-python"
      }
   }


def create_sample_job_config() -> Dict[str, Any]:
   """Create a sample job configuration for testing."""
   return {
      "job_id": "globus_test_001",
      "experiment_id": "globus_experiment_001",
      "model_type": "resnet18",
      "hidden_size": 512,
      "num_layers": 3,
      "dropout_rate": 0.3,
      "learning_rate": 0.001,
      "batch_size": 64,
      "num_epochs": 10,
      "weight_decay": 1e-4,
      "num_workers": 4,
      "data_dir": "data/cifar-100-python"
   }


def test_globus_compute_job_submission():
   """Test basic Globus Compute job submission functionality."""
   logger = logging.getLogger(__name__)
   
   # Create sample configuration
   config = create_sample_config()
   
   # Check if endpoint ID is configured
   if config["globus_compute"]["endpoint_id"] == "your-endpoint-id-here":
      logger.error("Please configure your Globus Compute endpoint ID in the config")
      logger.error("Run 'globus-compute-endpoint show aurora-training' to get your endpoint ID")
      return False
   
   try:
      # Initialize job scheduler
      scheduler = GlobusJobScheduler(
         endpoint_id=config["globus_compute"]["endpoint_id"],
         working_dir_config=config["working_dirs"],
         auth_method=config["globus_compute"]["auth_method"],
         function_timeout=config["globus_compute"]["function_timeout"],
         max_retries=config["globus_compute"]["max_retries"]
      )
      
      # Create job configuration
      job_config = create_sample_job_config()
      
      # Submit job
      logger.info("Submitting test job...")
      job_id = scheduler.submit_job(job_config)
      logger.info(f"Job submitted with ID: {job_id}")
      
      # Monitor job status
      logger.info("Monitoring job status...")
      max_wait_time = 600  # 10 minutes
      wait_interval = 30   # 30 seconds
      waited_time = 0
      
      while waited_time < max_wait_time:
         status = scheduler.get_job_status(job_id)
         logger.info(f"Job status: {status}")
         
         if scheduler.is_job_complete(job_id):
            logger.info("Job completed!")
            break
         
         time.sleep(wait_interval)
         waited_time += wait_interval
      
      if waited_time >= max_wait_time:
         logger.warning("Job monitoring timed out")
         return False
      
      # Get results
      logger.info("Retrieving job results...")
      results = scheduler.get_job_results(job_id)
      
      # Display results
      logger.info("Job Results:")
      logger.info(f"  Status: {results.get('status', 'unknown')}")
      logger.info(f"  Execution Time: {results.get('execution_time', 0):.2f} seconds")
      
      if results.get('status') == 'completed':
         training_results = results.get('training_results', {})
         logger.info(f"  Final Accuracy: {training_results.get('accuracy', 'N/A')}")
         logger.info(f"  Final Loss: {training_results.get('loss', 'N/A')}")
      elif results.get('status') == 'failed':
         logger.error(f"  Error: {results.get('error', 'Unknown error')}")
         logger.error(f"  Stderr: {results.get('stderr', 'No stderr')}")
      
      return True
      
   except Exception as e:
      logger.error(f"Error in job submission test: {e}")
      return False


def test_multiple_jobs():
   """Test submitting multiple jobs concurrently."""
   logger = logging.getLogger(__name__)
   
   # Create sample configuration
   config = create_sample_config()
   
   # Check if endpoint ID is configured
   if config["globus_compute"]["endpoint_id"] == "your-endpoint-id-here":
      logger.error("Please configure your Globus Compute endpoint ID in the config")
      return False
   
   try:
      # Initialize job scheduler
      scheduler = GlobusJobScheduler(
         endpoint_id=config["globus_compute"]["endpoint_id"],
         working_dir_config=config["working_dirs"],
         auth_method=config["globus_compute"]["auth_method"],
         function_timeout=config["globus_compute"]["function_timeout"],
         max_retries=config["globus_compute"]["max_retries"]
      )
      
      # Submit multiple jobs with different configurations
      job_configs = []
      learning_rates = [0.01, 0.001, 0.0001]
      
      for i, lr in enumerate(learning_rates):
         job_config = create_sample_job_config()
         job_config["job_id"] = f"globus_multi_test_{i:03d}"
         job_config["learning_rate"] = lr
         job_config["num_epochs"] = 5  # Shorter for testing
         job_configs.append(job_config)
      
      # Submit all jobs
      submitted_jobs = []
      for job_config in job_configs:
         job_id = scheduler.submit_job(job_config)
         submitted_jobs.append(job_id)
         logger.info(f"Submitted job {job_id} with learning rate {job_config['learning_rate']}")
      
      # Wait for all jobs to complete
      logger.info("Waiting for all jobs to complete...")
      completed_jobs = []
      max_wait_time = 1200  # 20 minutes
      wait_interval = 60    # 1 minute
      waited_time = 0
      
      while waited_time < max_wait_time and len(completed_jobs) < len(submitted_jobs):
         for job_id in submitted_jobs:
            if job_id not in completed_jobs and scheduler.is_job_complete(job_id):
               completed_jobs.append(job_id)
               logger.info(f"Job {job_id} completed")
         
         time.sleep(wait_interval)
         waited_time += wait_interval
      
      # Get results for all completed jobs
      logger.info("Retrieving results for all completed jobs...")
      all_results = []
      
      for job_id in completed_jobs:
         try:
            results = scheduler.get_job_results(job_id)
            all_results.append(results)
         except Exception as e:
            logger.error(f"Error getting results for job {job_id}: {e}")
      
      # Display summary
      logger.info("Job Summary:")
      for results in all_results:
         job_id = results.get('job_id', 'unknown')
         status = results.get('status', 'unknown')
         lr = results.get('config', {}).get('learning_rate', 'unknown')
         logger.info(f"  Job {job_id}: Status={status}, LR={lr}")
         
         if status == 'completed':
            training_results = results.get('training_results', {})
            accuracy = training_results.get('accuracy', 'N/A')
            logger.info(f"    Final Accuracy: {accuracy}")
      
      return True
      
   except Exception as e:
      logger.error(f"Error in multiple jobs test: {e}")
      return False


def main():
   """Main function to run the example."""
   setup_logging()
   logger = logging.getLogger(__name__)
   
   logger.info("Starting Globus Compute example...")
   
   # Test basic job submission
   logger.info("Testing basic job submission...")
   success = test_globus_compute_job_submission()
   
   if success:
      logger.info("Basic job submission test passed!")
      
      # Test multiple jobs
      logger.info("Testing multiple jobs submission...")
      success = test_multiple_jobs()
      
      if success:
         logger.info("Multiple jobs test passed!")
      else:
         logger.error("Multiple jobs test failed!")
   else:
      logger.error("Basic job submission test failed!")
   
   logger.info("Example completed.")


if __name__ == "__main__":
   main() 