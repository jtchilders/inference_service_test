#!/usr/bin/env python3
"""
Globus Compute Job Scheduler for Aurora training jobs.
Alternative to direct PBS submission using Globus Compute service.
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

try:
   from globus_compute_sdk import Client, Executor
   GLOBUS_COMPUTE_AVAILABLE = True
except ImportError:
   GLOBUS_COMPUTE_AVAILABLE = False
   Client = None
   Executor = None


class GlobusJobScheduler:
   """Manages job submission and monitoring using Globus Compute on Aurora."""
   
   def __init__(self, endpoint_id: str, working_dir_config: Optional[Dict[str, Any]] = None,
                auth_method: str = "native_client", function_timeout: int = 3600,
                max_retries: int = 2):
      """
      Initialize the Globus Compute job scheduler.
      
      Args:
         endpoint_id: Globus Compute endpoint ID for Aurora
         working_dir_config: Working directory configuration
         auth_method: Authentication method for Globus Compute
         function_timeout: Timeout for function execution in seconds
         max_retries: Maximum number of retries for failed jobs
      """
      if not GLOBUS_COMPUTE_AVAILABLE:
         raise ImportError("globus-compute-sdk is required for GlobusJobScheduler. "
                          "Install with: pip install globus-compute-sdk")
      
      self.endpoint_id = endpoint_id
      self.auth_method = auth_method
      self.function_timeout = function_timeout
      self.max_retries = max_retries
      self.logger = logging.getLogger(__name__)
      
      # Working directory configuration
      self.working_dir_config = working_dir_config or {}
      self.aurora_base_dir = self.working_dir_config.get("aurora_base")
      self.launch_iteration_pattern = self.working_dir_config.get("launch_iteration_pattern", 
                                                                  "{timestamp}_{experiment_id}")
      self.job_pattern = self.working_dir_config.get("job_pattern", "job_{job_id}")
      
      # Track submitted jobs
      self.submitted_jobs = {}
      
      # Current launch iteration directory (set when first job is submitted)
      self.current_launch_dir = None
      
      # Initialize Globus Compute client
      self.client = Client()
      
      # Register training function
      self.training_function_id = None
      self._register_training_function()
      
      self.logger.info(f"Initialized GlobusJobScheduler with endpoint {endpoint_id}")
   
   def _register_training_function(self):
      """Register the training function with Globus Compute."""
      
      def training_function(job_config: Dict[str, Any]) -> Dict[str, Any]:
         """
         Training function that runs on Aurora via Globus Compute.
         
         Args:
            job_config: Configuration for the training job
            
         Returns:
            Dict containing training results
         """
         import json
         import os
         import sys
         import subprocess
         from pathlib import Path
         from datetime import datetime
         
         # Setup environment
         job_id = job_config["job_id"]
         output_dir = job_config["output_dir"]
         
         # Create output directory
         os.makedirs(output_dir, exist_ok=True)
         
         # Log start time
         start_time = datetime.now()
         
         try:
            # Change to repo directory
            repo_path = job_config.get("repo_path", "/lus/flare/projects/datascience/parton/workflows/inference_service_test")
            os.chdir(repo_path)
            
            # Add repo to Python path
            sys.path.insert(0, repo_path)
            
            # Prepare training command
            train_cmd = [
               "python", "-m", "src.training.train",
               "--job-id", job_id,
               "--output-dir", output_dir,
               "--data-dir", job_config["data_dir"],
               "--model-type", job_config.get("model_type", "resnet18"),
               "--hidden-size", str(job_config.get("hidden_size", 1024)),
               "--num-layers", str(job_config.get("num_layers", 3)),
               "--dropout-rate", str(job_config.get("dropout_rate", 0.2)),
               "--learning-rate", str(job_config.get("learning_rate", 0.001)),
               "--batch-size", str(job_config.get("batch_size", 128)),
               "--num-epochs", str(job_config.get("num_epochs", 50)),
               "--weight-decay", str(job_config.get("weight_decay", 1e-4)),
               "--num-workers", str(job_config.get("num_workers", 4))
            ]
            
            # Execute training
            result = subprocess.run(
               train_cmd,
               capture_output=True,
               text=True,
               timeout=3600  # 1 hour timeout
            )
            
            # Process results
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            # Read results file if it exists
            results_file = f"{output_dir}/results.json"
            training_results = {}
            
            if os.path.exists(results_file):
               with open(results_file, 'r') as f:
                  training_results = json.load(f)
            
            # Compile complete results
            complete_results = {
               "job_id": job_id,
               "status": "completed" if result.returncode == 0 else "failed",
               "execution_time": execution_time,
               "start_time": start_time.isoformat(),
               "end_time": end_time.isoformat(),
               "return_code": result.returncode,
               "stdout": result.stdout,
               "stderr": result.stderr,
               "training_results": training_results,
               "config": job_config
            }
            
            # Save results to file
            with open(f"{output_dir}/globus_compute_results.json", 'w') as f:
               json.dump(complete_results, f, indent=2)
            
            return complete_results
            
         except Exception as e:
            # Handle errors
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            error_results = {
               "job_id": job_id,
               "status": "error",
               "execution_time": execution_time,
               "start_time": start_time.isoformat(),
               "end_time": end_time.isoformat(),
               "error": str(e),
               "config": job_config
            }
            
            # Save error results
            try:
               with open(f"{output_dir}/globus_compute_results.json", 'w') as f:
                  json.dump(error_results, f, indent=2)
            except:
               pass
            
            return error_results
      
      # Register the function
      self.training_function_id = self.client.register_function(training_function)
      self.logger.info(f"Registered training function with ID: {self.training_function_id}")
   
   def _generate_launch_iteration_dir(self, experiment_id: str) -> str:
      """Generate the launch iteration directory name using the configured pattern."""
      timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
      
      # Replace placeholders in the pattern
      dir_name = self.launch_iteration_pattern.format(
         timestamp=timestamp,
         experiment_id=experiment_id
      )
      
      return dir_name
   
   def _get_aurora_job_dir(self, job_id: str, experiment_id: str) -> str:
      """Get the full Aurora job directory path using hierarchical structure."""
      # Generate launch iteration directory if not set
      if self.current_launch_dir is None:
         launch_dir_name = self._generate_launch_iteration_dir(experiment_id)
         self.current_launch_dir = f"{self.aurora_base_dir}/{launch_dir_name}"
         self.logger.info(f"Created launch iteration directory: {self.current_launch_dir}")
      
      # Generate job directory name
      job_dir_name = self.job_pattern.format(job_id=job_id)
      
      # Full path: {aurora_base}/{launch_iteration}/{job_dir}
      full_job_dir = f"{self.current_launch_dir}/{job_dir_name}"
      
      return full_job_dir
   
   def submit_job(self, job_config: Dict[str, Any]) -> str:
      """Submit a training job via Globus Compute."""
      
      try:
         # Get job information
         job_id = job_config["job_id"]
         experiment_id = job_config.get("experiment_id", job_id.split("_")[0])
         
         # Get Aurora job directory
         aurora_job_dir = self._get_aurora_job_dir(job_id, experiment_id)
         
         # Update job config with Aurora-specific paths
         job_config["output_dir"] = aurora_job_dir
         job_config["repo_path"] = f"{self.aurora_base_dir}/inference_service_test"
         
         # Submit function to Globus Compute
         task_id = self.client.run(
            function_id=self.training_function_id,
            endpoint_id=self.endpoint_id,
            function_args=[job_config],
            function_kwargs={}
         )
         
         # Store job information
         self.submitted_jobs[job_id] = {
            "task_id": task_id,
            "aurora_job_dir": aurora_job_dir,
            "config": job_config,
            "submitted_at": time.time(),
            "status": "submitted"
         }
         
         self.logger.info(f"Submitted job {job_id} with task ID {task_id} to {aurora_job_dir}")
         return job_id
         
      except Exception as e:
         self.logger.error(f"Error submitting job {job_config.get('job_id', 'unknown')}: {e}")
         raise
   
   def is_job_complete(self, job_id: str) -> bool:
      """Check if a job has completed."""
      
      if job_id not in self.submitted_jobs:
         return False
      
      try:
         task_id = self.submitted_jobs[job_id]["task_id"]
         
         # Check task status using the correct Globus Compute SDK method
         task_info = self.client.get_task(task_id)
         
         # Extract status from task info dict
         task_status = task_info.get('status', 'unknown')
         
         if task_status in ["success", "failed", "cancelled", "completed"]:
            self.submitted_jobs[job_id]["status"] = task_status
            return True
         
         return False
         
      except Exception as e:
         self.logger.error(f"Error checking job status for {job_id}: {e}")
         return False
   
   def get_job_results(self, job_id: str) -> Dict[str, Any]:
      """Get results from a completed job."""
      
      if job_id not in self.submitted_jobs:
         raise ValueError(f"Job {job_id} not found in submitted jobs")
      
      try:
         task_id = self.submitted_jobs[job_id]["task_id"]
         
         # Get task result
         result = self.client.get_result(task_id)
         
         # The result is already in our expected format from the training function
         return result
         
      except Exception as e:
         self.logger.error(f"Error getting results for job {job_id}: {e}")
         raise
   
   def cancel_job(self, job_id: str) -> bool:
      """Cancel a running job."""
      
      if job_id not in self.submitted_jobs:
         return False
      
      try:
         task_id = self.submitted_jobs[job_id]["task_id"]
         
         # Cancel task
         self.client.cancel_task(task_id)
         
         self.submitted_jobs[job_id]["status"] = "cancelled"
         self.logger.info(f"Cancelled job {job_id}")
         return True
         
      except Exception as e:
         self.logger.error(f"Error cancelling job {job_id}: {e}")
         return False
   
   def get_job_status(self, job_id: str) -> str:
      """Get current status of a job."""
      
      if job_id not in self.submitted_jobs:
         return "unknown"
      
      if self.is_job_complete(job_id):
         return self.submitted_jobs[job_id]["status"]
      
      return "running"
   
   def cleanup_jobs(self, max_age_hours: int = 24) -> None:
      """Clean up old job directories on Aurora."""
      
      # Note: For Globus Compute, we rely on the endpoint's cleanup mechanisms
      # and Aurora's standard job cleanup processes
      self.logger.info("Cleanup for Globus Compute jobs is handled by the endpoint")
   
   def run_diagnostic_test(self, test_type: str = "simple") -> Dict[str, Any]:
      """
      Run diagnostic tests to verify endpoint functionality.
      
      Args:
         test_type: Type of diagnostic test to run
                   - "simple": Basic connectivity and PyTorch test
                   - "torch": PyTorch and Intel GPU test
                   - "parallel": Multiple parallel jobs test
      
      Returns:
         Dict containing test results
      """
      
      def simple_diagnostic():
         """Simple diagnostic function that runs on the endpoint."""
         import torch
         import socket
         import os
         from datetime import datetime
         
         try:
            import intel_extension_for_pytorch as ipex
            ipex_available = True
            ipex_version = ipex.__version__
         except ImportError:
            ipex_available = False
            ipex_version = "Not available"
         
         hostname = socket.gethostname()
         torch_version = torch.__version__
         xpu_available = torch.xpu.is_available()
         
         if xpu_available:
            device_count = torch.xpu.device_count()
            current_device = torch.xpu.current_device()
            device_name = torch.xpu.get_device_name(current_device)
         else:
            device_count = 0
            current_device = 'cpu'
            device_name = 'No GPU'
         
         return {
            'hostname': hostname,
            'torch_version': torch_version,
            'ipex_available': ipex_available,
            'ipex_version': ipex_version,
            'xpu_available': xpu_available,
            'device_count': device_count,
            'current_device': current_device,
            'device_name': device_name,
            'timestamp': datetime.now().isoformat(),
            'test_type': 'simple_diagnostic'
         }
      
      def torch_diagnostic():
         """Torch diagnostic function with tensor computation."""
         import torch
         import socket
         import os
         from datetime import datetime
         
         try:
            import intel_extension_for_pytorch as ipex
            ipex_available = True
            ipex_version = ipex.__version__
         except ImportError:
            ipex_available = False
            ipex_version = "Not available"
         
         hostname = socket.gethostname()
         start_time = datetime.now()
         
         # Tensor computation test
         device = 'xpu' if torch.xpu.is_available() else 'cpu'
         a = torch.randn(500, 500, device=device)
         b = torch.randn(500, 500, device=device)
         c = torch.matmul(a, b)
         
         end_time = datetime.now()
         execution_time = (end_time - start_time).total_seconds()
         
         return {
            'hostname': hostname,
            'torch_version': torch.__version__,
            'ipex_available': ipex_available,
            'ipex_version': ipex_version,
            'xpu_available': torch.xpu.is_available(),
            'device_used': device,
            'tensor_shape': c.shape,
            'execution_time': execution_time,
            'timestamp': datetime.now().isoformat(),
            'test_type': 'torch_diagnostic'
         }
      
      try:
         if test_type == "simple":
            # Register and run simple diagnostic
            func_id = self.client.register_function(simple_diagnostic)
            task_id = self.client.run(function_id=func_id, endpoint_id=self.endpoint_id)
            result = self.client.get_result(task_id)
            
            self.logger.info("Simple diagnostic test completed:")
            self.logger.info(f"  Hostname: {result['hostname']}")
            self.logger.info(f"  PyTorch Version: {result['torch_version']}")
            self.logger.info(f"  Intel Extension Available: {result['ipex_available']}")
            self.logger.info(f"  XPU Available: {result['xpu_available']}")
            self.logger.info(f"  Device Name: {result['device_name']}")
            
            return result
         
         elif test_type == "torch":
            # Register and run torch diagnostic
            func_id = self.client.register_function(torch_diagnostic)
            task_id = self.client.run(function_id=func_id, endpoint_id=self.endpoint_id)
            result = self.client.get_result(task_id)
            
            self.logger.info("Torch diagnostic test completed:")
            self.logger.info(f"  Hostname: {result['hostname']}")
            self.logger.info(f"  Device Used: {result['device_used']}")
            self.logger.info(f"  Execution Time: {result['execution_time']:.3f}s")
            self.logger.info(f"  Tensor Shape: {result['tensor_shape']}")
            
            return result
         
         elif test_type == "parallel":
            # Run multiple parallel torch diagnostics
            func_id = self.client.register_function(torch_diagnostic)
            
            # Submit multiple tasks
            task_ids = []
            for i in range(4):  # 4 parallel tasks
               task_id = self.client.run(function_id=func_id, endpoint_id=self.endpoint_id)
               task_ids.append(task_id)
            
            # Collect results
            results = []
            for task_id in task_ids:
               result = self.client.get_result(task_id)
               results.append(result)
            
            # Summary
            hostnames = set(r['hostname'] for r in results)
            avg_time = sum(r['execution_time'] for r in results) / len(results)
            
            self.logger.info("Parallel diagnostic test completed:")
            self.logger.info(f"  Tasks executed: {len(results)}")
            self.logger.info(f"  Hosts used: {len(hostnames)}")
            self.logger.info(f"  Average execution time: {avg_time:.3f}s")
            
            return {
               'test_type': 'parallel_diagnostic',
               'num_tasks': len(results),
               'hostnames': list(hostnames),
               'average_execution_time': avg_time,
               'individual_results': results
            }
         
         else:
            raise ValueError(f"Unknown test type: {test_type}")
      
      except Exception as e:
         self.logger.error(f"Diagnostic test failed: {e}")
         return {
            'test_type': test_type,
            'status': 'failed',
            'error': str(e)
         }
   
   def get_current_launch_dir(self) -> Optional[str]:
      """Get the current launch iteration directory."""
      return self.current_launch_dir
   
   def __enter__(self):
      """Context manager entry."""
      return self
   
   def __exit__(self, exc_type, exc_val, exc_tb):
      """Context manager exit."""
      pass 