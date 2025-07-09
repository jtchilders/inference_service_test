#!/usr/bin/env python3
"""
Job Scheduler for Aurora PBS job management.
Handles SSH connections and PBS job submission/monitoring.
Uses SSH ControlMaster for Aurora authentication (MFA via mobile app).
"""

import json
import logging
import os
import re
import time
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime


class JobScheduler:
   """Manages PBS job submission and monitoring on Aurora using SSH ControlMaster."""
   
   def __init__(self, aurora_host: str, aurora_user: str, pbs_template_path: str, 
                working_dir_config: Optional[Dict[str, Any]] = None, queue: str = "workq"):
      self.aurora_host = aurora_host
      self.aurora_user = aurora_user
      self.pbs_template_path = pbs_template_path
      self.queue = queue
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
   
   def _execute_ssh_command(self, command: str) -> str:
      """Execute a command on Aurora using SSH ControlMaster."""
      try:
         # Use SSH with ControlMaster to reuse existing authenticated connection
         ssh_cmd = [
            "ssh", 
            f"{self.aurora_host}",
            command
         ]
         
         result = subprocess.run(
            ssh_cmd,
            capture_output=True,
            text=True,
            timeout=30
         )
         
         if result.returncode != 0:
            raise Exception(f"SSH command failed: {result.stderr}")
         
         return result.stdout.strip()
         
      except subprocess.TimeoutExpired:
         raise Exception(f"SSH command timed out: {command}")
      except Exception as e:
         self.logger.error(f"SSH command failed: {e}")
         raise
   
   def _upload_file_content(self, content: str, remote_path: str) -> None:
      """Upload file content to Aurora using SSH ControlMaster."""
      try:
         # Create a temporary local file
         import tempfile
         with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
            temp_file.write(content)
            temp_file_path = temp_file.name
         
         # Upload using scp with ControlMaster
         scp_cmd = [
            "scp",
            temp_file_path,
            f"{self.aurora_host}:{remote_path}"
         ]
         
         result = subprocess.run(scp_cmd, capture_output=True, text=True, timeout=30)
         
         # Clean up temporary file
         os.unlink(temp_file_path)
         
         if result.returncode != 0:
            raise Exception(f"SCP upload failed: {result.stderr}")
         
      except Exception as e:
         self.logger.error(f"File upload failed: {e}")
         raise
   
   def _download_file_content(self, remote_path: str) -> str:
      """Download file content from Aurora using SSH ControlMaster."""
      try:
         # Create a temporary local file
         import tempfile
         with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file_path = temp_file.name
         
         # Download using scp with ControlMaster
         scp_cmd = [
            "scp",
            f"{self.aurora_host}:{remote_path}",
            temp_file_path
         ]
         
         result = subprocess.run(scp_cmd, capture_output=True, text=True, timeout=30)
         
         if result.returncode != 0:
            raise Exception(f"SCP download failed: {result.stderr}")
         
         # Read content and clean up
         with open(temp_file_path, 'r') as f:
            content = f.read()
         
         os.unlink(temp_file_path)
         return content
         
      except Exception as e:
         self.logger.error(f"File download failed: {e}")
         raise
   
   def submit_job(self, job_config: Dict[str, Any]) -> str:
      """Submit a PBS job to Aurora."""
      
      try:
         # Generate PBS script from template
         pbs_script = self._generate_pbs_script(job_config)
         
         # Create job directory on Aurora using hierarchical structure
         job_id = job_config["job_id"]
         experiment_id = job_config.get("experiment_id", job_id.split("_")[0])
         
         aurora_job_dir = self._get_aurora_job_dir(job_id, experiment_id)
         
         # Create the full directory structure
         self._execute_ssh_command(f"mkdir -p {aurora_job_dir}")
         
         # Upload PBS script to Aurora
         script_path = f"{aurora_job_dir}/job.pbs"
         self._upload_file_content(pbs_script, script_path)
         
         # Submit job using qsub
         qsub_output = self._execute_ssh_command(f"cd {aurora_job_dir} && qsub job.pbs")
         
         # Get job submission ID
         pbs_job_id = self._extract_pbs_job_id(qsub_output)
         
         if not pbs_job_id:
            raise Exception(f"Failed to get PBS job ID from output: {qsub_output}")
         
         # Store job information
         self.submitted_jobs[job_id] = {
            "pbs_job_id": pbs_job_id,
            "aurora_job_dir": aurora_job_dir,
            "config": job_config,
            "submitted_at": time.time(),
            "status": "submitted"
         }
         
         self.logger.info(f"Submitted job {job_id} with PBS ID {pbs_job_id} to {aurora_job_dir}")
         return job_id
         
      except Exception as e:
         self.logger.error(f"Error submitting job {job_config.get('job_id', 'unknown')}: {e}")
         raise
   
   def is_job_complete(self, job_id: str) -> bool:
      """Check if a job has completed."""
      
      if job_id not in self.submitted_jobs:
         return False
      
      try:
         pbs_job_id = self.submitted_jobs[job_id]["pbs_job_id"]
         
         # Check job status using qstat
         qstat_output = self._execute_ssh_command(f"qstat {pbs_job_id}")
         
         # If qstat returns nothing, job is complete
         if not qstat_output or "Unknown Job Id" in qstat_output:
            self.submitted_jobs[job_id]["status"] = "completed"
            return True
         
         # Check if job is in error state
         if "E" in qstat_output:
            self.submitted_jobs[job_id]["status"] = "error"
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
         aurora_job_dir = self.submitted_jobs[job_id]["aurora_job_dir"]
         
         # Check if results file exists
         results_file = f"{aurora_job_dir}/results.json"
         
         exists_check = self._execute_ssh_command(f"test -f {results_file} && echo 'exists'")
         if not exists_check:
            raise Exception(f"Results file not found: {results_file}")
         
         # Download results file
         results_content = self._download_file_content(results_file)
         results = json.loads(results_content)
         
         # Also get job output and error files
         stdout_file = f"{aurora_job_dir}/job_{job_id}.out"
         stderr_file = f"{aurora_job_dir}/job_{job_id}.err"
         
         try:
            stdout_content = self._download_file_content(stdout_file)
            stderr_content = self._download_file_content(stderr_file)
            
            results["job_stdout"] = stdout_content
            results["job_stderr"] = stderr_content
            
         except Exception as e:
            self.logger.warning(f"Could not download job output files: {e}")
         
         return results
         
      except Exception as e:
         self.logger.error(f"Error getting results for job {job_id}: {e}")
         raise
   
   def cancel_job(self, job_id: str) -> bool:
      """Cancel a running job."""
      
      if job_id not in self.submitted_jobs:
         return False
      
      try:
         pbs_job_id = self.submitted_jobs[job_id]["pbs_job_id"]
         
         # Cancel job using qdel
         self._execute_ssh_command(f"qdel {pbs_job_id}")
         
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
   
   def _generate_pbs_script(self, job_config: Dict[str, Any]) -> str:
      """Generate PBS script from template with job configuration."""
      
      # Read template
      with open(self.pbs_template_path, 'r') as f:
         template = f.read()
      
      # Replace placeholders with actual values
      script = template
      
      # Get repository path on Aurora
      repo_path = f"{self.aurora_base_dir}/inference_service_test"
      
      # Get full data directory path on Aurora
      aurora_data_dir = f"{repo_path}/{job_config['data_dir']}"
      
      # Required parameters
      replacements = {
         "JOB_ID": job_config["job_id"],
         "OUTPUT_DIR": job_config["output_dir"],
         "DATA_DIR": aurora_data_dir,
         "REPO_PATH": repo_path,
         "QUEUE": self.queue,
         "MODEL_TYPE": job_config.get("model_type", "resnet18"),
         "HIDDEN_SIZE": str(job_config.get("hidden_size", 1024)),
         "NUM_LAYERS": str(job_config.get("num_layers", 3)),
         "DROPOUT_RATE": str(job_config.get("dropout_rate", 0.2)),
         "LEARNING_RATE": str(job_config.get("learning_rate", 0.001)),
         "BATCH_SIZE": str(job_config.get("batch_size", 128)),
         "NUM_EPOCHS": str(job_config.get("num_epochs", 50)),
         "WEIGHT_DECAY": str(job_config.get("weight_decay", 1e-4)),
         "NUM_WORKERS": str(job_config.get("num_workers", 4))
      }
      
      for placeholder, value in replacements.items():
         script = script.replace(f"${{{placeholder}}}", value)
      
      return script
   
   def _extract_pbs_job_id(self, qsub_output: str) -> Optional[str]:
      """Extract PBS job ID from qsub output."""
      
      # Common patterns for PBS job IDs
      patterns = [
         r'(\d+\.aurora)',  # Aurora format
         r'(\d+\.\w+)',     # Generic PBS format
         r'(\d+)'           # Just numbers
      ]
      
      for pattern in patterns:
         match = re.search(pattern, qsub_output)
         if match:
            jobid = match.group(1)
            # remove the .aurora from the jobid
            jobid = jobid.replace('.aurora', '')
            return jobid
      
      return None
   
   def cleanup_jobs(self, max_age_hours: int = 24) -> None:
      """Clean up old job directories on Aurora."""
      
      try:
         current_time = time.time()
         workflows_dir = self.aurora_base_dir
         
         # List all launch iteration directories
         output = self._execute_ssh_command(f"ls -la {workflows_dir}")
         
         for line in output.split('\n'):
            if line.startswith('d'):
               parts = line.split()
               if len(parts) >= 9:
                  dir_name = parts[-1]
                  if dir_name not in ['.', '..']:
                     # Check if this is an old launch iteration directory
                     launch_path = f"{workflows_dir}/{dir_name}"
                     
                     # Get directory modification time
                     mtime_str = self._execute_ssh_command(f"stat -c %Y {launch_path}")
                     mtime = float(mtime_str)
                     
                     if current_time - mtime > max_age_hours * 3600:
                        # Remove old launch iteration directory and all its jobs
                        self._execute_ssh_command(f"rm -rf {launch_path}")
                        self.logger.info(f"Cleaned up old launch iteration directory: {launch_path}")
         
      except Exception as e:
         self.logger.error(f"Error during job cleanup: {e}")
   
   def get_current_launch_dir(self) -> Optional[str]:
      """Get the current launch iteration directory."""
      return self.current_launch_dir
   
   def __enter__(self):
      """Context manager entry."""
      return self
   
   def __exit__(self, exc_type, exc_val, exc_tb):
      """Context manager exit."""
      pass 