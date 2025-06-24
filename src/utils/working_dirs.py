#!/usr/bin/env python3
"""
Working directory management utilities.
Handles hierarchical directory structure for workflow runs.
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional


class WorkingDirManager:
   """Manages hierarchical working directories for workflow runs."""
   
   def __init__(self, config: Dict[str, Any]):
      self.config = config
      self.logger = logging.getLogger(__name__)
      
      # Working directory configuration
      self.working_dir_config = config.get("working_dirs", {})
      self.local_base_dir = Path(self.working_dir_config.get("local_base", "results"))
      self.launch_iteration_pattern = self.working_dir_config.get("launch_iteration_pattern", 
                                                                  "{timestamp}_{experiment_id}")
      self.job_pattern = self.working_dir_config.get("job_pattern", "job_{job_id}")
      
      # Current launch iteration directory (set when first job is created)
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
   
   def get_local_job_dir(self, job_id: str, experiment_id: str) -> Path:
      """Get the full local job directory path using hierarchical structure."""
      # Generate launch iteration directory if not set
      if self.current_launch_dir is None:
         launch_dir_name = self._generate_launch_iteration_dir(experiment_id)
         self.current_launch_dir = self.local_base_dir / launch_dir_name
         self.current_launch_dir.mkdir(parents=True, exist_ok=True)
         self.logger.info(f"Created local launch iteration directory: {self.current_launch_dir}")
      
      # Generate job directory name
      job_dir_name = self.job_pattern.format(job_id=job_id)
      
      # Full path: {local_base}/{launch_iteration}/{job_dir}
      full_job_dir = self.current_launch_dir / job_dir_name
      full_job_dir.mkdir(parents=True, exist_ok=True)
      
      return full_job_dir
   
   def get_local_launch_dir(self) -> Optional[Path]:
      """Get the current local launch iteration directory."""
      return self.current_launch_dir
   
   def create_experiment_directories(self, experiment_id: str) -> Dict[str, Path]:
      """Create all necessary directories for an experiment."""
      # Generate launch iteration directory
      launch_dir_name = self._generate_launch_iteration_dir(experiment_id)
      self.current_launch_dir = self.local_base_dir / launch_dir_name
      
      # Create main directories
      directories = {
         "launch_dir": self.current_launch_dir,
         "logs_dir": self.current_launch_dir / "logs",
         "checkpoints_dir": self.current_launch_dir / "checkpoints",
         "tensorboard_dir": self.current_launch_dir / "tensorboard",
         "reports_dir": self.current_launch_dir / "reports"
      }
      
      # Create all directories
      for dir_path in directories.values():
         dir_path.mkdir(parents=True, exist_ok=True)
      
      self.logger.info(f"Created experiment directories in: {self.current_launch_dir}")
      return directories
   
   def get_job_specific_dirs(self, job_id: str, experiment_id: str) -> Dict[str, Path]:
      """Get job-specific directories within the experiment structure."""
      job_dir = self.get_local_job_dir(job_id, experiment_id)
      
      directories = {
         "job_dir": job_dir.resolve(),
         "output_dir": (job_dir / "output").resolve(),
         "logs_dir": (job_dir / "logs").resolve(),
         "checkpoints_dir": (job_dir / "checkpoints").resolve(),
         "tensorboard_dir": (job_dir / "tensorboard").resolve()
      }
      
      # Create all directories
      for dir_path in directories.values():
         dir_path.mkdir(parents=True, exist_ok=True)
      
      return directories
   
   def get_aurora_job_specific_dirs(self, job_id: str, experiment_id: str) -> Dict[str, str]:
      """Get job-specific directories for Aurora using the configured aurora_base, launch_iteration_pattern, and job_pattern."""
      working_dir_config = self.working_dir_config
      aurora_base = working_dir_config.get("aurora_base")
      launch_iteration_pattern = working_dir_config.get("launch_iteration_pattern", "{timestamp}_{experiment_id}")
      job_pattern = working_dir_config.get("job_pattern", "job_{job_id}")

      # Use the same timestamp logic as local, but if current_launch_dir is set, extract the launch dir name
      if self.current_launch_dir is not None:
         launch_dir_name = self.current_launch_dir.name
      else:
         # If not set, generate a new one (should match local logic)
         launch_dir_name = self._generate_launch_iteration_dir(experiment_id)

      job_dir_name = job_pattern.format(job_id=job_id)
      aurora_job_dir = f"{aurora_base}/{launch_dir_name}/{job_dir_name}"

      directories = {
         "job_dir": aurora_job_dir,
         "output_dir": f"{aurora_job_dir}/output",
         "logs_dir": f"{aurora_job_dir}/logs",
         "checkpoints_dir": f"{aurora_job_dir}/checkpoints",
         "tensorboard_dir": f"{aurora_job_dir}/tensorboard"
      }
      return directories
   
   def cleanup_old_experiments(self, max_age_hours: int = 24) -> None:
      """Clean up old experiment directories."""
      try:
         current_time = datetime.now().timestamp()
         
         if not self.local_base_dir.exists():
            return
         
         for item in self.local_base_dir.iterdir():
            if item.is_dir():
               # Check if this is an old experiment directory
               mtime = item.stat().st_mtime
               
               if current_time - mtime > max_age_hours * 3600:
                  # Remove old experiment directory
                  import shutil
                  shutil.rmtree(item)
                  self.logger.info(f"Cleaned up old experiment directory: {item}")
         
      except Exception as e:
         self.logger.error(f"Error during experiment cleanup: {e}")
   
   def get_experiment_summary(self) -> Dict[str, Any]:
      """Get a summary of the current experiment directory structure."""
      if self.current_launch_dir is None:
         return {"status": "no_experiment_started"}
      
      summary = {
         "launch_dir": str(self.current_launch_dir),
         "experiment_id": self.current_launch_dir.name.split("_", 1)[1] if "_" in self.current_launch_dir.name else "unknown",
         "timestamp": self.current_launch_dir.name.split("_")[0] if "_" in self.current_launch_dir.name else "unknown",
         "job_dirs": [],
         "total_size": 0
      }
      
      try:
         # Count job directories and calculate total size
         for item in self.current_launch_dir.iterdir():
            if item.is_dir() and item.name.startswith("job_"):
               summary["job_dirs"].append(item.name)
               
               # Calculate directory size
               dir_size = sum(f.stat().st_size for f in item.rglob('*') if f.is_file())
               summary["total_size"] += dir_size
         
         summary["num_jobs"] = len(summary["job_dirs"])
         summary["total_size_mb"] = summary["total_size"] / (1024 * 1024)
         
      except Exception as e:
         self.logger.error(f"Error getting experiment summary: {e}")
         summary["error"] = str(e)
      
      return summary 