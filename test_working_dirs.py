#!/usr/bin/env python3
"""
Test script to demonstrate the new hierarchical working directory structure.
"""

import yaml
import logging
from pathlib import Path
from src.utils.working_dirs import WorkingDirManager


def setup_logging():
   """Setup basic logging."""
   logging.basicConfig(
      level=logging.INFO,
      format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
      datefmt="%d-%m %H:%M"
   )


def test_working_dirs():
   """Test the working directory manager."""
   print("=" * 60)
   print("TESTING HIERARCHICAL WORKING DIRECTORY STRUCTURE")
   print("=" * 60)
   
   # Sample configuration
   config = {
      "working_dirs": {
         "aurora_base": "/lus/eagle/projects/datascience/testuser/workflows",
         "local_base": "test_results",
         "launch_iteration_pattern": "{timestamp}_{experiment_id}",
         "job_pattern": "job_{job_id}"
      }
   }
   
   # Create working directory manager
   wdm = WorkingDirManager(config)
   
   # Test experiment directory creation
   experiment_id = "test_exp_001"
   print(f"\nCreating experiment directories for: {experiment_id}")
   
   experiment_dirs = wdm.create_experiment_directories(experiment_id)
   print(f"Created directories: {experiment_dirs}")
   
   # Test job-specific directories
   job_id = f"{experiment_id}_iter_0"
   print(f"\nCreating job-specific directories for: {job_id}")
   
   job_dirs = wdm.get_job_specific_dirs(job_id, experiment_id)
   print(f"Job directories: {job_dirs}")
   
   # Test another job
   job_id_2 = f"{experiment_id}_iter_1"
   print(f"\nCreating job-specific directories for: {job_id_2}")
   
   job_dirs_2 = wdm.get_job_specific_dirs(job_id_2, experiment_id)
   print(f"Job directories: {job_dirs_2}")
   
   # Get experiment summary
   print(f"\nExperiment summary:")
   summary = wdm.get_experiment_summary()
   for key, value in summary.items():
      print(f"  {key}: {value}")
   
   # Show directory structure
   print(f"\nDirectory structure:")
   launch_dir = wdm.get_local_launch_dir()
   if launch_dir:
      print(f"Launch directory: {launch_dir}")
      for item in launch_dir.iterdir():
         if item.is_dir():
            print(f"  ├── {item.name}/")
            for subitem in item.iterdir():
               if subitem.is_dir():
                  print(f"  │   ├── {subitem.name}/")
                  for file in subitem.iterdir():
                     if file.is_file():
                        print(f"  │   │   └── {file.name}")
   
   print(f"\nTest completed successfully!")
   print("=" * 60)


def test_config_loading():
   """Test loading configuration from YAML file."""
   print("\n" + "=" * 60)
   print("TESTING CONFIGURATION LOADING")
   print("=" * 60)
   
   # Create a test config file
   test_config = {
      "sophia": {
         "url": "https://data-portal-dev.cels.anl.gov/resource_server/sophia/vllm/v1"
      },
      "aurora": {
         "host": "aurora",
         "user": "${AURORA_USER}",
         "control_master": True,
         "pbs_template": "jobs/train_cifar100.pbs.template"
      },
      "working_dirs": {
         "aurora_base": "/lus/eagle/projects/datascience/${AURORA_USER}/workflows",
         "local_base": "test_results_config",
         "launch_iteration_pattern": "{timestamp}_{experiment_id}",
         "job_pattern": "job_{job_id}"
      },
      "data": {
         "dir": "data/cifar-100-python"
      },
      "results": {
         "dir": "results"
      },
      "polling_interval": 60
   }
   
   # Save test config
   with open("test_config.yaml", "w") as f:
      yaml.dump(test_config, f, default_flow_style=False)
   
   print("Created test_config.yaml")
   
   # Load and test
   with open("test_config.yaml", "r") as f:
      loaded_config = yaml.safe_load(f)
   
   wdm = WorkingDirManager(loaded_config)
   experiment_id = "config_test_001"
   
   experiment_dirs = wdm.create_experiment_directories(experiment_id)
   print(f"Created directories from config: {experiment_dirs}")
   
   # Clean up
   Path("test_config.yaml").unlink(missing_ok=True)
   
   print("Configuration test completed successfully!")
   print("=" * 60)


if __name__ == "__main__":
   setup_logging()
   
   try:
      test_working_dirs()
      test_config_loading()
      
      print("\n" + "=" * 60)
      print("ALL TESTS PASSED!")
      print("=" * 60)
      
   except Exception as e:
      print(f"\nTest failed with error: {e}")
      raise 