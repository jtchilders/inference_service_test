#!/usr/bin/env python3
"""
Example script demonstrating enhanced error reporting capabilities.
This shows how to interpret different types of errors from Aurora training jobs.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any

def setup_logging():
   """Setup logging configuration."""
   log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
   date_format = "%d-%m %H:%M"
   
   logging.basicConfig(
      level=logging.INFO,
      format=log_format,
      datefmt=date_format
   )


def create_sample_results() -> List[Dict[str, Any]]:
   """Create sample results showing different types of failures."""
   
   return [
      # Successful job
      {
         "job_id": "test_001",
         "status": "completed",
         "hostname": "aurora-gpu-001", 
         "execution_time": 45.2,
         "training_results": {"auc": 0.85, "accuracy": 0.82},
         "error_info": None
      },
      
      # Data access error
      {
         "job_id": "test_002", 
         "status": "failed",
         "hostname": "aurora-gpu-002",
         "execution_time": 5.1,
         "training_results": {"auc": 0.0, "accuracy": 0.0},
         "error_info": {
            "error_category": "data_access_error",
            "error_severity": "critical",
            "error_description": "Cannot access training data: [Errno 2] No such file or directory: '/wrong/path/cifar-100-python'",
            "execution_stage": "data_loading",
            "error_message": "[Errno 2] No such file or directory: '/wrong/path/cifar-100-python'",
            "error_type": "FileNotFoundError"
         }
      },
      
      # Environment error
      {
         "job_id": "test_003",
         "status": "failed", 
         "hostname": "aurora-gpu-003",
         "execution_time": 2.3,
         "training_results": {"auc": 0.0, "accuracy": 0.0},
         "error_info": {
            "error_category": "environment_error",
            "error_severity": "critical",
            "error_description": "Missing dependency or module: No module named 'torch'",
            "execution_stage": "imports",
            "error_message": "No module named 'torch'",
            "error_type": "ModuleNotFoundError"
         }
      },
      
      # Resource error
      {
         "job_id": "test_004",
         "status": "failed",
         "hostname": "aurora-gpu-004", 
         "execution_time": 12.7,
         "training_results": {"auc": 0.0, "accuracy": 0.0},
         "error_info": {
            "error_category": "resource_error",
            "error_severity": "high",
            "error_description": "Insufficient compute resources: CUDA out of memory. Tried to allocate 2.00 GiB",
            "execution_stage": "training",
            "error_message": "CUDA out of memory. Tried to allocate 2.00 GiB",
            "error_type": "RuntimeError"
         }
      },
      
      # Training error (hyperparameter issue)
      {
         "job_id": "test_005",
         "status": "failed",
         "hostname": "aurora-gpu-005",
         "execution_time": 8.9,
         "training_results": {"auc": 0.0, "accuracy": 0.0},
         "error_info": {
            "error_category": "training_error",
            "error_severity": "medium",
            "error_description": "Training process error: Loss became NaN during training",
            "execution_stage": "training",
            "error_message": "Loss became NaN during training",
            "error_type": "ValueError"
         }
      },
      
      # Configuration error
      {
         "job_id": "test_006",
         "status": "failed",
         "hostname": "aurora-gpu-006",
         "execution_time": 1.5,
         "training_results": {"auc": 0.0, "accuracy": 0.0},
         "error_info": {
            "error_category": "configuration_error", 
            "error_severity": "medium",
            "error_description": "Invalid configuration or parameters: Unsupported model type: 'nonexistent_model'",
            "execution_stage": "model_creation",
            "error_message": "Unsupported model type: 'nonexistent_model'",
            "error_type": "ValueError"
         }
      }
   ]


def analyze_sample_results():
   """Demonstrate error analysis on sample results."""
   logger = logging.getLogger(__name__)
   
   logger.info("🔍 Error Analysis Example")
   logger.info("=" * 60)
   logger.info("This demonstrates how to interpret different failure types in Aurora training jobs.\n")
   
   # Get sample results
   results = create_sample_results()
   
   # Import the analysis functions from the test script
   import sys
   sys.path.append(str(Path(__file__).parent))
   from test_globus_training import print_results_table, print_detailed_error_report
   
   # Show the results table
   print_results_table(results)
   
   # Show detailed error report
   print_detailed_error_report(results)
   
   # Interpretation guide
   logger.info("\n📖 How to Interpret Results:")
   logger.info("=" * 50)
   logger.info("• CRITICAL errors (environment, data_access) require infrastructure fixes")
   logger.info("• HIGH errors (resource, network) may need job configuration changes")
   logger.info("• MEDIUM errors (training, configuration) are often hyperparameter issues")
   logger.info("")
   logger.info("💡 Quick Fixes by Error Category:")
   logger.info("• environment_error → Check module installation, virtual environment")
   logger.info("• data_access_error → Verify data paths, set CIFAR100_DATA_DIR")
   logger.info("• resource_error → Reduce batch size, request more memory/GPU")
   logger.info("• training_error → Adjust learning rate, check model compatibility")
   logger.info("• configuration_error → Validate hyperparameters, model types")
   logger.info("• network_error → Check connectivity, retry failed jobs")


def main():
   """Main function."""
   setup_logging()
   logger = logging.getLogger(__name__)
   
   logger.info("Enhanced Error Reporting Example for Aurora Training Jobs")
   logger.info("This shows how the new error categorization helps debug issues.")
   logger.info("")
   
   analyze_sample_results()
   
   logger.info("\n🚀 Next Steps:")
   logger.info("• Run your actual test with: python test_globus_training.py --show-detailed-errors")
   logger.info("• Use the error categories to quickly identify and fix issues")
   logger.info("• Focus on critical/high severity errors first")


if __name__ == "__main__":
   main() 