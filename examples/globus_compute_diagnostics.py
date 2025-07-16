#!/usr/bin/env python3
"""
Globus Compute diagnostic test script for Aurora PyTorch and Intel GPU verification.
This script runs on the host (laptop) and submits test jobs to Aurora via Globus Compute.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Any

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

try:
   from globus_compute_sdk import Executor, Client
   GLOBUS_COMPUTE_AVAILABLE = True
except ImportError:
   GLOBUS_COMPUTE_AVAILABLE = False
   Executor = None
   Client = None

from examples.torch_test import train_wrapper, simple_torch_test


def setup_logging(log_level: str = "INFO"):
   """Setup logging configuration."""
   logging.basicConfig(
      level=getattr(logging, log_level.upper()),
      format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
      datefmt="%d-%m %H:%M"
   )


def test_basic_connectivity(endpoint_id: str) -> bool:
   """Test basic connectivity to the Globus Compute endpoint."""
   logger = logging.getLogger(__name__)
   
   if not GLOBUS_COMPUTE_AVAILABLE:
      logger.error("globus-compute-sdk not available. Please install with: pip install globus-compute-sdk")
      return False
   
   try:
      logger.info("Testing basic connectivity...")
      
      # Simple test function
      def hello_test():
         import socket
         from datetime import datetime
         
         return {
            'message': 'Hello from Aurora!',
            'hostname': socket.gethostname(),
            'timestamp': datetime.now().isoformat()
         }
      
      # Submit test
      client = Client()
      function_id = client.register_function(hello_test)
      task_id = client.run(function_id=function_id, endpoint_id=endpoint_id)
      
      # Wait for result
      result = client.get_result(task_id)
      
      logger.info("Basic connectivity test passed:")
      logger.info(f"  Message: {result['message']}")
      logger.info(f"  Hostname: {result['hostname']}")
      logger.info(f"  Timestamp: {result['timestamp']}")
      
      return True
      
   except Exception as e:
      logger.error(f"Basic connectivity test failed: {e}")
      return False


def test_torch_simple(endpoint_id: str) -> bool:
   """Test simple PyTorch and Intel GPU availability."""
   logger = logging.getLogger(__name__)
   
   try:
      logger.info("Testing PyTorch and Intel GPU availability...")
      
      # Submit test using the simple_torch_test function
      client = Client()
      function_id = client.register_function(simple_torch_test)
      task_id = client.run(function_id=function_id, endpoint_id=endpoint_id)
      
      # Wait for result
      result = client.get_result(task_id)
      
      logger.info("PyTorch test results:")
      logger.info(f"  Hostname: {result['hostname']}")
      logger.info(f"  PyTorch Version: {result['torch_version']}")
      logger.info(f"  Intel Extension Version: {result['ipex_version']}")
      logger.info(f"  XPU Available: {result['xpu_available']}")
      logger.info(f"  Device Count: {result['device_count']}")
      logger.info(f"  Current Device: {result['current_device']}")
      logger.info(f"  Device Name: {result['device_name']}")
      
      if result['xpu_available']:
         logger.info("✓ Intel GPU access confirmed!")
      else:
         logger.warning("⚠ Intel GPU not available - running on CPU")
      
      return True
      
   except Exception as e:
      logger.error(f"PyTorch test failed: {e}")
      return False


def test_torch_parallel(endpoint_id: str, num_tasks: int = 12) -> bool:
   """Test parallel PyTorch jobs with different hyperparameters."""
   logger = logging.getLogger(__name__)
   
   try:
      logger.info(f"Testing parallel PyTorch jobs ({num_tasks} tasks)...")
      
      # Generate hyperparameter sets
      hparam_sets = [
         {
            'lr': 0.01 + 0.001 * i,
            'batch': 32 + i,
            'optimizer': 'adam',
            'task_id': i
         }
         for i in range(num_tasks)
      ]
      
      # Submit parallel jobs using Executor
      with Executor(endpoint_id=endpoint_id) as executor:
         # Submit all tasks
         futures = [executor.submit(train_wrapper, hparams) for hparams in hparam_sets]
         
         # Collect results
         results = []
         for i, future in enumerate(futures):
            try:
               result = future.result()
               results.append(result)
               logger.info(f"Task {i+1}/{num_tasks} completed on {result['hostname']}")
            except Exception as e:
               logger.error(f"Task {i+1}/{num_tasks} failed: {e}")
         
         # Summary statistics
         if results:
            logger.info("Parallel execution summary:")
            hostnames = set(r['hostname'] for r in results)
            logger.info(f"  Executed on {len(hostnames)} hosts: {', '.join(hostnames)}")
            
            # GPU info summary
            gpu_devices = [r['device_name'] for r in results if r['device_name'] != 'No GPU']
            if gpu_devices:
               unique_gpus = set(gpu_devices)
               logger.info(f"  GPU devices used: {', '.join(unique_gpus)}")
            else:
               logger.warning("  No GPU devices were used")
            
            # Timing statistics
            exec_times = [r['elapsed_seconds'] for r in results]
            avg_time = sum(exec_times) / len(exec_times)
            logger.info(f"  Average execution time: {avg_time:.2f} seconds")
            
            return True
         else:
            logger.error("No tasks completed successfully")
            return False
      
   except Exception as e:
      logger.error(f"Parallel PyTorch test failed: {e}")
      return False


def test_mpi_and_tiles(endpoint_id: str) -> bool:
   """Test MPI and GPU tile information."""
   logger = logging.getLogger(__name__)
   
   try:
      logger.info("Testing MPI and GPU tile information...")
      
      # Test with a single hyperparameter set
      test_hparams = {
         'lr': 0.001,
         'batch': 64,
         'optimizer': 'adam',
         'test_type': 'mpi_tile_test'
      }
      
      # Submit test
      client = Client()
      function_id = client.register_function(train_wrapper)
      task_id = client.run(function_id=function_id, endpoint_id=endpoint_id, function_args=[test_hparams])
      
      # Wait for result
      result = client.get_result(task_id)
      
      logger.info("MPI and GPU tile test results:")
      logger.info(f"  Hostname: {result['hostname']}")
      logger.info(f"  Process ID: {result['pid']}")
      logger.info(f"  MPI Rank: {result['mpi_rank']}")
      logger.info(f"  GPU Tile ID: {result['gpu_tile_id']}")
      logger.info(f"  Torch XPU Device: {result['torch_xpu_device_id']}")
      logger.info(f"  Device Name: {result['device_name']}")
      logger.info(f"  Execution Time: {result['elapsed_seconds']:.2f} seconds")
      
      return True
      
   except Exception as e:
      logger.error(f"MPI and tile test failed: {e}")
      return False


def run_comprehensive_diagnostics(endpoint_id: str, num_parallel_tasks: int = 12) -> bool:
   """Run comprehensive diagnostics on the Globus Compute endpoint."""
   logger = logging.getLogger(__name__)
   
   logger.info("Starting comprehensive Globus Compute diagnostics...")
   logger.info(f"Endpoint ID: {endpoint_id}")
   
   # Test 1: Basic connectivity
   logger.info("\n" + "="*50)
   logger.info("Test 1: Basic Connectivity")
   logger.info("="*50)
   if not test_basic_connectivity(endpoint_id):
      logger.error("Basic connectivity test failed. Stopping diagnostics.")
      return False
   
   # Test 2: PyTorch and Intel GPU
   logger.info("\n" + "="*50)
   logger.info("Test 2: PyTorch and Intel GPU")
   logger.info("="*50)
   if not test_torch_simple(endpoint_id):
      logger.error("PyTorch test failed. Continuing with remaining tests.")
   
   # Test 3: MPI and GPU tiles
   logger.info("\n" + "="*50)
   logger.info("Test 3: MPI and GPU Tiles")
   logger.info("="*50)
   if not test_mpi_and_tiles(endpoint_id):
      logger.error("MPI and tiles test failed. Continuing with remaining tests.")
   
   # Test 4: Parallel execution
   logger.info("\n" + "="*50)
   logger.info("Test 4: Parallel Execution")
   logger.info("="*50)
   if not test_torch_parallel(endpoint_id, num_parallel_tasks):
      logger.error("Parallel execution test failed.")
      return False
   
   logger.info("\n" + "="*50)
   logger.info("Diagnostics completed successfully!")
   logger.info("="*50)
   
   return True


def main():
   """Main function to run diagnostics."""
   parser = argparse.ArgumentParser(description="Globus Compute Aurora diagnostics")
   parser.add_argument("--endpoint-id", "-e", required=True,
                      help="Globus Compute endpoint ID for Aurora")
   parser.add_argument("--num-parallel-tasks", "-n", type=int, default=12,
                      help="Number of parallel tasks to run (default: 12)")
   parser.add_argument("--log-level", "-l", default="INFO",
                      choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                      help="Logging level")
   parser.add_argument("--test-type", "-t", default="comprehensive",
                      choices=["connectivity", "torch", "mpi", "parallel", "comprehensive"],
                      help="Type of test to run")
   
   args = parser.parse_args()
   
   # Setup logging
   setup_logging(args.log_level)
   logger = logging.getLogger(__name__)
   
   # Check if globus-compute-sdk is available
   if not GLOBUS_COMPUTE_AVAILABLE:
      logger.error("globus-compute-sdk not available. Please install with:")
      logger.error("  pip install globus-compute-sdk")
      return 1
   
   # Run selected tests
   success = True
   
   if args.test_type == "connectivity":
      success = test_basic_connectivity(args.endpoint_id)
   elif args.test_type == "torch":
      success = test_torch_simple(args.endpoint_id)
   elif args.test_type == "mpi":
      success = test_mpi_and_tiles(args.endpoint_id)
   elif args.test_type == "parallel":
      success = test_torch_parallel(args.endpoint_id, args.num_parallel_tasks)
   elif args.test_type == "comprehensive":
      success = run_comprehensive_diagnostics(args.endpoint_id, args.num_parallel_tasks)
   
   if success:
      logger.info("All tests completed successfully!")
      return 0
   else:
      logger.error("Some tests failed. Check the logs for details.")
      return 1


if __name__ == "__main__":
   sys.exit(main()) 