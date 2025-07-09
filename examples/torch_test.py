#!/usr/bin/env python3
"""
Torch test wrapper function for testing PyTorch and Intel GPU access on Aurora.
This function runs on the Aurora endpoint to verify hardware and software availability.
"""


def train_wrapper(hparams: dict):
   """
   Test function that runs on Aurora via Globus Compute to verify PyTorch and Intel GPU access.
   
   Args:
      hparams: Dictionary containing hyperparameters for testing
      
   Returns:
      Dict containing diagnostic information about the execution environment
   """
   import torch
   import intel_extension_for_pytorch as ipex
   import socket
   import os
   import logging
   from datetime import datetime

   hostname = socket.gethostname()
   pid = os.getpid()

   # Try to get MPI rank and tile info (if available)
   mpi_rank = os.environ.get('PMI_RANK', 'unknown')
   gpu_tile_id = os.environ.get('ZE_AFFINITY_MASK', 'unknown')

   if torch.xpu.is_available():
      device_id = torch.xpu.current_device()
      device_name = torch.xpu.get_device_name(device_id)
   else:
      device_id = 'cpu'
      device_name = 'No GPU'

   start_time = datetime.now()
   logging.info(f"Job started at {start_time.isoformat()}")

   # Dummy tensor computation
   a = torch.randn(1000, 1000, device='xpu' if torch.xpu.is_available() else 'cpu')
   b = torch.randn(1000, 1000, device='xpu' if torch.xpu.is_available() else 'cpu')
   c = torch.matmul(a, b)

   end_time = datetime.now()
   logging.info(f"Job finished at {end_time.isoformat()}")

   diagnostics = {
      'hostname': hostname,
      'pid': pid,
      'mpi_rank': mpi_rank,
      'gpu_tile_id': gpu_tile_id,
      'torch_xpu_device_id': device_id,
      'device_name': device_name,
      'start_time': start_time.isoformat(),
      'end_time': end_time.isoformat(),
      'elapsed_seconds': (end_time - start_time).total_seconds(),
      'hparams_received': hparams,
      'tensor_result_shape': c.shape
   }

   # Clearly print out the diagnostic info
   print("=== Tile/GPU Diagnostic Info ===")
   print(f"Hostname: {hostname}")
   print(f"Process ID: {pid}")
   print(f"MPI Rank: {mpi_rank}")
   print(f"GPU Tile ID (ZE_AFFINITY_MASK): {gpu_tile_id}")
   print(f"Torch XPU Device ID: {device_id}")
   print(f"Device Name: {device_name}")
   print(f"Start Time: {start_time.isoformat()}")
   print(f"End Time: {end_time.isoformat()}")
   print("===============================")

   return diagnostics


def simple_torch_test():
   """
   Simple torch test function for basic verification.
   
   Returns:
      Dict containing basic torch and GPU information
   """
   import torch
   import intel_extension_for_pytorch as ipex
   import socket
   from datetime import datetime

   hostname = socket.gethostname()
   
   # Check PyTorch and Intel GPU availability
   torch_version = torch.__version__
   ipex_version = ipex.__version__
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
      'ipex_version': ipex_version,
      'xpu_available': xpu_available,
      'device_count': device_count,
      'current_device': current_device,
      'device_name': device_name,
      'timestamp': datetime.now().isoformat()
   }


if __name__ == "__main__":
   """Test the functions locally."""
   print("Testing train_wrapper function...")
   test_hparams = {'lr': 0.01, 'batch': 32, 'optimizer': 'adam'}
   result = train_wrapper(test_hparams)
   
   print("\nTrain wrapper results:")
   for key, value in result.items():
      print(f"  {key}: {value}")
   
   print("\nTesting simple_torch_test function...")
   simple_result = simple_torch_test()
   
   print("\nSimple torch test results:")
   for key, value in simple_result.items():
      print(f"  {key}: {value}") 