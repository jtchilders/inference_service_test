#!/usr/bin/env python3
"""
Test script to verify Aurora configuration and PyTorch setup.
This script tests the Intel GPU configuration, IPEX integration, and environment setup.
"""

import os
import sys
import logging
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from training.train import setup_aurora_environment, detect_device


def setup_logging():
   """Setup logging configuration."""
   log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
   date_format = "%d-%m %H:%M"
   
   logging.basicConfig(
      level=logging.INFO,
      format=log_format,
      datefmt=date_format,
      handlers=[
         logging.StreamHandler(sys.stdout)
      ]
   )


def test_environment_variables():
   """Test Aurora environment variable setup."""
   print("=" * 60)
   print("TESTING AURORA ENVIRONMENT VARIABLES")
   print("=" * 60)
   
   # Setup Aurora environment
   setup_aurora_environment()
   
   # Check environment variables
   env_vars = [
      'OMP_NUM_THREADS',
      'OMP_PLACES', 
      'OMP_PROC_BIND',
      'I_MPI_PIN_DOMAIN',
      'I_MPI_PIN_ORDER',
      'ZE_AFFINITY_MASK',
      'SYCL_DEVICE_FILTER',
      'ONEAPI_DEVICE_SELECTOR'
   ]
   
   for var in env_vars:
      value = os.environ.get(var, "NOT SET")
      print(f"{var}: {value}")
   
   print("Environment variables test completed!")


def test_ipex_import():
   """Test Intel PyTorch Extension (IPEX) import and functionality."""
   print("\n" + "=" * 60)
   print("TESTING INTEL PYTORCH EXTENSION (IPEX)")
   print("=" * 60)
   
   try:
      import intel_extension_for_pytorch as ipex
      print("✓ Intel PyTorch Extension (IPEX) imported successfully")
      print(f"IPEX version: {ipex.__version__}")
      
      # Test IPEX optimization with a simple model
      import torch
      import torch.nn as nn
      
      # Create a simple model
      model = nn.Sequential(
         nn.Linear(10, 5),
         nn.ReLU(),
         nn.Linear(5, 1)
      )
      
      optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
      
      # Test IPEX optimization
      try:
         optimized_model, optimized_optimizer = ipex.optimize(model, optimizer=optimizer)
         print("✓ IPEX optimization successful")
         
         # Test forward pass with optimized model
         x = torch.randn(1, 10)
         with torch.no_grad():
            output = optimized_model(x)
         print(f"✓ Optimized model forward pass successful: {output.shape}")
         
      except Exception as e:
         print(f"✗ IPEX optimization failed: {e}")
      
   except ImportError as e:
      print(f"✗ Intel PyTorch Extension (IPEX) import failed: {e}")
      print("Note: IPEX may not be available in the current environment")


def test_pytorch_setup():
   """Test PyTorch setup and device detection."""
   print("\n" + "=" * 60)
   print("TESTING PYTORCH SETUP")
   print("=" * 60)
   
   try:
      import torch
      print(f"PyTorch version: {torch.__version__}")
      print(f"Intel PyTorch Extension available: {hasattr(torch, 'xpu')}")
      
      # Test device detection
      device_type = detect_device()
      print(f"Detected device type: {device_type}")
      
      # Test device availability
      if hasattr(torch, 'xpu') and torch.xpu.is_available():
         device_count = torch.xpu.device_count()
         print(f"Intel GPU count: {device_count}")
         for i in range(device_count):
            print(f"Intel GPU {i}: {torch.xpu.get_device_name(i)}")
      
      if torch.cuda.is_available():
         device_count = torch.cuda.device_count()
         print(f"CUDA GPU count: {device_count}")
         for i in range(device_count):
            print(f"CUDA GPU {i}: {torch.cuda.get_device_name(i)}")
      
      # Test tensor operations
      print("\nTesting tensor operations...")
      if device_type == "xpu":
         device = torch.device("xpu:0")
         x = torch.randn(3, 3, device=device)
         y = torch.randn(3, 3, device=device)
         z = torch.mm(x, y)
         print(f"Intel GPU tensor operation successful: {z.shape}")
      elif device_type == "cuda":
         device = torch.device("cuda:0")
         x = torch.randn(3, 3, device=device)
         y = torch.randn(3, 3, device=device)
         z = torch.mm(x, y)
         print(f"CUDA tensor operation successful: {z.shape}")
      else:
         device = torch.device("cpu")
         x = torch.randn(3, 3, device=device)
         y = torch.randn(3, 3, device=device)
         z = torch.mm(x, y)
         print(f"CPU tensor operation successful: {z.shape}")
      
      print("PyTorch setup test completed successfully!")
      
   except ImportError as e:
      print(f"PyTorch import failed: {e}")
   except Exception as e:
      print(f"PyTorch test failed: {e}")


def test_module_loading():
   """Test module loading for Aurora."""
   print("\n" + "=" * 60)
   print("TESTING MODULE LOADING")
   print("=" * 60)
   
   # Check if we're in a PBS job environment
   pbs_job_id = os.environ.get('PBS_JOBID', 'NOT SET')
   print(f"PBS Job ID: {pbs_job_id}")
   
   # Check working directory
   working_dir = os.getcwd()
   print(f"Working directory: {working_dir}")
   
   # Check if we're on Aurora
   hostname = os.uname().nodename if hasattr(os, 'uname') else 'unknown'
   print(f"Hostname: {hostname}")
   
   # Check Python path
   python_path = sys.executable
   print(f"Python executable: {python_path}")
   
   # Check if Intel modules are loaded
   intel_vars = [
      'INTEL_LICENSE_FILE',
      'INTEL_DEVICE_REDIRECT',
      'INTEL_OPENCL_CONFIG'
   ]
   
   print("\nIntel environment variables:")
   for var in intel_vars:
      value = os.environ.get(var, "NOT SET")
      print(f"{var}: {value}")
   
   # Check for Intel-specific modules
   try:
      import subprocess
      result = subprocess.run(['module', 'list'], capture_output=True, text=True, timeout=10)
      if result.returncode == 0:
         print("\nLoaded modules:")
         for line in result.stdout.split('\n'):
            if any(keyword in line.lower() for keyword in ['intel', 'pytorch', 'framework']):
               print(f"  {line.strip()}")
   except Exception as e:
      print(f"Could not check loaded modules: {e}")
   
   print("Module loading test completed!")


def test_training_components():
   """Test training components with IPEX integration."""
   print("\n" + "=" * 60)
   print("TESTING TRAINING COMPONENTS")
   print("=" * 60)
   
   try:
      # Test importing training components
      from training.trainer import Trainer
      from training.model import get_model
      from utils.data_utils import get_cifar100_dataloaders
      
      print("✓ Training components imported successfully")
      
      # Test model creation
      model = get_model(
         model_type="resnet18",
         num_classes=100,
         hidden_size=1024,
         num_layers=3,
         dropout_rate=0.2
      )
      print("✓ Model creation successful")
      
      # Test IPEX optimization if available
      try:
         import intel_extension_for_pytorch as ipex
         import torch.optim as optim
         
         optimizer = optim.Adam(model.parameters(), lr=0.001)
         optimized_model, optimized_optimizer = ipex.optimize(model, optimizer=optimizer)
         print("✓ IPEX model optimization successful")
         
      except ImportError:
         print("ℹ IPEX not available, skipping optimization test")
      except Exception as e:
         print(f"✗ IPEX optimization test failed: {e}")
      
      print("Training components test completed successfully!")
      
   except Exception as e:
      print(f"Training components test failed: {e}")


def main():
   """Main test function."""
   setup_logging()
   
   print("AURORA CONFIGURATION TEST")
   print("=" * 60)
   print("This script tests the Aurora configuration for PyTorch jobs with IPEX.")
   print("=" * 60)
   
   # Run tests
   test_environment_variables()
   test_ipex_import()
   test_pytorch_setup()
   test_module_loading()
   test_training_components()
   
   print("\n" + "=" * 60)
   print("ALL TESTS COMPLETED")
   print("=" * 60)
   print("If all tests passed, your Aurora configuration is ready!")
   print("You can now submit PyTorch jobs using the updated PBS template with IPEX.")
   print("\nKey features verified:")
   print("- Intel environment variables")
   print("- Intel PyTorch Extension (IPEX)")
   print("- PyTorch Intel GPU support")
   print("- Module loading")
   print("- Training component integration")


if __name__ == "__main__":
   main() 