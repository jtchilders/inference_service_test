#!/usr/bin/env python3
"""
Example script following the exact usage pattern provided by the user.
This demonstrates the direct Globus Compute SDK usage for testing Aurora.
"""

import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

try:
   from globus_compute_sdk import Executor
   GLOBUS_COMPUTE_AVAILABLE = True
except ImportError:
   GLOBUS_COMPUTE_AVAILABLE = False
   print("Error: globus-compute-sdk not available. Please install with: pip install globus-compute-sdk")
   sys.exit(1)

from torch_test import train_wrapper

# Update this with your actual endpoint ID
ENDPOINT_ID = 'your-endpoint-id-here'  # Replace with your actual endpoint ID

# Generate multiple sets of dummy hyperparameters for testing
hparam_sets = [
   {
      'lr': 0.01 + 0.001 * i,
      'batch': 32 + i,
      'optimizer': 'adam',
      'task_id': i
   }
   for i in range(36)
]


def main():
   """Main function following the user's exact pattern."""
   
   # Check if endpoint ID is configured
   if ENDPOINT_ID == 'your-endpoint-id-here':
      print("Error: Please configure your Globus Compute endpoint ID")
      print("Update the ENDPOINT_ID variable in this script with your actual endpoint ID")
      print("You can get your endpoint ID by running: globus-compute-endpoint show aurora-training")
      return 1
   
   print(f"Testing with endpoint ID: {ENDPOINT_ID}")
   print(f"Submitting {len(hparam_sets)} tasks...")
   
   try:
      with Executor(endpoint_id=ENDPOINT_ID) as aurora_ep:
         # Submit parallel tasks
         futures = [aurora_ep.submit(train_wrapper, hp) for hp in hparam_sets]
         
         # Collect results as they complete
         for i, future in enumerate(futures):
            try:
               result = future.result()
               print(f"\nDiagnostics from remote execution (task {i+1}):")
               for key, val in result.items():
                  print(f"{key}: {val}")
            except Exception as e:
               print(f"\nTask {i+1} failed with error: {e}")
         
         print("\nAll tasks completed!")
         
   except Exception as e:
      print(f"Error during execution: {e}")
      return 1
   
   return 0


if __name__ == '__main__':
   sys.exit(main()) 