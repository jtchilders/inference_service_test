#!/usr/bin/env python3
"""
Test script for deployment agent functionality.
This script tests the deployment agent without actually deploying to Aurora.
"""

import os
import sys
import logging
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from agents.deployment_agent import DeploymentAgent


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


def test_deployment_agent_creation():
   """Test that the deployment agent can be created with valid config."""
   print("\n" + "=" * 60)
   print("TESTING DEPLOYMENT AGENT CREATION")
   print("=" * 60)
   
   try:
      # Load test configuration
      import yaml
      config = {
         "aurora": {
            "host": "aurora.alcf.anl.gov",
            "user": "test_user"
         },
         "working_dirs": {
            "aurora_base": "/lus/eagle/projects/datascience/test_user/workflows"
         },
         "data": {
            "dir": "data/cifar-100-python"
         },
         "repository": {
            "url": "https://github.com/test/inference_service_test.git",
            "branch": "main"
         }
      }
      
      # Create deployment agent
      agent = DeploymentAgent(config)
      print("✓ Deployment agent created successfully")
      
      # Test configuration access
      assert agent.aurora_host == "aurora.alcf.anl.gov"
      assert agent.aurora_user == "test_user"
      assert agent.aurora_base_dir == "/lus/eagle/projects/datascience/test_user/workflows"
      assert agent.data_dir == "data/cifar-100-python"
      assert agent.repo_url == "https://github.com/test/inference_service_test.git"
      assert agent.repo_branch == "main"
      
      print("✓ Configuration access working correctly")
      
   except Exception as e:
      print(f"✗ Deployment agent creation failed: {e}")


def test_git_info_detection():
   """Test git repository information detection."""
   print("\n" + "=" * 60)
   print("TESTING GIT INFO DETECTION")
   print("=" * 60)
   
   try:
      # Load test configuration
      config = {
         "aurora": {
            "host": "aurora.alcf.anl.gov",
            "user": "test_user"
         },
         "working_dirs": {
            "aurora_base": "/lus/eagle/projects/datascience/test_user/workflows"
         },
         "data": {
            "dir": "data/cifar-100-python"
         }
      }
      
      # Create deployment agent
      agent = DeploymentAgent(config)
      
      # Test git info detection (if in git repo)
      try:
         repo_url, branch = agent._get_current_git_info()
         print(f"✓ Git repository detected: {repo_url}")
         print(f"✓ Current branch: {branch}")
      except Exception as e:
         print(f"ℹ Git info detection failed (expected if not in git repo): {e}")
      
   except Exception as e:
      print(f"✗ Git info detection test failed: {e}")


def test_configuration_loading():
   """Test configuration loading from file."""
   print("\n" + "=" * 60)
   print("TESTING CONFIGURATION LOADING")
   print("=" * 60)
   
   try:
      # Check if config file exists
      config_file = "config.yaml"
      if os.path.exists(config_file):
         import yaml
         with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
         
         print(f"✓ Configuration loaded from {config_file}")
         
         # Test required fields
         required_fields = [
            "aurora.host",
            "aurora.user", 
            "working_dirs.aurora_base",
            "data.dir"
         ]
         
         for field in required_fields:
            keys = field.split('.')
            value = config
            for key in keys:
               value = value.get(key)
               if value is None:
                  raise ValueError(f"Missing required field: {field}")
         
         print("✓ All required configuration fields present")
         
         # Test repository configuration
         if "repository" in config:
            print("✓ Repository configuration found")
         else:
            print("ℹ Repository configuration not found (will auto-detect)")
         
      else:
         print(f"ℹ Configuration file {config_file} not found")
         print("  Please copy config.yaml.example to config.yaml and update settings")
      
   except Exception as e:
      print(f"✗ Configuration loading test failed: {e}")


def test_deployment_script():
   """Test that the deployment script exists and is executable."""
   print("\n" + "=" * 60)
   print("TESTING DEPLOYMENT SCRIPT")
   print("=" * 60)
   
   try:
      script_path = "scripts/deploy_to_aurora.sh"
      
      if os.path.exists(script_path):
         print(f"✓ Deployment script found: {script_path}")
         
         # Check if executable
         if os.access(script_path, os.X_OK):
            print("✓ Deployment script is executable")
         else:
            print("ℹ Deployment script is not executable")
            print("  Run: chmod +x scripts/deploy_to_aurora.sh")
         
         # Check script content
         with open(script_path, 'r') as f:
            content = f.read()
         
         if "Aurora Deployment Script" in content:
            print("✓ Deployment script has correct header")
         
         if "git clone" in content:
            print("✓ Deployment script includes git operations")
         
         if "cifar-100-python" in content:
            print("✓ Deployment script includes dataset setup")
         
      else:
         print(f"✗ Deployment script not found: {script_path}")
      
   except Exception as e:
      print(f"✗ Deployment script test failed: {e}")


def main():
   """Main test function."""
   setup_logging()
   
   print("DEPLOYMENT AGENT TEST")
   print("=" * 60)
   print("This script tests the deployment agent functionality.")
   print("Note: This does not actually deploy to Aurora.")
   print("=" * 60)
   
   # Run tests
   test_deployment_agent_creation()
   test_git_info_detection()
   test_configuration_loading()
   test_deployment_script()
   
   print("\n" + "=" * 60)
   print("DEPLOYMENT TESTS COMPLETED")
   print("=" * 60)
   print("If all tests passed, the deployment agent is ready!")
   print("\nTo deploy to Aurora:")
   print("1. Ensure SSH ControlMaster connection to Aurora is active")
   print("2. Run: python src/agents/deployment_agent.py --config config.yaml --action all")
   print("3. Or run: ./scripts/deploy_to_aurora.sh")
   print("\nThe LangGraph workflow will automatically deploy before starting experiments.")


if __name__ == "__main__":
   main() 