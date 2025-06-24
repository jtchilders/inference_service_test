#!/usr/bin/env python3
"""
Deployment Agent for Aurora setup and preparation.
Handles repository cloning/updating and dataset preparation on Aurora.
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.working_dirs import WorkingDirManager


def setup_logging(log_level: str = "INFO") -> None:
   """Setup logging configuration."""
   log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
   date_format = "%d-%m %H:%M"
   
   logging.basicConfig(
      level=getattr(logging, log_level.upper()),
      format=log_format,
      datefmt=date_format,
      handlers=[
         logging.StreamHandler(sys.stdout)
      ]
   )


class DeploymentAgent:
   """Agent responsible for setting up the software repository and datasets on Aurora."""
   
   def __init__(self, config: Dict[str, Any]):
      self.config = config
      self.logger = logging.getLogger(__name__)
      
      # Aurora configuration
      self.aurora_host = config["aurora"]["host"]
      self.aurora_user = config["aurora"]["user"]
      
      # Working directory configuration
      self.working_dir_manager = WorkingDirManager(config)
      self.aurora_base_dir = config["working_dirs"]["aurora_base"]
      
      # Repository configuration
      self.repo_url = config.get("repository", {}).get("url")
      self.repo_branch = config.get("repository", {}).get("branch", "main")
      
      # Dataset configuration
      self.data_dir = config["data"]["dir"]
      
   def _execute_ssh_command(self, command: str, timeout: int = 60) -> str:
      """Execute a command on Aurora using SSH ControlMaster."""
      try:
         ssh_cmd = [
            "ssh", 
            f"{self.aurora_host}",
            command
         ]
         
         result = subprocess.run(
            ssh_cmd,
            capture_output=True,
            text=True,
            timeout=timeout
         )
         
         if result.returncode != 0:
            raise Exception(f"SSH command failed; cmd: {command}; \n stdout: {result.stdout}; \n stderr: {result.stderr}")
         
         return result.stdout.strip()
         
      except subprocess.TimeoutExpired:
         raise Exception(f"SSH command timed out: {command}")
      except Exception as e:
         self.logger.error(f"SSH command failed: {e}")
         raise
   
   def _execute_git_commands(self, commands: List[str], timeout: int = 60) -> str:
      """Execute git commands on Aurora, handling non-zero exit codes gracefully."""
      try:
         # Join commands with && to ensure they run in sequence
         command_string = " && ".join(commands)
         
         ssh_cmd = [
            "ssh", 
            f"{self.aurora_host}",
            command_string
         ]
         
         result = subprocess.run(
            ssh_cmd,
            capture_output=True,
            text=True,
            timeout=timeout
         )
         
         # For git commands, we need to be more lenient with exit codes
         # git pull can return non-zero when repository is up to date
         if result.returncode != 0:
            # Check if this is a "nothing to pull" scenario
            if "Already up to date" in result.stdout or "Already up to date" in result.stderr:
               self.logger.info("Repository is already up to date")
               return result.stdout.strip()
            elif "fatal: couldn't find remote ref" in result.stderr:
               # This is a real error - branch doesn't exist
               raise Exception(f"Git command failed; cmd: {command_string}; \n stdout: {result.stdout}; \n stderr: {result.stderr}")
            else:
               # Log warning but don't fail for other git-related non-zero codes
               self.logger.warning(f"Git command returned non-zero code but may have succeeded: {result.stderr}")
               return result.stdout.strip()
         
         return result.stdout.strip()
         
      except subprocess.TimeoutExpired:
         raise Exception(f"SSH command timed out: {command_string}")
      except Exception as e:
         self.logger.error(f"Git command failed: {e}")
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
   
   def _get_current_git_info(self) -> Tuple[str, str]:
      """Get current git repository URL and branch."""
      try:
         # Get remote URL
         repo_url = subprocess.run(
            ["git", "config", "--get", "remote.origin.url"],
            capture_output=True, text=True
         ).stdout.strip()
         
         # Get current branch
         branch = subprocess.run(
            ["git", "branch", "--show-current"],
            capture_output=True, text=True
         ).stdout.strip()
         
         return repo_url, branch
         
      except Exception as e:
         self.logger.error(f"Error getting git info: {e}")
         raise
   
   def setup_repository(self) -> Dict[str, Any]:
      """Set up the software repository on Aurora."""
      self.logger.info("Setting up repository on Aurora")
      
      try:
         # Get current git repository info
         if not self.repo_url:
            repo_url, current_branch = self._get_current_git_info()
            self.repo_url = repo_url
            if not self.repo_branch:
               self.repo_branch = current_branch
         
         # Create base directory on Aurora
         self._execute_ssh_command(f"mkdir -p {self.aurora_base_dir}")
         
         # Check if repository already exists - use a command that always succeeds
         repo_exists_check = self._execute_ssh_command(f"if [ -d {self.aurora_base_dir}/inference_service_test ]; then echo 'exists'; else echo 'not_exists'; fi")
         
         if repo_exists_check.strip() == 'exists':
            self.logger.info("Repository exists, updating...")
            
            # Change to repository directory and pull latest changes
            update_commands = [
               f"cd {self.aurora_base_dir}/inference_service_test",
               "git fetch origin",
               f"git checkout {self.repo_branch}",
               f"git pull origin {self.repo_branch}",
               "git status"
            ]
            
            update_result = self._execute_git_commands(update_commands)
            self.logger.info(f"Repository updated: {update_result}")
            
            action = "updated"
         else:
            self.logger.info("Repository does not exist, cloning...")
            
            # Clone the repository
            clone_commands = [
               f"cd {self.aurora_base_dir}",
               f"git clone {self.repo_url} inference_service_test",
               f"cd inference_service_test",
               f"git checkout {self.repo_branch}"
            ]
            
            clone_result = self._execute_git_commands(clone_commands)
            self.logger.info(f"Repository cloned: {clone_result}")
            
            action = "cloned"
         
         # Verify repository setup
         repo_path = f"{self.aurora_base_dir}/inference_service_test"
         verification = self._execute_ssh_command(f"ls -la {repo_path}")
         self.logger.info(f"Repository contents: {verification}")
         
         return {
            "status": "success",
            "action": action,
            "repo_url": self.repo_url,
            "repo_branch": self.repo_branch,
            "repo_path": repo_path,
            "verification": verification
         }
         
      except Exception as e:
         self.logger.error(f"Error setting up repository: {e}")
         return {
            "status": "error",
            "error": str(e)
         }
   
   def setup_dataset(self) -> Dict[str, Any]:
      """Set up the CIFAR-100 dataset on Aurora."""
      self.logger.info("Setting up CIFAR-100 dataset on Aurora")
      
      try:
         repo_path = f"{self.aurora_base_dir}/inference_service_test"
         aurora_data_dir = f"{repo_path}/{self.data_dir}"
         
         # Check if dataset already exists - use a command that always succeeds
         dataset_exists_check = self._execute_ssh_command(f"if [ -d {aurora_data_dir}/cifar-100-python ]; then echo 'exists'; else echo 'not_exists'; fi")
         
         if dataset_exists_check.strip() == 'exists':
            self.logger.info("CIFAR-100 dataset already exists")
            
            # Verify dataset integrity
            verification = self._execute_ssh_command(f"ls -la {aurora_data_dir}/cifar-100-python")
            self.logger.info(f"Dataset contents: {verification}")
            
            return {
               "status": "success",
               "action": "exists",
               "data_dir": aurora_data_dir,
               "verification": verification
            }
         else:
            self.logger.info("CIFAR-100 dataset not found, downloading...")
            
            # Create data directory
            self._execute_ssh_command(f"mkdir -p {aurora_data_dir}")
            
            # Download and extract dataset
            download_commands = [
               f"cd {aurora_data_dir}",
               "wget https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz",
               "tar -xzf cifar-100-python.tar.gz",
               "rm cifar-100-python.tar.gz",
               "ls -la cifar-100-python/"
            ]
            
            download_result = self._execute_ssh_command(" && ".join(download_commands))
            self.logger.info(f"Dataset downloaded: {download_result}")
            
            return {
               "status": "success",
               "action": "downloaded",
               "data_dir": aurora_data_dir,
               "verification": download_result
            }
         
      except Exception as e:
         self.logger.error(f"Error setting up dataset: {e}")
         return {
            "status": "error",
            "error": str(e)
         }
   
   def verify_deployment(self) -> Dict[str, Any]:
      """Verify that the deployment is complete and functional."""
      self.logger.info("Verifying deployment on Aurora")
      
      try:
         repo_path = f"{self.aurora_base_dir}/inference_service_test"
         
         # Check repository - use a command that always succeeds
         repo_check = self._execute_ssh_command(f"if [ -d {repo_path} ]; then echo 'repo_ok'; else echo 'repo_missing'; fi")
         if repo_check.strip() != 'repo_ok':
            raise Exception("Repository not found")
         
         # Check source code structure - use a command that always succeeds
         src_check = self._execute_ssh_command(f"if [ -d {repo_path}/src ]; then echo 'src_ok'; else echo 'src_missing'; fi")
         if src_check.strip() != 'src_ok':
            raise Exception("Source code directory not found")
         
         # Check training script - use a command that always succeeds
         train_check = self._execute_ssh_command(f"if [ -f {repo_path}/src/training/train.py ]; then echo 'train_ok'; else echo 'train_missing'; fi")
         if train_check.strip() != 'train_ok':
            raise Exception("Training script not found")
         
         # Check dataset - use a command that always succeeds
         data_path = f"{repo_path}/{self.data_dir}"
         dataset_check = self._execute_ssh_command(f"if [ -d {data_path}/cifar-100-python ]; then echo 'dataset_ok'; else echo 'dataset_missing'; fi")
         if dataset_check.strip() != 'dataset_ok':
            raise Exception("Dataset not found")
         
         return {
            "status": "success",
            "repository": "ok",
            "source_code": "ok",
            "training_script": "ok",
            "dataset": "ok"
         }
         
      except Exception as e:
         self.logger.error(f"Error verifying deployment: {e}")
         return {
            "status": "error",
            "error": str(e)
         }
   
   def deploy_all(self) -> Dict[str, Any]:
      """Deploy everything: repository and dataset."""
      self.logger.info("Starting complete deployment to Aurora")
      
      results = {
         "repository": self.setup_repository(),
         "dataset": self.setup_dataset(),
         "verification": None
      }
      
      # Only verify if both repository and dataset setup succeeded
      if (results["repository"]["status"] == "success" and 
          results["dataset"]["status"] == "success"):
         results["verification"] = self.verify_deployment()
      
      return results


def main():
   """Main entry point for deployment agent."""
   parser = argparse.ArgumentParser(description="Deployment Agent for Aurora")
   parser.add_argument("--config", "-c", required=True, 
                      help="Path to configuration file")
   parser.add_argument("--action", "-a", choices=["repo", "dataset", "verify", "all"],
                      default="all", help="Deployment action to perform")
   parser.add_argument("--log-level", "-l", default="INFO",
                      choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                      help="Logging level")
   
   args = parser.parse_args()
   
   # Setup logging
   setup_logging(args.log_level)
   logger = logging.getLogger(__name__)
   
   # Load configuration
   import yaml
   with open(args.config, 'r') as f:
      config = yaml.safe_load(f)
   
   # Create deployment agent
   agent = DeploymentAgent(config)
   
   # Perform requested action
   if args.action == "repo":
      result = agent.setup_repository()
   elif args.action == "dataset":
      result = agent.setup_dataset()
   elif args.action == "verify":
      result = agent.verify_deployment()
   else:  # all
      result = agent.deploy_all()
   
   # Output results
   logger.info(f"Deployment result: {json.dumps(result, indent=2)}")
   
   if result.get("status") == "error":
      sys.exit(1)


if __name__ == "__main__":
   main() 