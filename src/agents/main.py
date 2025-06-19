#!/usr/bin/env python3
"""
Main orchestrator for the agentic workflow on Crux.
Coordinates between hyperparameter agent (Sophia LLM) and job scheduler (Aurora).
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from agents.hyperparam_agent import HyperparameterAgent
from agents.job_scheduler import JobScheduler
from utils.metrics import calculate_auc
from utils.data_utils import load_training_results
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
         logging.FileHandler("logs/agent_workflow.log"),
         logging.StreamHandler(sys.stdout)
      ]
   )


class AgentWorkflow:
   """Main workflow orchestrator for the agentic hyperparameter search."""
   
   def __init__(self, config: Dict):
      self.config = config
      self.logger = logging.getLogger(__name__)
      
      # Initialize working directory manager
      self.working_dir_manager = WorkingDirManager(config)
      
      # Initialize components
      self.hyperparam_agent = HyperparameterAgent(
         sophia_url=config["sophia"]["url"]
      )
      
      self.job_scheduler = JobScheduler(
         aurora_host=config["aurora"]["host"],
         aurora_user=config["aurora"]["user"],
         pbs_template_path=config["aurora"]["pbs_template"],
         working_dir_config=config.get("working_dirs", {})
      )
      
      # Track experiment state
      self.experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
      self.completed_jobs = []
      self.pending_jobs = []
      
      # Create experiment directories
      self.experiment_dirs = self.working_dir_manager.create_experiment_directories(self.experiment_id)
      self.logger.info(f"Created experiment directories: {self.experiment_dirs}")
      
   def run_experiment(self, max_iterations: int = 10) -> None:
      """Run the main experiment loop."""
      self.logger.info(f"Starting agentic workflow experiment: {self.experiment_id}")
      
      for iteration in range(max_iterations):
         self.logger.info(f"Starting iteration {iteration + 1}/{max_iterations}")
         
         # Get next hyperparameters from LLM agent
         hyperparams = self._get_next_hyperparameters(iteration)
         
         if hyperparams is None:
            self.logger.warning("No hyperparameters suggested, ending experiment")
            break
         
         # Submit job to Aurora
         job_id = self._submit_training_job(hyperparams, iteration)
         
         if job_id:
            self.pending_jobs.append({
               "job_id": job_id,
               "hyperparams": hyperparams,
               "iteration": iteration,
               "submitted_at": datetime.now()
            })
         
         # Wait for jobs to complete and collect results
         self._collect_completed_jobs()
         
         # Update agent with results
         self._update_agent_with_results()
         
         # Check convergence criteria
         if self._should_stop_experiment():
            self.logger.info("Convergence criteria met, ending experiment")
            break
         
         time.sleep(self.config.get("polling_interval", 60))
      
      self.logger.info("Experiment completed")
      self._generate_final_report()
   
   def _get_next_hyperparameters(self, iteration: int) -> Optional[Dict]:
      """Get next hyperparameters from the LLM agent."""
      try:
         # Prepare context for the agent
         context = {
            "iteration": iteration,
            "completed_jobs": self.completed_jobs,
            "experiment_id": self.experiment_id,
            "dataset": "cifar100",
            "objective": "maximize_accuracy"
         }
         
         hyperparams = self.hyperparam_agent.suggest_hyperparameters(context)
         self.logger.info(f"Suggested hyperparameters for iteration {iteration}: {hyperparams}")
         
         return hyperparams
         
      except Exception as e:
         self.logger.error(f"Error getting hyperparameters: {e}")
         return None
   
   def _submit_training_job(self, hyperparams: Dict, iteration: int) -> Optional[str]:
      """Submit a training job to Aurora."""
      try:
         # Get job-specific directories
         job_dirs = self.working_dir_manager.get_job_specific_dirs(
            f"{self.experiment_id}_iter_{iteration}", 
            self.experiment_id
         )
         
         job_config = {
            "job_id": f"{self.experiment_id}_iter_{iteration}",
            "experiment_id": self.experiment_id,
            "output_dir": str(job_dirs["output_dir"]),
            "data_dir": self.config["data"]["dir"],
            **hyperparams
         }
         
         job_id = self.job_scheduler.submit_job(job_config)
         self.logger.info(f"Submitted job {job_id} for iteration {iteration}")
         
         return job_id
         
      except Exception as e:
         self.logger.error(f"Error submitting job: {e}")
         return None
   
   def _collect_completed_jobs(self) -> None:
      """Collect results from completed jobs."""
      completed_jobs = []
      
      for job in self.pending_jobs:
         if self.job_scheduler.is_job_complete(job["job_id"]):
            try:
               results = self.job_scheduler.get_job_results(job["job_id"])
               job["results"] = results
               job["completed_at"] = datetime.now()
               
               self.completed_jobs.append(job)
               completed_jobs.append(job)
               
               self.logger.info(f"Job {job['job_id']} completed with results: {results}")
               
            except Exception as e:
               self.logger.error(f"Error collecting results for job {job['job_id']}: {e}")
      
      # Remove completed jobs from pending
      for job in completed_jobs:
         self.pending_jobs.remove(job)
   
   def _update_agent_with_results(self) -> None:
      """Update the LLM agent with new results."""
      if not self.completed_jobs:
         return
      
      try:
         # Prepare results summary for the agent
         results_summary = []
         for job in self.completed_jobs:
            results_summary.append({
               "hyperparams": job["hyperparams"],
               "accuracy": job["results"].get("final_accuracy", 0.0),
               "auc": job["results"].get("auc", 0.0),
               "training_time": job["results"].get("training_time", 0.0)
            })
         
         self.hyperparam_agent.update_with_results(results_summary)
         
      except Exception as e:
         self.logger.error(f"Error updating agent with results: {e}")
   
   def _should_stop_experiment(self) -> bool:
      """Check if experiment should stop based on convergence criteria."""
      if len(self.completed_jobs) < 3:
         return False
      
      # Check if accuracy has plateaued
      recent_accuracies = [job["results"].get("final_accuracy", 0.0) 
                          for job in self.completed_jobs[-3:]]
      
      if max(recent_accuracies) - min(recent_accuracies) < 0.01:
         self.logger.info("Accuracy plateaued, stopping experiment")
         return True
      
      return False
   
   def _generate_final_report(self) -> None:
      """Generate final experiment report."""
      report_path = self.experiment_dirs["reports_dir"] / f"experiment_report_{self.experiment_id}.json"
      
      report = {
         "experiment_id": self.experiment_id,
         "total_jobs": len(self.completed_jobs),
         "best_accuracy": max([job["results"].get("final_accuracy", 0.0) 
                             for job in self.completed_jobs]),
         "best_hyperparams": None,
         "all_results": self.completed_jobs
      }
      
      # Find best hyperparameters
      best_job = max(self.completed_jobs, 
                    key=lambda x: x["results"].get("final_accuracy", 0.0))
      report["best_hyperparams"] = best_job["hyperparams"]
      
      import json
      with open(report_path, 'w') as f:
         json.dump(report, f, indent=3, default=str)
      
      self.logger.info(f"Final report saved to: {report_path}")


def main():
   """Main entry point."""
   parser = argparse.ArgumentParser(description="Agentic workflow orchestrator")
   parser.add_argument("--config", "-c", required=True, 
                      help="Path to configuration file")
   parser.add_argument("--max-iterations", "-m", type=int, default=10,
                      help="Maximum number of iterations")
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
   
   # Create logs directory
   Path("logs").mkdir(exist_ok=True)
   
   # Run workflow
   workflow = AgentWorkflow(config)
   workflow.run_experiment(max_iterations=args.max_iterations)


if __name__ == "__main__":
   main() 