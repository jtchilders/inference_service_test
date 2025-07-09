#!/usr/bin/env python3
"""
Advanced LangGraph-based agentic workflow with tools and conditional routing.
Uses LangGraph for structured workflow management with advanced features.
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import os

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from agents.langchain_hyperparam_agent import LangChainHyperparameterAgent, AuthenticationError
from agents.job_scheduler import JobScheduler
from agents.globus_job_scheduler import GlobusJobScheduler
from utils.metrics import calculate_auc
from utils.data_utils import load_training_results
from utils.working_dirs import WorkingDirManager
from inference_auth_token import get_access_token


def setup_logging(log_level: str = "INFO") -> None:
   """Setup logging configuration."""
   log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
   date_format = "%d-%m %H:%M"
   
   logging.basicConfig(
      level=getattr(logging, log_level.upper()),
      format=log_format,
      datefmt=date_format,
      handlers=[
         logging.FileHandler("logs/advanced_langgraph_workflow.log"),
         logging.StreamHandler(sys.stdout)
      ]
   )


class WorkflowStatus(Enum):
   """Status of the workflow."""
   INITIALIZING = "initializing"
   RUNNING = "running"
   CONVERGED = "converged"
   ERROR = "error"
   COMPLETED = "completed"


@dataclass
class AdvancedWorkflowState:
   """Advanced state for the LangGraph workflow."""
   iteration: int = 0
   max_iterations: int = 10
   experiment_id: str = ""
   status: WorkflowStatus = WorkflowStatus.INITIALIZING
   completed_jobs: List[Dict[str, Any]] = field(default_factory=list)
   pending_jobs: List[Dict[str, Any]] = field(default_factory=list)
   current_hyperparams: Optional[Dict[str, Any]] = None
   should_stop: bool = False
   error_message: Optional[str] = None
   best_accuracy: float = 0.0
   final_report: Optional[Dict[str, Any]] = None
   messages: List[BaseMessage] = field(default_factory=list)
   
   def __post_init__(self):
      if not self.experiment_id:
         self.experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")


class AdvancedLangGraphWorkflow:
   """Advanced LangGraph-based workflow with tools and conditional routing."""
   
   def __init__(self, config: Dict[str, Any]):
      self.config = config
      self.logger = logging.getLogger(__name__)
      
      # Get authentication token
      self.api_key = get_access_token()
      
      # Initialize working directory manager
      self.working_dir_manager = WorkingDirManager(config)
      
      # Initialize components
      self.hyperparam_agent = LangChainHyperparameterAgent(
         sophia_url=config["sophia"]["url"],
         api_key=self.api_key
      )
      
      # Choose job scheduler based on configuration
      if config.get("globus_compute", {}).get("enabled", False):
         self.job_scheduler = GlobusJobScheduler(
            endpoint_id=config["globus_compute"]["endpoint_id"],
            working_dir_config=config.get("working_dirs", {}),
            auth_method=config["globus_compute"].get("auth_method", "native_client"),
            function_timeout=config["globus_compute"].get("function_timeout", 3600),
            max_retries=config["globus_compute"].get("max_retries", 2)
         )
         self.logger.info("Using Globus Compute job scheduler")
      else:
         self.job_scheduler = JobScheduler(
            aurora_host=config["aurora"]["host"],
            aurora_user=config["aurora"]["user"],
            pbs_template_path=config["aurora"]["pbs_template"],
            working_dir_config=config.get("working_dirs", {}),
            queue=config["aurora"].get("queue", "workq")
         )
         self.logger.info("Using direct PBS job scheduler")
      
      # Create the workflow graph
      self.workflow = self._create_workflow()
   
   def _create_workflow(self) -> StateGraph:
      """Create the advanced LangGraph workflow."""
      
      # Create the state graph
      workflow = StateGraph(AdvancedWorkflowState)
      
      # Add nodes
      workflow.add_node("initialize", self._initialize_node)
      workflow.add_node("suggest_hyperparams", self._suggest_hyperparams_node)
      workflow.add_node("submit_job", self._submit_job_node)
      workflow.add_node("collect_results", self._collect_results_node)
      workflow.add_node("check_convergence", self._check_convergence_node)
      workflow.add_node("generate_report", self._generate_report_node)
      
      # Add edges with conditional routing
      workflow.add_edge("initialize", "suggest_hyperparams")
      workflow.add_edge("suggest_hyperparams", "submit_job")
      workflow.add_edge("submit_job", "collect_results")
      workflow.add_edge("collect_results", "check_convergence")
      workflow.add_conditional_edges(
         "check_convergence",
         self._should_continue,
         {
            "continue": "suggest_hyperparams",
            "stop": "generate_report"
         }
      )
      workflow.add_edge("generate_report", END)
      
      # Set entry point
      workflow.set_entry_point("initialize")
      
      return workflow.compile()
   
   def _initialize_node(self, state: AdvancedWorkflowState) -> AdvancedWorkflowState:
      """Initialize the workflow state."""
      self.logger.info(f"Initializing workflow for experiment {state.experiment_id}")
      
      try:
         state.status = WorkflowStatus.RUNNING
         state.messages.append(SystemMessage(content="Starting hyperparameter optimization workflow"))
         
         return state
         
      except Exception as e:
         self.logger.error(f"Error in initialize_node: {e}")
         state.error_message = str(e)
         state.status = WorkflowStatus.ERROR
         return state
   
   def _suggest_hyperparams_node(self, state: AdvancedWorkflowState) -> AdvancedWorkflowState:
      """Get hyperparameter suggestions from the LLM agent."""
      self.logger.info(f"Getting hyperparameter suggestions for iteration {state.iteration}")
      
      try:
         # Prepare context for the agent
         context = {
            "iteration": state.iteration,
            "completed_jobs": state.completed_jobs,
            "experiment_id": state.experiment_id,
            "dataset": "cifar100",
            "objective": "maximize_accuracy"
         }
         
         hyperparams = self.hyperparam_agent.suggest_hyperparameters(context)
         state.current_hyperparams = hyperparams
         
         # Add suggestion message
         state.messages.append(AIMessage(content=f"Suggested hyperparameters: {json.dumps(hyperparams, indent=2)}"))
         
         self.logger.info(f"Suggested hyperparameters: {hyperparams}")
         return state
         
      except AuthenticationError as e:
         self.logger.error(f"Authentication error in suggest_hyperparams_node: {e}")
         state.error_message = str(e)
         state.status = WorkflowStatus.ERROR
         state.should_stop = True
         raise  # Re-raise to halt execution
      except Exception as e:
         self.logger.error(f"Error in suggest_hyperparams_node: {e}")
         state.error_message = str(e)
         state.status = WorkflowStatus.ERROR
         return state
   
   def _submit_job_node(self, state: AdvancedWorkflowState) -> AdvancedWorkflowState:
      """Submit a training job to Aurora."""
      self.logger.info(f"Submitting job for iteration {state.iteration}")
      
      try:
         # Get job-specific directories (local and Aurora)
         job_dirs = self.working_dir_manager.get_job_specific_dirs(
            f"{state.experiment_id}_iter_{state.iteration}", 
            state.experiment_id
         )
         aurora_job_dirs = self.working_dir_manager.get_aurora_job_specific_dirs(
            f"{state.experiment_id}_iter_{state.iteration}",
            state.experiment_id
         )
         
         job_config = {
            "job_id": f"{state.experiment_id}_iter_{state.iteration}",
            "experiment_id": state.experiment_id,
            "output_dir": aurora_job_dirs["output_dir"],
            "data_dir": self.config["data"]["dir"],
            "queue": self.config["aurora"].get("queue", "workq"),
            **state.current_hyperparams
         }
         
         job_id = self.job_scheduler.submit_job(job_config)
         
         # Add to pending jobs
         state.pending_jobs.append({
            "job_id": job_id,
            "hyperparams": state.current_hyperparams,
            "iteration": state.iteration,
            "submitted_at": datetime.now()
         })
         
         # Add submission message
         state.messages.append(AIMessage(content=f"Submitted job {job_id} for iteration {state.iteration}"))
         
         self.logger.info(f"Submitted job {job_id} for iteration {state.iteration}")
         return state
         
      except Exception as e:
         self.logger.error(f"Error in submit_job_node: {e}")
         state.error_message = str(e)
         state.status = WorkflowStatus.ERROR
         return state
   
   def _collect_results_node(self, state: AdvancedWorkflowState) -> AdvancedWorkflowState:
      """Collect results from completed jobs."""
      self.logger.info("Collecting job results")
      
      try:
         # Wait for jobs to complete
         max_wait_time = 300  # 5 minutes
         wait_interval = 30   # 30 seconds
         waited_time = 0
         
         while waited_time < max_wait_time:
            completed_jobs = []
            
            for job in state.pending_jobs:
               if self.job_scheduler.is_job_complete(job["job_id"]):
                  try:
                     results = self.job_scheduler.get_job_results(job["job_id"])
                     job["results"] = results
                     job["completed_at"] = datetime.now()
                     
                     state.completed_jobs.append(job)
                     completed_jobs.append(job)
                     
                     self.logger.info(f"Job {job['job_id']} completed with results: {results}")
                     
                  except Exception as e:
                     self.logger.error(f"Error collecting results for job {job['job_id']}: {e}")
            
            # Remove completed jobs from pending
            for job in completed_jobs:
               state.pending_jobs.remove(job)
            
            # If we have results, break
            if completed_jobs:
               break
            
            time.sleep(wait_interval)
            waited_time += wait_interval
         
         return state
         
      except Exception as e:
         self.logger.error(f"Error in collect_results_node: {e}")
         state.error_message = str(e)
         state.status = WorkflowStatus.ERROR
         return state
   
   def _check_convergence_node(self, state: AdvancedWorkflowState) -> AdvancedWorkflowState:
      """Check if the experiment should stop based on convergence criteria."""
      self.logger.info("Checking convergence criteria")
      
      try:
         # Increment iteration
         state.iteration += 1
         
         # Check max iterations
         if state.iteration >= state.max_iterations:
            state.should_stop = True
            state.status = WorkflowStatus.COMPLETED
            return state
         
         # Check if we have enough results
         if len(state.completed_jobs) < 3:
            return state
         
         # Check accuracy plateau
         recent_accuracies = [job["results"].get("final_accuracy", 0.0) 
                            for job in state.completed_jobs[-3:]]
         
         if max(recent_accuracies) - min(recent_accuracies) < 0.01:
            state.should_stop = True
            state.status = WorkflowStatus.CONVERGED
         
         return state
         
      except Exception as e:
         self.logger.error(f"Error in check_convergence_node: {e}")
         state.error_message = str(e)
         state.status = WorkflowStatus.ERROR
         return state
   
   def _should_continue(self, state: AdvancedWorkflowState) -> str:
      """Determine if the workflow should continue or stop."""
      if state.should_stop or state.error_message:
         return "stop"
      return "continue"
   
   def _generate_report_node(self, state: AdvancedWorkflowState) -> AdvancedWorkflowState:
      """Generate final experiment report."""
      self.logger.info("Generating final experiment report")
      
      try:
         # Create experiment directories if not already created
         experiment_dirs = self.working_dir_manager.create_experiment_directories(state.experiment_id)
         report_path = experiment_dirs["reports_dir"] / f"advanced_experiment_report_{state.experiment_id}.json"
         
         report = {
            "experiment_id": state.experiment_id,
            "status": state.status.value,
            "total_jobs": len(state.completed_jobs),
            "total_iterations": state.iteration,
            "best_accuracy": 0.0,
            "best_hyperparams": None,
            "all_results": state.completed_jobs,
            "error_message": state.error_message,
            "workflow_messages": [msg.content for msg in state.messages]
         }
         
         if state.completed_jobs:
            report["best_accuracy"] = max([job["results"].get("final_accuracy", 0.0) 
                                         for job in state.completed_jobs])
            
            # Find best hyperparameters
            best_job = max(state.completed_jobs, 
                          key=lambda x: x["results"].get("final_accuracy", 0.0))
            report["best_hyperparams"] = best_job["hyperparams"]
         
         state.final_report = report
         
         with open(report_path, 'w') as f:
            json.dump(report, f, indent=3, default=str)
         
         self.logger.info(f"Final report saved to: {report_path}")
         return state
         
      except Exception as e:
         self.logger.error(f"Error in generate_report_node: {e}")
         state.error_message = str(e)
         state.status = WorkflowStatus.ERROR
         return state
   
   def run_experiment(self, max_iterations: int = 10) -> Dict[str, Any]:
      """Run the advanced LangGraph workflow."""
      self.logger.info(f"Starting advanced LangGraph workflow experiment")
      
      # Initialize state
      initial_state = AdvancedWorkflowState(
         max_iterations=max_iterations,
         experiment_id=datetime.now().strftime("%Y%m%d_%H%M%S")
      )
      
      try:
         # Run the workflow
         final_state = self.workflow.invoke(initial_state)
         
         self.logger.info(f"Advanced LangGraph workflow completed with status: {final_state.status.value}")
         return final_state.final_report or {}
         
      except Exception as e:
         self.logger.error(f"Error running advanced LangGraph workflow: {e}")
         return {"error": str(e), "status": "failed"}


def main():
   """Main entry point."""
   parser = argparse.ArgumentParser(description="Advanced LangGraph agentic workflow orchestrator")
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
   workflow = AdvancedLangGraphWorkflow(config)
   result = workflow.run_experiment(max_iterations=args.max_iterations)
   
   logger.info(f"Advanced workflow completed with result: {result}")


if __name__ == "__main__":
   main() 