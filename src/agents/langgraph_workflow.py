#!/usr/bin/env python3
"""
LangGraph-based agentic workflow orchestrator for hyperparameter optimization.
Uses LangGraph to create a structured workflow with state management.
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, TypedDict, Annotated
from dataclasses import dataclass

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from agents.langchain_hyperparam_agent import LangChainHyperparameterAgent
from agents.job_scheduler import JobScheduler
from utils.metrics import calculate_auc
from utils.data_utils import load_training_results
from utils.working_dirs import WorkingDirManager
from inference_auth_token import get_access_token


def setup_logging(log_level: str = "INFO") -> None:
   """Setup logging configuration."""
   log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
   date_format = "%d-%m %H:%M"

   # Create logs directory
   Path("logs").mkdir(exist_ok=True)

   # create date/time string for log file name
   log_file_name = datetime.now().strftime("%Y%m%d_%H%M%S")

   logging.basicConfig(
      level=getattr(logging, log_level.upper()),
      format=log_format,
      datefmt=date_format,
      handlers=[
         logging.FileHandler(f"logs/langgraph_workflow_{log_file_name}.log"),
         logging.StreamHandler(sys.stdout)
      ]
   )


@dataclass
class WorkflowState:
   """State for the LangGraph workflow."""
   iteration: int = 0
   max_iterations: int = 10
   experiment_id: str = ""
   completed_jobs: List[Dict[str, Any]] = None
   pending_jobs: List[Dict[str, Any]] = None
   current_hyperparams: Optional[Dict[str, Any]] = None
   current_job_id: Optional[str] = None
   should_stop: bool = False
   error_message: Optional[str] = None
   final_report: Optional[Dict[str, Any]] = None
   
   def __post_init__(self):
      if self.completed_jobs is None:
         self.completed_jobs = []
      if self.pending_jobs is None:
         self.pending_jobs = []
      if not self.experiment_id:
         self.experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")


class LangGraphWorkflow:
   """LangGraph-based workflow orchestrator for hyperparameter optimization."""
   
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
      
      self.job_scheduler = JobScheduler(
         aurora_host=config["aurora"]["host"],
         aurora_user=config["aurora"]["user"],
         pbs_template_path=config["aurora"]["pbs_template"],
         working_dir_config=config.get("working_dirs", {})
      )
      
      # Initialize LangChain components
      self.llm = ChatOpenAI(
         base_url=config["sophia"]["url"],
         api_key=self.api_key,
         model="meta-llama/Llama-3.3-70B-Instruct",
         temperature=0.1
      )
      
      # Create the workflow graph
      self.workflow = self._create_workflow()
   
   def _create_workflow(self) -> StateGraph:
      """Create the LangGraph workflow."""
      
      # Create the state graph
      workflow = StateGraph(WorkflowState)
      
      # Add nodes
      workflow.add_node("analyze_results", self._analyze_results_node)
      workflow.add_node("suggest_hyperparams", self._suggest_hyperparams_node)
      workflow.add_node("submit_job", self._submit_job_node)
      workflow.add_node("collect_results", self._collect_results_node)
      workflow.add_node("check_convergence", self._check_convergence_node)
      workflow.add_node("generate_report", self._generate_report_node)
      
      # Add edges
      workflow.add_edge("analyze_results", "suggest_hyperparams")
      workflow.add_edge("suggest_hyperparams", "submit_job")
      workflow.add_edge("submit_job", "collect_results")
      workflow.add_edge("collect_results", "check_convergence")
      workflow.add_conditional_edges(
         "check_convergence",
         self._should_continue,
         {
            "continue": "analyze_results",
            "stop": "generate_report"
         }
      )
      workflow.add_edge("generate_report", END)
      
      # Set entry point
      workflow.set_entry_point("analyze_results")
      
      return workflow.compile()
   
   def _analyze_results_node(self, state: WorkflowState) -> WorkflowState:
      """Analyze previous results and prepare context for hyperparameter suggestion."""
      self.logger.info(f"Analyzing results for iteration {state.iteration}")
      
      try:
         # Update agent with latest results
         if state.completed_jobs:
            results_summary = []
            for job in state.completed_jobs:
               results_summary.append({
                  "hyperparams": job["hyperparams"],
                  "accuracy": job["results"].get("final_accuracy", 0.0),
                  "auc": job["results"].get("auc", 0.0),
                  "training_time": job["results"].get("training_time", 0.0)
               })
            
            self.hyperparam_agent.update_with_results(results_summary)
         
         return state
         
      except Exception as e:
         self.logger.error(f"Error in analyze_results_node: {e}")
         state.error_message = str(e)
         return state
   
   def _suggest_hyperparams_node(self, state: WorkflowState) -> WorkflowState:
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
         
         self.logger.info(f"Suggested hyperparameters: {hyperparams}")
         return state
         
      except Exception as e:
         self.logger.error(f"Error in suggest_hyperparams_node: {e}")
         state.error_message = str(e)
         return state
   
   def _submit_job_node(self, state: WorkflowState) -> WorkflowState:
      """Submit a training job to Aurora."""
      self.logger.info(f"Submitting job for iteration {state.iteration}")
      
      try:
         if not state.current_hyperparams:
            raise ValueError("No hyperparameters available for job submission")
         
         # Get job-specific directories
         job_dirs = self.working_dir_manager.get_job_specific_dirs(
            f"{state.experiment_id}_iter_{state.iteration}", 
            state.experiment_id
         )
         
         job_config = {
            "job_id": f"{state.experiment_id}_iter_{state.iteration}",
            "experiment_id": state.experiment_id,
            "output_dir": str(job_dirs["output_dir"]),
            "data_dir": self.config["data"]["dir"],
            **state.current_hyperparams
         }
         
         job_id = self.job_scheduler.submit_job(job_config)
         state.current_job_id = job_id
         
         # Add to pending jobs
         state.pending_jobs.append({
            "job_id": job_id,
            "hyperparams": state.current_hyperparams,
            "iteration": state.iteration,
            "submitted_at": datetime.now()
         })
         
         self.logger.info(f"Submitted job {job_id} for iteration {state.iteration}")
         return state
         
      except Exception as e:
         self.logger.error(f"Error in submit_job_node: {e}")
         state.error_message = str(e)
         return state
   
   def _collect_results_node(self, state: WorkflowState) -> WorkflowState:
      """Collect results from completed jobs."""
      self.logger.info(f"Collecting results for iteration {state.iteration}")
      
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
         
         if waited_time >= max_wait_time:
            self.logger.warning(f"Timeout waiting for job completion in iteration {state.iteration}")
         
         return state
         
      except Exception as e:
         self.logger.error(f"Error in collect_results_node: {e}")
         state.error_message = str(e)
         return state
   
   def _check_convergence_node(self, state: WorkflowState) -> WorkflowState:
      """Check if the experiment should stop based on convergence criteria."""
      self.logger.info(f"Checking convergence for iteration {state.iteration}")
      
      try:
         # Increment iteration
         state.iteration += 1
         
         # Check if we've reached max iterations
         if state.iteration >= state.max_iterations:
            self.logger.info("Reached maximum iterations, stopping experiment")
            state.should_stop = True
            return state
         
         # Check if we have enough results to assess convergence
         if len(state.completed_jobs) < 3:
            return state
         
         # Check if accuracy has plateaued
         recent_accuracies = [job["results"].get("final_accuracy", 0.0) 
                            for job in state.completed_jobs[-3:]]
         
         if max(recent_accuracies) - min(recent_accuracies) < 0.01:
            self.logger.info("Accuracy plateaued, stopping experiment")
            state.should_stop = True
         
         return state
         
      except Exception as e:
         self.logger.error(f"Error in check_convergence_node: {e}")
         state.error_message = str(e)
         return state
   
   def _should_continue(self, state: WorkflowState) -> str:
      """Determine if the workflow should continue or stop."""
      if state.should_stop or state.error_message:
         return "stop"
      return "continue"
   
   def _generate_report_node(self, state: WorkflowState) -> WorkflowState:
      """Generate final experiment report."""
      self.logger.info("Generating final experiment report")
      
      try:
         # Create experiment directories if not already created
         experiment_dirs = self.working_dir_manager.create_experiment_directories(state.experiment_id)
         report_path = experiment_dirs["reports_dir"] / f"experiment_report_{state.experiment_id}.json"
         
         report = {
            "experiment_id": state.experiment_id,
            "total_jobs": len(state.completed_jobs),
            "total_iterations": state.iteration,
            "best_accuracy": 0.0,
            "best_hyperparams": None,
            "all_results": state.completed_jobs,
            "error_message": state.error_message
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
         return state
   
   def run_experiment(self, max_iterations: int = 10) -> Dict[str, Any]:
      """Run the LangGraph workflow."""
      self.logger.info(f"Starting LangGraph workflow experiment")
      
      # Initialize state
      initial_state = WorkflowState(
         max_iterations=max_iterations,
         experiment_id=datetime.now().strftime("%Y%m%d_%H%M%S")
      )
      
      try:
         # Run the workflow
         final_state = self.workflow.invoke(initial_state)
         
         self.logger.info("LangGraph workflow completed")
         return final_state.final_report or {}
         
      except Exception as e:
         self.logger.error(f"Error running LangGraph workflow: {e}")
         return {"error": str(e)}


def main():
   """Main entry point."""
   parser = argparse.ArgumentParser(description="LangGraph agentic workflow orchestrator")
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
   workflow = LangGraphWorkflow(config)
   result = workflow.run_experiment(max_iterations=args.max_iterations)
   
   logger.info(f"Workflow completed with result: {result}")


if __name__ == "__main__":
   main() 