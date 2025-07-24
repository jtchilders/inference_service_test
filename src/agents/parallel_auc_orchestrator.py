#!/usr/bin/env python3
"""
Parallel AUC Optimization Orchestrator.
Manages multiple concurrent training jobs with real-time monitoring and intelligent job scheduling.
"""

import asyncio
import json
import logging
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, deque
import concurrent.futures
import threading
import sys
import os

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from agents.auc_optimization_agent import AUCOptimizationAgent
from agents.globus_job_scheduler import GlobusJobScheduler
from utils.working_dirs import WorkingDirManager


@dataclass
class JobStatus:
   """Track status of individual training jobs."""
   job_id: str
   hyperparams: Dict[str, Any]
   iteration: int
   submitted_at: datetime
   status: str = "submitted"  # submitted, running, completed, failed, cancelled
   task_id: Optional[str] = None
   results: Optional[Dict[str, Any]] = None
   completed_at: Optional[datetime] = None
   error_message: Optional[str] = None
   retry_count: int = 0


@dataclass
class OptimizationMetrics:
   """Track optimization progress metrics."""
   total_jobs_submitted: int = 0
   total_jobs_completed: int = 0
   total_jobs_failed: int = 0
   best_auc: float = 0.0
   best_hyperparams: Optional[Dict[str, Any]] = None
   recent_auc_history: deque = field(default_factory=lambda: deque(maxlen=20))
   iteration_times: List[float] = field(default_factory=list)
   exploration_factors: List[float] = field(default_factory=list)
   
   @property
   def success_rate(self) -> float:
      total = self.total_jobs_completed + self.total_jobs_failed
      return (self.total_jobs_completed / total) if total > 0 else 0.0
   
   @property
   def recent_auc_mean(self) -> float:
      return np.mean(self.recent_auc_history) if self.recent_auc_history else 0.0
   
   @property
   def recent_auc_std(self) -> float:
      return np.std(self.recent_auc_history) if len(self.recent_auc_history) > 1 else 0.0


class ParallelAUCOrchestrator:
   """
   Orchestrates parallel hyperparameter optimization for AUC maximization.
   
   Features:
   - Parallel job submission and monitoring
   - Real-time statistics and progress tracking
   - Adaptive job scheduling based on system load
   - Periodic plot generation and reporting
   - Intelligent retry and error handling
   """
   
   def __init__(self, config: Dict[str, Any]):
      self.config = config
      self.logger = logging.getLogger(__name__)
      
      # Initialize components
      self.working_dir_manager = WorkingDirManager(config)
      self.optimization_agent = AUCOptimizationAgent(
         sophia_url=config["sophia"]["url"],
         config=config
      )
      self.job_scheduler = GlobusJobScheduler(
         endpoint_id=config["globus_compute"]["endpoint_id"],
         working_dir_config=config.get("working_dirs", {}),
         auth_method=config["globus_compute"].get("auth_method", "native_client"),
         function_timeout=config["globus_compute"].get("function_timeout", 7200),
         max_retries=config["globus_compute"].get("max_retries", 3)
      )
      
      # Parallel execution settings
      self.parallel_config = config.get("globus_compute", {}).get("parallel_jobs", {})
      self.max_concurrent_jobs = self.parallel_config.get("max_concurrent", 12)
      self.initial_batch_size = self.parallel_config.get("initial_batch_size", 8)
      self.adaptive_scaling = self.parallel_config.get("adaptive_scaling", True)
      self.job_spacing_seconds = self.parallel_config.get("job_spacing_seconds", 5)
      
      # Optimization settings
      self.optimization_config = config.get("optimization", {})
      self.max_iterations = self.optimization_config.get("max_iterations", 100)
      self.convergence_patience = self.optimization_config.get("convergence_patience", 15)
      
      # Monitoring settings
      self.monitoring_config = config.get("monitoring", {})
      self.update_interval = self.monitoring_config.get("update_interval_seconds", 30)
      self.plot_update_interval = self.monitoring_config.get("plot_update_interval_seconds", 120)
      
      # State tracking
      self.experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
      self.start_time = datetime.now()
      
      self.active_jobs: Dict[str, JobStatus] = {}
      self.completed_jobs: List[JobStatus] = []
      self.failed_jobs: List[JobStatus] = []
      
      self.metrics = OptimizationMetrics()
      self.iteration_count = 0
      self.convergence_counter = 0
      
      # Threading for monitoring
      self.monitoring_active = False
      self.monitoring_thread = None
      self.plot_thread = None
      
      # Results directory
      self.results_dir = Path(f"results/agentic_optimization_{self.experiment_id}")
      self.results_dir.mkdir(parents=True, exist_ok=True)
      self.plots_dir = self.results_dir / "plots"
      self.plots_dir.mkdir(exist_ok=True)
      
      self.logger.info(f"Initialized Parallel AUC Orchestrator for experiment {self.experiment_id}")
      self.logger.info(f"Results directory: {self.results_dir}")
   
   def run_optimization(self) -> Dict[str, Any]:
      """
      Run the complete parallel hyperparameter optimization.
      
      Returns:
         Dict containing final optimization results
      """
      
      self.logger.info("=" * 80)
      self.logger.info(f"Starting Parallel AUC Optimization Experiment: {self.experiment_id}")
      self.logger.info(f"Target: Maximize AUC on CIFAR-100")
      self.logger.info(f"Max iterations: {self.max_iterations}")
      self.logger.info(f"Max concurrent jobs: {self.max_concurrent_jobs}")
      self.logger.info("=" * 80)
      
      try:
         # Start monitoring threads
         self._start_monitoring()
         
         # Main optimization loop
         while self.iteration_count < self.max_iterations:
            self.logger.info(f"\n--- Iteration {self.iteration_count + 1}/{self.max_iterations} ---")
            
            # Determine batch size for this iteration
            batch_size = self._calculate_batch_size()
            
            # Submit batch of jobs
            submitted_jobs = self._submit_job_batch(batch_size)
            
            if not submitted_jobs:
               self.logger.warning("No jobs submitted, ending optimization")
               break
            
            # Wait for some jobs to complete before submitting more
            self._wait_for_job_completion(min_completions=1, timeout_minutes=30)
            
            # Update optimization state
            self._update_optimization_state()
            
            # Check convergence
            if self._check_convergence():
               self.logger.info(f"Convergence achieved after {self.iteration_count} iterations")
               break
            
            self.iteration_count += 1
         
         # Wait for all remaining jobs to complete
         self.logger.info("Waiting for remaining jobs to complete...")
         self._wait_for_all_jobs(timeout_minutes=60)
         
         # Generate final report
         final_report = self._generate_final_report()
         
         self.logger.info("=" * 80)
         self.logger.info("OPTIMIZATION COMPLETED")
         self.logger.info(f"Best AUC: {self.metrics.best_auc:.6f}")
         self.logger.info(f"Total jobs: {self.metrics.total_jobs_submitted}")
         self.logger.info(f"Success rate: {self.metrics.success_rate:.1%}")
         self.logger.info("=" * 80)
         
         return final_report
         
      except KeyboardInterrupt:
         self.logger.info("Optimization interrupted by user")
         return self._handle_interruption()
         
      except Exception as e:
         self.logger.error(f"Optimization failed: {e}")
         return {"status": "failed", "error": str(e)}
         
      finally:
         self._stop_monitoring()
         self._cleanup()
   
   def _submit_job_batch(self, batch_size: int) -> List[str]:
      """Submit a batch of training jobs."""
      
      submitted_job_ids = []
      
      for i in range(batch_size):
         try:
            # Get hyperparameter suggestions from agent
            context = self._build_agent_context()
            hyperparams = self.optimization_agent.suggest_hyperparameters(context)
            
            if not hyperparams:
               self.logger.warning("No hyperparameters suggested")
               continue
            
            # Create job configuration
            job_id = f"{self.experiment_id}_iter_{self.iteration_count}_{i:03d}"
            job_config = self._build_job_config(job_id, hyperparams)
            
            # Submit job
            submitted_job_id = self.job_scheduler.submit_job(job_config)
            
            # Track job status
            job_status = JobStatus(
               job_id=submitted_job_id,
               hyperparams=hyperparams,
               iteration=self.iteration_count,
               submitted_at=datetime.now(),
               task_id=submitted_job_id  # For Globus Compute, job_id == task_id
            )
            
            self.active_jobs[submitted_job_id] = job_status
            submitted_job_ids.append(submitted_job_id)
            
            self.metrics.total_jobs_submitted += 1
            
            self.logger.info(f"Submitted job {submitted_job_id}: {hyperparams}")
            
            # Space out job submissions
            if i < batch_size - 1:
               time.sleep(self.job_spacing_seconds)
               
         except Exception as e:
            self.logger.error(f"Failed to submit job {i+1}/{batch_size}: {e}")
            continue
      
      return submitted_job_ids
   
   def _wait_for_job_completion(self, min_completions: int = 1, timeout_minutes: int = 30) -> None:
      """Wait for a minimum number of jobs to complete."""
      
      start_time = time.time()
      timeout_seconds = timeout_minutes * 60
      completed_count = 0
      
      while completed_count < min_completions and time.time() - start_time < timeout_seconds:
         # Check status of active jobs
         newly_completed = self._check_job_statuses()
         completed_count += newly_completed
         
         if completed_count >= min_completions:
            break
         
         # Brief wait before checking again
         time.sleep(10)
      
      if completed_count == 0:
         self.logger.warning(f"No jobs completed within {timeout_minutes} minutes")
   
   def _wait_for_all_jobs(self, timeout_minutes: int = 60) -> None:
      """Wait for all active jobs to complete."""
      
      start_time = time.time()
      timeout_seconds = timeout_minutes * 60
      
      while self.active_jobs and time.time() - start_time < timeout_seconds:
         self._check_job_statuses()
         time.sleep(15)
      
      # Force timeout for remaining jobs
      if self.active_jobs:
         self.logger.warning(f"Timing out {len(self.active_jobs)} remaining jobs")
         for job_id in list(self.active_jobs.keys()):
            job_status = self.active_jobs[job_id]
            job_status.status = "timeout"
            job_status.completed_at = datetime.now()
            self.failed_jobs.append(job_status)
            del self.active_jobs[job_id]
            self.metrics.total_jobs_failed += 1
   
   def _check_job_statuses(self) -> int:
      """Check status of all active jobs and process completed ones."""
      
      newly_completed = 0
      completed_job_ids = []
      
      for job_id, job_status in self.active_jobs.items():
         try:
            if self.job_scheduler.is_job_complete(job_id):
               # Get job results
               results = self.job_scheduler.get_job_results(job_id)
               
               job_status.results = results
               job_status.completed_at = datetime.now()
               
               if results.get("status") == "completed":
                  job_status.status = "completed"
                  self.completed_jobs.append(job_status)
                  self.metrics.total_jobs_completed += 1
                  newly_completed += 1
                  
                  # Update metrics
                  auc = results.get("training_results", {}).get("auc", 0.0)
                  self.metrics.recent_auc_history.append(auc)
                  
                  if auc > self.metrics.best_auc:
                     self.metrics.best_auc = auc
                     self.metrics.best_hyperparams = job_status.hyperparams.copy()
                     self.logger.info(f"🎉 NEW BEST AUC: {auc:.6f} from job {job_id}")
                  
                  self.logger.info(f"✅ Job {job_id} completed: AUC = {auc:.6f}")
               else:
                  job_status.status = "failed"
                  job_status.error_message = results.get("error", "Unknown error")
                  self.failed_jobs.append(job_status)
                  self.metrics.total_jobs_failed += 1
                  
                  self.logger.warning(f"❌ Job {job_id} failed: {job_status.error_message}")
               
               completed_job_ids.append(job_id)
               
         except Exception as e:
            self.logger.error(f"Error checking status for job {job_id}: {e}")
      
      # Remove completed jobs from active list
      for job_id in completed_job_ids:
         del self.active_jobs[job_id]
      
      return newly_completed
   
   def _calculate_batch_size(self) -> int:
      """Calculate optimal batch size for current iteration."""
      
      if self.iteration_count == 0:
         return self.initial_batch_size
      
      if not self.adaptive_scaling:
         return min(self.initial_batch_size, self.max_concurrent_jobs - len(self.active_jobs))
      
      # Adaptive scaling based on success rate and system performance
      current_load = len(self.active_jobs)
      available_slots = self.max_concurrent_jobs - current_load
      
      if available_slots <= 0:
         return 0
      
      # Scale based on recent success rate
      if self.metrics.success_rate > 0.8:
         # High success rate, can be more aggressive
         batch_size = min(available_slots, self.initial_batch_size + 2)
      elif self.metrics.success_rate > 0.6:
         # Moderate success rate, maintain current pace
         batch_size = min(available_slots, self.initial_batch_size)
      else:
         # Lower success rate, be more conservative
         batch_size = min(available_slots, max(1, self.initial_batch_size - 2))
      
      return batch_size
   
   def _build_agent_context(self) -> Dict[str, Any]:
      """Build context for hyperparameter agent."""
      
      # Convert job statuses to agent format
      completed_jobs_context = []
      for job_status in self.completed_jobs:
         if job_status.results and job_status.status == "completed":
            completed_jobs_context.append({
               "hyperparams": job_status.hyperparams,
               "results": job_status.results.get("training_results", {}),
               "job_id": job_status.job_id,
               "iteration": job_status.iteration
            })
      
      return {
         "iteration": self.iteration_count,
         "completed_jobs": completed_jobs_context,
         "experiment_id": self.experiment_id,
         "dataset": "cifar100",
         "objective": "maximize_auc"
      }
   
   def _build_job_config(self, job_id: str, hyperparams: Dict[str, Any]) -> Dict[str, Any]:
      """Build job configuration for training."""
      
      # Get Aurora job directory
      aurora_job_dirs = self.working_dir_manager.get_aurora_job_specific_dirs(
         job_id, self.experiment_id
      )
      
      job_config = {
         "job_id": job_id,
         "experiment_id": self.experiment_id,
         "output_dir": aurora_job_dirs["output_dir"],
         "data_dir": self.config["data"]["dir"],
         "repo_path": self.config["repository"]["aurora_path"],
         **hyperparams
      }
      
      return job_config
   
   def _update_optimization_state(self) -> None:
      """Update optimization state and convergence tracking."""
      
      if len(self.completed_jobs) == 0:
         return
      
      # Check for improvement in recent iterations
      recent_window = 10
      if len(self.completed_jobs) >= recent_window:
         recent_aucs = [job.results.get("training_results", {}).get("auc", 0.0) 
                       for job in self.completed_jobs[-recent_window:]]
         older_aucs = [job.results.get("training_results", {}).get("auc", 0.0) 
                      for job in self.completed_jobs[-2*recent_window:-recent_window]]
         
         if older_aucs and recent_aucs:
            recent_max = max(recent_aucs)
            older_max = max(older_aucs) if older_aucs else 0
            
            improvement = recent_max - older_max
            threshold = self.optimization_config.get("auc_optimization", {}).get("auc_improvement_threshold", 0.001)
            
            if improvement < threshold:
               self.convergence_counter += 1
            else:
               self.convergence_counter = 0
   
   def _check_convergence(self) -> bool:
      """Check if optimization has converged."""
      
      return self.convergence_counter >= self.convergence_patience
   
   def _start_monitoring(self) -> None:
      """Start monitoring threads for real-time updates."""
      
      self.monitoring_active = True
      
      # Start statistics monitoring thread
      self.monitoring_thread = threading.Thread(
         target=self._monitoring_loop,
         daemon=True
      )
      self.monitoring_thread.start()
      
      # Start plotting thread if enabled
      if self.monitoring_config.get("plots", {}).get("enabled", True):
         self.plot_thread = threading.Thread(
            target=self._plotting_loop,
            daemon=True
         )
         self.plot_thread.start()
   
   def _stop_monitoring(self) -> None:
      """Stop monitoring threads."""
      
      self.monitoring_active = False
      
      if self.monitoring_thread:
         self.monitoring_thread.join(timeout=5)
      
      if self.plot_thread:
         self.plot_thread.join(timeout=5)
   
   def _monitoring_loop(self) -> None:
      """Main monitoring loop for real-time statistics."""
      
      while self.monitoring_active:
         try:
            # Check job statuses
            self._check_job_statuses()
            
            # Print progress update
            self._print_progress_update()
            
            # Save checkpoint
            if self.monitoring_config.get("progress", {}).get("save_checkpoints", True):
               self._save_checkpoint()
            
            time.sleep(self.update_interval)
            
         except Exception as e:
            self.logger.error(f"Error in monitoring loop: {e}")
            time.sleep(self.update_interval)
   
   def _plotting_loop(self) -> None:
      """Generate and update plots periodically."""
      
      while self.monitoring_active:
         try:
            if len(self.completed_jobs) >= 3:  # Minimum jobs for meaningful plots
               self._generate_plots()
            
            time.sleep(self.plot_update_interval)
            
         except Exception as e:
            self.logger.error(f"Error in plotting loop: {e}")
            time.sleep(self.plot_update_interval)
   
   def _print_progress_update(self) -> None:
      """Print current progress statistics."""
      
      elapsed_time = datetime.now() - self.start_time
      
      # Basic statistics
      active_count = len(self.active_jobs)
      completed_count = len(self.completed_jobs)
      failed_count = len(self.failed_jobs)
      total_submitted = self.metrics.total_jobs_submitted
      
      # Performance statistics
      recent_auc_mean = self.metrics.recent_auc_mean
      recent_auc_std = self.metrics.recent_auc_std
      success_rate = self.metrics.success_rate
      
      # Progress summary
      progress_msg = f"""
📊 OPTIMIZATION PROGRESS UPDATE 📊
Time Elapsed: {elapsed_time}
Iteration: {self.iteration_count}/{self.max_iterations}

🎯 Job Statistics:
   Active: {active_count}
   Completed: {completed_count}
   Failed: {failed_count}
   Total Submitted: {total_submitted}
   Success Rate: {success_rate:.1%}

🏆 Performance:
   Best AUC: {self.metrics.best_auc:.6f}
   Recent AUC: {recent_auc_mean:.6f} ± {recent_auc_std:.6f}
   Convergence Counter: {self.convergence_counter}/{self.convergence_patience}
"""
      
      # Add exploration factor info
      if hasattr(self.optimization_agent, '_calculate_exploration_factor'):
         exploration_factor = self.optimization_agent._calculate_exploration_factor()
         progress_msg += f"   Exploration Factor: {exploration_factor:.2f}\n"
      
      self.logger.info(progress_msg)
   
   def _generate_plots(self) -> None:
      """Generate optimization progress plots."""
      
      try:
         plt.style.use('default')
         
         # Extract data for plotting
         iterations = [job.iteration for job in self.completed_jobs]
         aucs = [job.results.get("training_results", {}).get("auc", 0.0) for job in self.completed_jobs]
         timestamps = [job.completed_at for job in self.completed_jobs if job.completed_at]
         
         # 1. AUC Progress Plot
         fig, axes = plt.subplots(2, 2, figsize=(15, 12))
         fig.suptitle(f'AUC Optimization Progress - {self.experiment_id}', fontsize=16)
         
         # AUC over iterations
         ax1 = axes[0, 0]
         ax1.scatter(iterations, aucs, alpha=0.6, s=30)
         ax1.plot(iterations, np.cummax(aucs), 'r-', linewidth=2, label='Best AUC So Far')
         ax1.set_xlabel('Iteration')
         ax1.set_ylabel('AUC')
         ax1.set_title('AUC vs Iteration')
         ax1.legend()
         ax1.grid(True, alpha=0.3)
         
         # AUC over time
         ax2 = axes[0, 1]
         if timestamps:
            time_hours = [(ts - self.start_time).total_seconds() / 3600 for ts in timestamps]
            ax2.scatter(time_hours, aucs, alpha=0.6, s=30)
            ax2.set_xlabel('Time (hours)')
            ax2.set_ylabel('AUC')
            ax2.set_title('AUC vs Time')
            ax2.grid(True, alpha=0.3)
         
         # AUC distribution
         ax3 = axes[1, 0]
         ax3.hist(aucs, bins=20, alpha=0.7, edgecolor='black')
         ax3.axvline(np.mean(aucs), color='red', linestyle='--', label=f'Mean: {np.mean(aucs):.4f}')
         ax3.axvline(self.metrics.best_auc, color='green', linestyle='--', label=f'Best: {self.metrics.best_auc:.4f}')
         ax3.set_xlabel('AUC')
         ax3.set_ylabel('Frequency')
         ax3.set_title('AUC Distribution')
         ax3.legend()
         ax3.grid(True, alpha=0.3)
         
         # Success rate over time
         ax4 = axes[1, 1]
         batch_size = 10
         if len(self.completed_jobs) >= batch_size:
            batch_success_rates = []
            batch_indices = []
            for i in range(batch_size, len(self.completed_jobs) + 1, batch_size):
               batch_jobs = self.completed_jobs[i-batch_size:i]
               batch_successes = sum(1 for job in batch_jobs if job.status == "completed")
               batch_success_rates.append(batch_successes / batch_size)
               batch_indices.append(i)
            
            ax4.plot(batch_indices, batch_success_rates, 'o-')
            ax4.set_xlabel('Job Index')
            ax4.set_ylabel('Success Rate')
            ax4.set_title(f'Success Rate (per {batch_size} jobs)')
            ax4.grid(True, alpha=0.3)
         
         plt.tight_layout()
         plot_path = self.plots_dir / f"auc_progress_{datetime.now().strftime('%H%M%S')}.png"
         plt.savefig(plot_path, dpi=300, bbox_inches='tight')
         plt.close()
         
         # 2. Hyperparameter Correlation Plot
         if len(self.completed_jobs) >= 10:
            self._generate_hyperparameter_correlation_plot()
         
         # 3. Exploration Heatmap
         if len(self.completed_jobs) >= 15:
            self._generate_exploration_heatmap()
         
      except Exception as e:
         self.logger.error(f"Error generating plots: {e}")
   
   def _generate_hyperparameter_correlation_plot(self) -> None:
      """Generate hyperparameter correlation with AUC plot."""
      
      try:
         # Extract hyperparameters and AUCs
         param_data = defaultdict(list)
         aucs = []
         
         for job in self.completed_jobs:
            if job.status == "completed":
               auc = job.results.get("training_results", {}).get("auc", 0.0)
               aucs.append(auc)
               
               for param, value in job.hyperparams.items():
                  param_data[param].append(value)
         
         if len(aucs) < 10:
            return
         
         # Create correlation plot for numerical parameters
         numerical_params = {}
         for param, values in param_data.items():
            if all(isinstance(v, (int, float)) for v in values):
               numerical_params[param] = values
         
         if len(numerical_params) >= 2:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Create scatter plots for key parameters
            n_params = min(4, len(numerical_params))
            param_names = list(numerical_params.keys())[:n_params]
            
            for i, param in enumerate(param_names):
               ax.scatter(numerical_params[param], aucs, 
                         alpha=0.6, label=param, s=50)
            
            ax.set_ylabel('AUC')
            ax.set_title('Hyperparameter vs AUC Correlation')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plot_path = self.plots_dir / f"hyperparam_correlation_{datetime.now().strftime('%H%M%S')}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
      
      except Exception as e:
         self.logger.error(f"Error generating correlation plot: {e}")
   
   def _generate_exploration_heatmap(self) -> None:
      """Generate exploration heatmap for 2D parameter space."""
      
      try:
         # Focus on learning rate vs batch size for visualization
         lr_values = []
         batch_values = []
         auc_values = []
         
         for job in self.completed_jobs:
            if job.status == "completed":
               hyperparams = job.hyperparams
               if "learning_rate" in hyperparams and "batch_size" in hyperparams:
                  lr_values.append(np.log10(hyperparams["learning_rate"]))
                  batch_values.append(hyperparams["batch_size"])
                  auc_values.append(job.results.get("training_results", {}).get("auc", 0.0))
         
         if len(lr_values) < 10:
            return
         
         fig, ax = plt.subplots(figsize=(10, 8))
         
         scatter = ax.scatter(lr_values, batch_values, c=auc_values, 
                            cmap='viridis', s=100, alpha=0.7)
         
         ax.set_xlabel('Learning Rate (log10)')
         ax.set_ylabel('Batch Size')
         ax.set_title('Parameter Space Exploration\n(Color = AUC)')
         
         cbar = plt.colorbar(scatter)
         cbar.set_label('AUC')
         
         plot_path = self.plots_dir / f"exploration_heatmap_{datetime.now().strftime('%H%M%S')}.png"
         plt.savefig(plot_path, dpi=300, bbox_inches='tight')
         plt.close()
         
      except Exception as e:
         self.logger.error(f"Error generating exploration heatmap: {e}")
   
   def _save_checkpoint(self) -> None:
      """Save current optimization state for potential resumption."""
      
      try:
         checkpoint_data = {
            "experiment_id": self.experiment_id,
            "iteration_count": self.iteration_count,
            "start_time": self.start_time.isoformat(),
            "best_auc": self.metrics.best_auc,
            "best_hyperparams": self.metrics.best_hyperparams,
            "completed_jobs": [
               {
                  "job_id": job.job_id,
                  "hyperparams": job.hyperparams,
                  "iteration": job.iteration,
                  "results": job.results,
                  "status": job.status
               } for job in self.completed_jobs
            ],
            "convergence_counter": self.convergence_counter
         }
         
         checkpoint_path = self.results_dir / "checkpoint.json"
         with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f, indent=3, default=str)
         
      except Exception as e:
         self.logger.error(f"Error saving checkpoint: {e}")
   
   def _generate_final_report(self) -> Dict[str, Any]:
      """Generate comprehensive final optimization report."""
      
      total_time = datetime.now() - self.start_time
      
      # Calculate final statistics
      final_aucs = [job.results.get("training_results", {}).get("auc", 0.0) 
                   for job in self.completed_jobs]
      
      report = {
         "experiment_id": self.experiment_id,
         "status": "completed",
         "total_time_seconds": total_time.total_seconds(),
         "total_iterations": self.iteration_count,
         
         # Job statistics
         "job_statistics": {
            "total_submitted": self.metrics.total_jobs_submitted,
            "total_completed": self.metrics.total_jobs_completed,
            "total_failed": self.metrics.total_jobs_failed,
            "success_rate": self.metrics.success_rate
         },
         
         # Performance results
         "performance": {
            "best_auc": self.metrics.best_auc,
            "best_hyperparams": self.metrics.best_hyperparams,
            "mean_auc": float(np.mean(final_aucs)) if final_aucs else 0.0,
            "std_auc": float(np.std(final_aucs)) if final_aucs else 0.0,
            "median_auc": float(np.median(final_aucs)) if final_aucs else 0.0,
            "auc_improvement": self.metrics.best_auc - (final_aucs[0] if final_aucs else 0.0)
         },
         
         # Optimization details
         "optimization_details": {
            "convergence_achieved": self.convergence_counter >= self.convergence_patience,
            "convergence_counter": self.convergence_counter,
            "max_concurrent_jobs": self.max_concurrent_jobs,
            "adaptive_scaling_used": self.adaptive_scaling
         },
         
         # All results
         "all_results": [
            {
               "job_id": job.job_id,
               "iteration": job.iteration,
               "hyperparams": job.hyperparams,
               "auc": job.results.get("training_results", {}).get("auc", 0.0) if job.results else 0.0,
               "accuracy": job.results.get("training_results", {}).get("accuracy", 0.0) if job.results else 0.0,
               "training_time": job.results.get("training_results", {}).get("training_time", 0.0) if job.results else 0.0,
               "status": job.status
            } for job in self.completed_jobs + self.failed_jobs
         ]
      }
      
      # Save report
      report_path = self.results_dir / "final_report.json"
      with open(report_path, 'w') as f:
         json.dump(report, f, indent=3, default=str)
      
      self.logger.info(f"Final report saved to: {report_path}")
      
      return report
   
   def _handle_interruption(self) -> Dict[str, Any]:
      """Handle user interruption gracefully."""
      
      self.logger.info("Handling user interruption...")
      
      # Cancel active jobs
      for job_id in list(self.active_jobs.keys()):
         try:
            self.job_scheduler.cancel_job(job_id)
            self.logger.info(f"Cancelled job {job_id}")
         except Exception as e:
            self.logger.error(f"Failed to cancel job {job_id}: {e}")
      
      # Generate partial report
      return self._generate_final_report()
   
   def _cleanup(self) -> None:
      """Cleanup resources and temporary files."""
      
      try:
         # Final plot generation
         if len(self.completed_jobs) > 0:
            self._generate_plots()
         
         self.logger.info("Cleanup completed")
         
      except Exception as e:
         self.logger.error(f"Error during cleanup: {e}")
   
   def get_current_status(self) -> Dict[str, Any]:
      """Get current optimization status for external monitoring."""
      
      return {
         "experiment_id": self.experiment_id,
         "iteration": self.iteration_count,
         "elapsed_time": (datetime.now() - self.start_time).total_seconds(),
         "active_jobs": len(self.active_jobs),
         "completed_jobs": len(self.completed_jobs),
         "best_auc": self.metrics.best_auc,
         "success_rate": self.metrics.success_rate,
         "convergence_counter": self.convergence_counter
      } 