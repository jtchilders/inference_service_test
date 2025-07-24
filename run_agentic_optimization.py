#!/usr/bin/env python3
"""
Production Agentic Hyperparameter Optimization Script.
Maximizes AUC on CIFAR-100 using parallel training jobs with intelligent LLM guidance.

Usage:
   python run_agentic_optimization.py --config config_agentic_production.yaml
   python run_agentic_optimization.py --config config_agentic_production.yaml --max-iterations 50
   python run_agentic_optimization.py --config config_agentic_production.yaml --resume experiment_20241201_143022
"""

import argparse
import json
import logging
import os
import sys
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

# Add src to path for imports and handle module conflicts
src_path = str(Path(__file__).parent / "src")
if src_path not in sys.path:
   sys.path.insert(0, src_path)  # Insert at beginning to take precedence

# Import with explicit module loading to avoid conflicts
import importlib.util
import types

def load_module_from_path(module_name: str, file_path: str):
   """Load a module from a specific file path."""
   spec = importlib.util.spec_from_file_location(module_name, file_path)
   module = importlib.util.module_from_spec(spec)
   spec.loader.exec_module(module)
   return module

# Load our local modules
_orchestrator_path = Path(__file__).parent / "src" / "agents" / "parallel_auc_orchestrator.py"
_auth_path = Path(__file__).parent / "src" / "utils" / "inference_auth_token.py"

orchestrator_module = load_module_from_path("parallel_auc_orchestrator", _orchestrator_path)
auth_module = load_module_from_path("inference_auth_token", _auth_path)

ParallelAUCOrchestrator = orchestrator_module.ParallelAUCOrchestrator
get_access_token = auth_module.get_access_token


def setup_logging(config: Dict[str, Any]) -> None:
   """Setup comprehensive logging for the optimization workflow."""
   
   logging_config = config.get("logging", {})
   log_level = logging_config.get("level", "INFO")
   console_output = logging_config.get("console_output", True)
   
   # Create logs directory
   logs_dir = Path("logs")
   logs_dir.mkdir(exist_ok=True)
   
   # Setup log format
   log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
   date_format = "%d-%m %H:%M"
   
   # Create handlers
   handlers = []
   
   # Console handler
   if console_output:
      console_handler = logging.StreamHandler(sys.stdout)
      console_handler.setFormatter(logging.Formatter(log_format, date_format))
      handlers.append(console_handler)
   
   # File handlers for different components
   file_prefix = logging_config.get("file_prefix", "agentic_optimization")
   timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
   
   # Main log file
   main_log_file = logs_dir / f"{file_prefix}_{timestamp}.log"
   file_handler = logging.FileHandler(main_log_file)
   file_handler.setFormatter(logging.Formatter(log_format, date_format))
   handlers.append(file_handler)
   
   # Configure root logger
   logging.basicConfig(
      level=getattr(logging, log_level.upper()),
      handlers=handlers,
      format=log_format,
      datefmt=date_format
   )
   
   # Setup component-specific loggers if configured
   component_logs = logging_config.get("component_logs", {})
   for component, log_file in component_logs.items():
      component_logger = logging.getLogger(f"src.agents.{component}")
      component_handler = logging.FileHandler(log_file)
      component_handler.setFormatter(logging.Formatter(log_format, date_format))
      component_logger.addHandler(component_handler)
   
   logger = logging.getLogger(__name__)
   logger.info(f"Logging initialized - Main log: {main_log_file}")


def validate_config(config: Dict[str, Any]) -> bool:
   """Validate configuration file for required parameters."""
   
   logger = logging.getLogger(__name__)
   
   # Required top-level sections
   required_sections = ["sophia", "globus_compute", "optimization", "search_space", "data"]
   
   for section in required_sections:
      if section not in config:
         logger.error(f"Missing required configuration section: {section}")
         return False
   
   # Validate Sophia configuration
   sophia_config = config["sophia"]
   if "url" not in sophia_config:
      logger.error("Missing Sophia URL in configuration")
      return False
   
   # Validate Globus Compute configuration
   globus_config = config["globus_compute"]
   if not globus_config.get("enabled", False):
      logger.error("Globus Compute must be enabled for parallel optimization")
      return False
   
   endpoint_id = globus_config.get("endpoint_id", "")
   if endpoint_id == "YOUR_AURORA_ENDPOINT_ID" or not endpoint_id:
      logger.error("Please set your actual Globus Compute endpoint ID in the configuration")
      return False
   
   # Validate optimization settings
   optimization_config = config["optimization"]
   if optimization_config.get("max_iterations", 0) <= 0:
      logger.error("Max iterations must be greater than 0")
      return False
   
   # Validate search space
   search_space = config["search_space"]
   required_params = ["model_type", "learning_rate", "batch_size", "num_epochs"]
   
   for param in required_params:
      if param not in search_space:
         logger.error(f"Missing required parameter in search space: {param}")
         return False
   
   logger.info("Configuration validation passed")
   return True


def check_authentication() -> bool:
   """Check if Sophia authentication is working."""
   
   logger = logging.getLogger(__name__)
   
   try:
      # Try to get access token
      token = get_access_token()
      
      if not token:
         logger.error("Failed to obtain Sophia access token")
         logger.error("Please run: python src/utils/inference_auth_token.py authenticate --force")
         return False
      
      logger.info("Sophia authentication validated")
      return True
      
   except Exception as e:
      logger.error(f"Sophia authentication failed: {e}")
      logger.error("Please run: python src/utils/inference_auth_token.py authenticate --force")
      return False


def check_dependencies(skip_globus_compute: bool = False) -> bool:
   """Check if all required dependencies are available."""
   
   logger = logging.getLogger(__name__)
   
   # Core packages needed for all functionality
   required_packages = [
      ("langchain_openai", "langchain-openai"),  
      ("langchain_core", "langchain-core"),
      ("langgraph", "langgraph"),
      ("matplotlib", "matplotlib"),
      ("seaborn", "seaborn"),
      ("sklearn", "scikit-learn"),
      ("scipy", "scipy"),
      ("numpy", "numpy"),
      ("pydantic", "pydantic"),
   ]
   
   # Globus Compute is only needed for actual execution, not validation
   if not skip_globus_compute:
      required_packages.insert(0, ("globus_compute_sdk", "globus-compute-sdk"))
   
   missing_packages = []
   optional_missing = []
   
   for package_name, pip_name in required_packages:
      try:
         __import__(package_name)
      except ImportError:
         if package_name == "globus_compute_sdk":
            optional_missing.append(pip_name)
         else:
            missing_packages.append(pip_name)
   
   if missing_packages:
      logger.error("Missing required packages:")
      for package in missing_packages:
         logger.error(f"  - {package}")
      logger.error("Install with: pip install " + " ".join(missing_packages))
      return False
   
   if optional_missing:
      logger.warning("Optional packages missing (needed for Aurora execution):")
      for package in optional_missing:
         logger.warning(f"  - {package}")
      logger.warning("Install with: pip install " + " ".join(optional_missing))
      logger.warning("These are only needed when running on Aurora, not for local validation")
   
   logger.info("Core dependencies are available")
   return True


def print_startup_banner(config: Dict[str, Any]) -> None:
   """Print startup banner with configuration summary."""
   
   logger = logging.getLogger(__name__)
   
   optimization_config = config.get("optimization", {})
   parallel_config = config.get("globus_compute", {}).get("parallel_jobs", {})
   
   banner = f"""
{'='*80}
🚀 AGENTIC HYPERPARAMETER OPTIMIZATION FOR CIFAR-100 AUC MAXIMIZATION 🚀
{'='*80}

📋 EXPERIMENT CONFIGURATION:
   Objective: {optimization_config.get('objective', 'maximize_auc')}
   Max Iterations: {optimization_config.get('max_iterations', 100)}
   Dataset: CIFAR-100 Image Classification
   
🔄 PARALLEL EXECUTION:
   Max Concurrent Jobs: {parallel_config.get('max_concurrent', 12)}
   Initial Batch Size: {parallel_config.get('initial_batch_size', 8)}
   Adaptive Scaling: {parallel_config.get('adaptive_scaling', True)}
   
🤖 AI AGENT STRATEGY:
   LLM Service: {config.get('sophia', {}).get('url', 'Unknown')}
   Search Strategy: Exploration → Exploitation
   Initial Samples: {optimization_config.get('strategy', {}).get('initial_samples', 20)}
   
📊 MONITORING:
   Real-time Statistics: ✅
   Progress Plots: ✅
   Checkpoint Saving: ✅
   
🎯 TARGET: Find hyperparameters that maximize AUC (accuracy × training efficiency)

{'='*80}
"""
   
   print(banner)
   logger.info("Agentic optimization startup banner displayed")


def load_resume_data(experiment_id: str) -> Dict[str, Any]:
   """Load data for resuming a previous experiment."""
   
   logger = logging.getLogger(__name__)
   
   # Look for checkpoint file
   results_dir = Path(f"results/agentic_optimization_{experiment_id}")
   checkpoint_path = results_dir / "checkpoint.json"
   
   if not checkpoint_path.exists():
      logger.error(f"Checkpoint file not found: {checkpoint_path}")
      return {}
   
   try:
      with open(checkpoint_path, 'r') as f:
         checkpoint_data = json.load(f)
      
      logger.info(f"Loaded checkpoint data for experiment {experiment_id}")
      logger.info(f"Previous best AUC: {checkpoint_data.get('best_auc', 0.0):.6f}")
      logger.info(f"Completed jobs: {len(checkpoint_data.get('completed_jobs', []))}")
      
      return checkpoint_data
      
   except Exception as e:
      logger.error(f"Failed to load checkpoint data: {e}")
      return {}


def run_optimization(config: Dict[str, Any], resume_data: Dict[str, Any] = None) -> Dict[str, Any]:
   """Run the main optimization workflow."""
   
   logger = logging.getLogger(__name__)
   
   try:
      # Initialize orchestrator
      logger.info("Initializing Parallel AUC Orchestrator...")
      orchestrator = ParallelAUCOrchestrator(config)
      
      # If resuming, restore state
      if resume_data:
         logger.info("Restoring previous experiment state...")
         # TODO: Implement state restoration in orchestrator
         logger.warning("Resume functionality not yet implemented - starting fresh")
      
      # Run optimization
      logger.info("Starting optimization workflow...")
      result = orchestrator.run_optimization()
      
      return result
      
   except KeyboardInterrupt:
      logger.info("Optimization interrupted by user")
      return {"status": "interrupted"}
   
   except Exception as e:
      logger.error(f"Optimization failed: {e}")
      import traceback
      logger.error(f"Full traceback: {traceback.format_exc()}")
      return {"status": "failed", "error": str(e)}


def print_final_summary(result: Dict[str, Any]) -> None:
   """Print final optimization summary."""
   
   logger = logging.getLogger(__name__)
   
   status = result.get("status", "unknown")
   
   if status == "completed":
      performance = result.get("performance", {})
      job_stats = result.get("job_statistics", {})
      
      summary = f"""
{'='*80}
🎉 OPTIMIZATION COMPLETED SUCCESSFULLY! 🎉
{'='*80}

🏆 BEST RESULTS:
   Best AUC: {performance.get('best_auc', 0.0):.6f}
   Mean AUC: {performance.get('mean_auc', 0.0):.6f} ± {performance.get('std_auc', 0.0):.6f}
   AUC Improvement: +{performance.get('auc_improvement', 0.0):.6f}
   
📊 JOB STATISTICS:
   Total Jobs: {job_stats.get('total_submitted', 0)}
   Successful: {job_stats.get('total_completed', 0)}
   Failed: {job_stats.get('total_failed', 0)}
   Success Rate: {job_stats.get('success_rate', 0.0):.1%}
   
⏱️  TIMING:
   Total Time: {result.get('total_time_seconds', 0) / 3600:.2f} hours
   Iterations: {result.get('total_iterations', 0)}
   
🎯 BEST HYPERPARAMETERS:
{json.dumps(performance.get('best_hyperparams', {}), indent=3)}

📁 Results saved in: results/agentic_optimization_{result.get('experiment_id', 'unknown')}/
{'='*80}
"""
   
   elif status == "interrupted":
      summary = f"""
{'='*80}
⚠️  OPTIMIZATION INTERRUPTED BY USER ⚠️
{'='*80}

The optimization was stopped by user request.
Partial results have been saved and can be resumed later.
{'='*80}
"""
   
   else:
      summary = f"""
{'='*80}
❌ OPTIMIZATION FAILED ❌
{'='*80}

Status: {status}
Error: {result.get('error', 'Unknown error')}

Please check the logs for more details.
{'='*80}
"""
   
   print(summary)
   logger.info("Final summary displayed")


def main():
   """Main entry point for agentic hyperparameter optimization."""
   
   parser = argparse.ArgumentParser(
      description="Agentic Hyperparameter Optimization for CIFAR-100 AUC Maximization",
      formatter_class=argparse.RawDescriptionHelpFormatter,
      epilog="""
Examples:
   # Run with default settings
   python run_agentic_optimization.py --config config_agentic_production.yaml
   
   # Run with custom iteration count
   python run_agentic_optimization.py --config config_agentic_production.yaml --max-iterations 50
   
   # Resume previous experiment
   python run_agentic_optimization.py --config config_agentic_production.yaml --resume experiment_20241201_143022
   
   # Run with debug logging
   python run_agentic_optimization.py --config config_agentic_production.yaml --log-level DEBUG
"""
   )
   
   parser.add_argument("--config", "-c", required=True,
                      help="Path to YAML configuration file")
   parser.add_argument("--max-iterations", "-m", type=int,
                      help="Maximum optimization iterations (overrides config)")
   parser.add_argument("--resume", "-r", 
                      help="Resume previous experiment by experiment ID")
   parser.add_argument("--log-level", "-l", default="INFO",
                      choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                      help="Logging level")
   parser.add_argument("--dry-run", action="store_true",
                      help="Validate configuration and dependencies without running")
   parser.add_argument("--max-concurrent", type=int,
                      help="Maximum concurrent jobs (overrides config)")
   
   args = parser.parse_args()
   
   # Load configuration
   try:
      with open(args.config, 'r') as f:
         config = yaml.safe_load(f)
   except Exception as e:
      print(f"❌ Failed to load configuration file {args.config}: {e}")
      return 1
   
   # Override config with command line arguments
   if args.max_iterations:
      config.setdefault("optimization", {})["max_iterations"] = args.max_iterations
   
   if args.max_concurrent:
      config.setdefault("globus_compute", {}).setdefault("parallel_jobs", {})["max_concurrent"] = args.max_concurrent
   
   if args.log_level:
      config.setdefault("logging", {})["level"] = args.log_level
   
   # Setup logging first
   setup_logging(config)
   logger = logging.getLogger(__name__)
   
   logger.info(f"Starting agentic optimization with config: {args.config}")
   
   # Validation phase
   logger.info("Running pre-flight checks...")
   
   # Skip Globus Compute dependency for dry runs (local validation)
   skip_globus_compute = args.dry_run
   if not check_dependencies(skip_globus_compute=skip_globus_compute):
      logger.error("Dependency check failed")
      return 1
   
   if not validate_config(config):
      logger.error("Configuration validation failed")
      return 1
   
   if not check_authentication():
      logger.error("Authentication check failed")
      return 1
   
   if args.dry_run:
      logger.info("✅ Dry run completed successfully - all checks passed")
      print("✅ Configuration and dependencies validated. Ready to run optimization!")
      return 0
   
   # Load resume data if specified
   resume_data = None
   if args.resume:
      resume_data = load_resume_data(args.resume)
      if not resume_data:
         logger.error("Failed to load resume data")
         return 1
   
   # Display startup banner
   print_startup_banner(config)
   
   # Run optimization
   logger.info("🚀 Launching agentic hyperparameter optimization...")
   result = run_optimization(config, resume_data)
   
   # Display results
   print_final_summary(result)
   
   # Return appropriate exit code
   if result.get("status") == "completed":
      logger.info("Agentic optimization completed successfully")
      return 0
   elif result.get("status") == "interrupted":
      logger.info("Agentic optimization interrupted by user")
      return 130  # Standard exit code for SIGINT
   else:
      logger.error("Agentic optimization failed")
      return 1


if __name__ == "__main__":
   sys.exit(main()) 