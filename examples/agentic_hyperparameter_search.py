#!/usr/bin/env python3
"""
Simple Example: Agentic Hyperparameter Search for CIFAR-100 AUC Optimization.

This example demonstrates how to set up and run the intelligent hyperparameter
optimization workflow that uses AI agents to find the best CIFAR-100 training
parameters for maximum AUC.

Usage:
   python examples/agentic_hyperparameter_search.py --endpoint YOUR_GLOBUS_ENDPOINT_ID
   python examples/agentic_hyperparameter_search.py --endpoint YOUR_GLOBUS_ENDPOINT_ID --max-iterations 20
   python examples/agentic_hyperparameter_search.py --endpoint YOUR_GLOBUS_ENDPOINT_ID --quick-test
"""

import argparse
import logging
import sys
import yaml
from pathlib import Path
from datetime import datetime

# Add project path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import the main optimization script functionality
from run_agentic_optimization import (
    setup_logging, validate_config, check_authentication, 
    check_dependencies, run_optimization, print_final_summary,
    print_startup_banner
)


def create_simple_config(endpoint_id: str, max_iterations: int = 20, quick_test: bool = False) -> dict:
   """
   Create a simplified configuration for easy experimentation.
   
   Args:
      endpoint_id: Your Globus Compute endpoint ID for Aurora
      max_iterations: Number of optimization iterations to run
      quick_test: If True, use minimal settings for fast testing
      
   Returns:
      Configuration dictionary
   """
   
   # Base configuration optimized for quick results
   config = {
      # Sophia LLM service
      "sophia": {
         "url": "https://data-portal-dev.cels.anl.gov/resource_server/sophia/vllm/v1",
         "model": "meta-llama/Llama-3.3-70B-Instruct",
         "temperature": 0.1
      },
      
      # Globus Compute for Aurora
      "globus_compute": {
         "enabled": True,
         "endpoint_id": endpoint_id,
         "auth_method": "native_client",
         "function_timeout": 3600 if quick_test else 7200,
         "max_retries": 2,
         "parallel_jobs": {
            "max_concurrent": 4 if quick_test else 8,
            "initial_batch_size": 2 if quick_test else 4,
            "adaptive_scaling": True,
            "job_spacing_seconds": 3
         }
      },
      
      # Working directories
      "working_dirs": {
         "aurora_base": "/lus/flare/projects/datascience/parton/workflows",
         "local_base": "results",
         "launch_iteration_pattern": "simple_agentic_{timestamp}_{experiment_id}",
         "job_pattern": "job_{job_id}"
      },
      
      # Data paths
      "data": {
         "dir": "/lus/flare/projects/datascience/parton/data",
         "dataset": "cifar100"
      },
      
      # Repository on Aurora
      "repository": {
         "aurora_path": "/lus/flare/projects/datascience/parton/inference_service_test"
      },
      
      # Optimization settings
      "optimization": {
         "objective": "maximize_auc",
         "max_iterations": max_iterations,
         "convergence_patience": 8 if quick_test else 15,
         "strategy": {
            "initial_phase": "broad_sampling",
            "initial_samples": 5 if quick_test else 10,
            "exploration_factor": 0.7,
            "exploitation_ramp": 0.1 if quick_test else 0.05,
            "min_exploration": 0.3 if quick_test else 0.2
         },
         "auc_optimization": {
            "min_epochs": 3 if quick_test else 5,
            "max_epochs": 20 if quick_test else 50,
            "early_stopping_patience": 5 if quick_test else 10,
            "auc_improvement_threshold": 0.002
         }
      },
      
      # Search space (focused for quick results)
      "search_space": {
         "model_type": {
            "values": ["resnet18", "resnet34"] if quick_test else ["resnet18", "resnet34", "resnet50", "vgg16"],
            "weights": [0.6, 0.4] if quick_test else [0.4, 0.3, 0.2, 0.1]
         },
         "learning_rate": {
            "type": "log_uniform",
            "min": 1e-4,
            "max": 1e-2,
            "initial_suggestions": [5e-4, 1e-3, 2e-3, 5e-3]
         },
         "batch_size": {
            "values": [64, 128] if quick_test else [32, 64, 128, 256],
            "weights": [0.4, 0.6] if quick_test else [0.1, 0.3, 0.4, 0.2]
         },
         "num_epochs": {
            "type": "int_uniform",
            "min": 3 if quick_test else 5,
            "max": 20 if quick_test else 50,
            "adaptive": True
         },
         "dropout_rate": {
            "type": "uniform",
            "min": 0.1,
            "max": 0.4,
            "initial_suggestions": [0.1, 0.2, 0.3, 0.4]
         },
         "weight_decay": {
            "type": "log_uniform",
            "min": 1e-5,
            "max": 1e-3,
            "initial_suggestions": [1e-5, 1e-4, 1e-3]
         },
         "hidden_size": {
            "values": [512, 1024] if quick_test else [512, 1024, 2048],
            "weights": [0.6, 0.4] if quick_test else [0.3, 0.5, 0.2]
         },
         "num_layers": {
            "values": [2, 3] if quick_test else [2, 3, 4],
            "weights": [0.4, 0.6] if quick_test else [0.3, 0.5, 0.2]
         }
      },
      
      # Monitoring (simplified for examples)
      "monitoring": {
         "update_interval_seconds": 20,
         "plot_update_interval_seconds": 60,
         "statistics": {
            "window_size": 10,
            "track_metrics": ["auc", "accuracy", "training_time"]
         },
         "plots": {
            "enabled": True,
            "output_dir": "results/plots",
            "formats": ["png"],
            "auc_progression": True,
            "hyperparameter_correlation": True,
            "exploration_heatmap": True
         },
         "progress": {
            "log_frequency": "every_job",
            "detailed_report_frequency": 5,
            "save_checkpoints": True
         }
      },
      
      # Logging
      "logging": {
         "level": "INFO",
         "file_prefix": "simple_agentic_optimization",
         "console_output": True,
         "detailed_errors": True
      },
      
      # Performance settings
      "performance": {
         "job_submission_rate_limit": 1,
         "memory_usage_limit_gb": 8,
         "disk_usage_limit_gb": 50
      },
      
      # Experiment metadata
      "experiment": {
         "name": f"Simple_CIFAR100_AUC_{'QuickTest' if quick_test else 'Standard'}",
         "description": "Simple agentic hyperparameter optimization example for CIFAR-100",
         "tags": ["example", "cifar100", "auc_optimization", "simple"]
      }
   }
   
   return config


def print_example_banner(quick_test: bool = False):
   """Print example-specific banner."""
   
   mode = "QUICK TEST" if quick_test else "STANDARD"
   
   banner = f"""
{'='*80}
🎯 SIMPLE AGENTIC HYPERPARAMETER SEARCH EXAMPLE ({mode}) 🎯
{'='*80}

This example demonstrates intelligent hyperparameter optimization using:
• 🤖 AI agents (LLM) for smart parameter suggestions  
• 🚀 Parallel job execution on Aurora via Globus Compute
• 📊 Real-time monitoring with AUC tracking
• 🎯 Focus on maximizing AUC for CIFAR-100 classification

{'QUICK TEST MODE: Fast settings for demonstration' if quick_test else 'STANDARD MODE: Full optimization capabilities'}

The AI agent will:
1. Start with broad parameter exploration
2. Learn from training results 
3. Focus on promising parameter regions
4. Progressively exploit high-performing areas

{'='*80}
"""
   
   print(banner)


def main():
   """Main function for the simple agentic hyperparameter search example."""
   
   parser = argparse.ArgumentParser(
      description="Simple Agentic Hyperparameter Search Example",
      formatter_class=argparse.RawDescriptionHelpFormatter,
      epilog="""
This example demonstrates the intelligent hyperparameter optimization workflow.
The AI agent will automatically explore parameter space and focus on regions
that produce higher AUC scores.

Examples:
   # Basic usage
   python examples/agentic_hyperparameter_search.py --endpoint YOUR_ENDPOINT_ID
   
   # Quick test (fast settings)
   python examples/agentic_hyperparameter_search.py --endpoint YOUR_ENDPOINT_ID --quick-test
   
   # Custom iteration count
   python examples/agentic_hyperparameter_search.py --endpoint YOUR_ENDPOINT_ID --max-iterations 30
   
   # Debug mode
   python examples/agentic_hyperparameter_search.py --endpoint YOUR_ENDPOINT_ID --debug
"""
   )
   
   parser.add_argument("--endpoint", "-e", required=True,
                      help="Globus Compute endpoint ID for Aurora")
   parser.add_argument("--max-iterations", "-m", type=int, default=20,
                      help="Maximum optimization iterations (default: 20)")
   parser.add_argument("--quick-test", "-q", action="store_true",
                      help="Use fast settings for quick demonstration")
   parser.add_argument("--debug", "-d", action="store_true",
                      help="Enable debug logging")
   parser.add_argument("--dry-run", action="store_true",
                      help="Validate setup without running optimization")
   
   args = parser.parse_args()
   
   # Create configuration
   config = create_simple_config(
      endpoint_id=args.endpoint,
      max_iterations=args.max_iterations,
      quick_test=args.quick_test
   )
   
   # Set debug logging if requested
   if args.debug:
      config["logging"]["level"] = "DEBUG"
   
   # Setup logging
   setup_logging(config)
   logger = logging.getLogger(__name__)
   
   logger.info(f"Starting simple agentic hyperparameter search example")
   logger.info(f"Endpoint ID: {args.endpoint}")
   logger.info(f"Max iterations: {args.max_iterations}")
   logger.info(f"Quick test mode: {args.quick_test}")
   
   # Print example banner
   print_example_banner(args.quick_test)
   
   # Run validation checks
   logger.info("🔍 Running pre-flight checks...")
   
   if not check_dependencies():
      logger.error("❌ Dependency check failed")
      print("\n❌ Missing required packages. Please install them and try again.")
      return 1
   
   if not validate_config(config):
      logger.error("❌ Configuration validation failed")
      print("\n❌ Configuration validation failed. Please check your settings.")
      return 1
   
   if not check_authentication():
      logger.error("❌ Authentication check failed")
      print("\n❌ Sophia authentication failed. Please authenticate first:")
      print("   python src/utils/inference_auth_token.py authenticate --force")
      return 1
   
   print("✅ All pre-flight checks passed!")
   
   if args.dry_run:
      print("\n🎉 Dry run completed successfully. Ready to run optimization!")
      return 0
   
   print(f"\n🚀 Starting {'quick test' if args.quick_test else 'standard'} optimization...")
   print(f"📈 Target: Find hyperparameters that maximize CIFAR-100 AUC")
   print(f"🔄 Max iterations: {args.max_iterations}")
   print(f"⚡ The AI agent will learn and adapt as it receives training results")
   
   try:
      # Run optimization
      result = run_optimization(config)
      
      # Display results
      print_final_summary(result)
      
      # Additional example-specific summary
      if result.get("status") == "completed":
         print(f"""
🎓 EXAMPLE COMPLETED SUCCESSFULLY! 

Key Takeaways:
• The AI agent {'explored quickly' if args.quick_test else 'thoroughly explored'} the hyperparameter space
• Best AUC achieved: {result.get('performance', {}).get('best_auc', 0.0):.6f}
• Total training jobs: {result.get('job_statistics', {}).get('total_submitted', 0)}
• Success rate: {result.get('job_statistics', {}).get('success_rate', 0.0):.1%}

🔍 Check the results directory for:
• Training plots and visualizations
• Detailed experiment logs  
• Best hyperparameter configurations
• Complete job history

🚀 Next Steps:
• Try running with more iterations for better results
• Experiment with different parameter ranges
• Use the best hyperparameters for your own training
""")
      
      return 0 if result.get("status") == "completed" else 1
      
   except KeyboardInterrupt:
      print("\n⚠️  Example interrupted by user. Partial results saved.")
      return 130
   
   except Exception as e:
      logger.error(f"Example failed: {e}")
      print(f"\n❌ Example failed: {e}")
      print("Please check the logs for more details.")
      return 1


if __name__ == "__main__":
   sys.exit(main()) 