#!/usr/bin/env python3
"""
Example script demonstrating the use of LangGraph-based agentic workflows.
Shows how to use both basic and advanced LangGraph workflows for hyperparameter optimization.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from agents.langgraph_workflow import LangGraphWorkflow
from agents.advanced_langgraph_workflow import AdvancedLangGraphWorkflow
from agents.langchain_hyperparam_agent import LangChainHyperparameterAgent


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


def run_basic_workflow_example(config: dict, max_iterations: int = 5):
   """Run the basic LangGraph workflow example."""
   print("\n" + "="*60)
   print("RUNNING BASIC LANGGRAPH WORKFLOW EXAMPLE")
   print("="*60)
   
   # Create workflow
   workflow = LangGraphWorkflow(config)
   
   # Run experiment
   result = workflow.run_experiment(max_iterations=max_iterations)
   
   print(f"\nBasic workflow completed!")
   print(f"Result: {result}")
   
   return result


def run_advanced_workflow_example(config: dict, max_iterations: int = 5):
   """Run the advanced LangGraph workflow example."""
   print("\n" + "="*60)
   print("RUNNING ADVANCED LANGGRAPH WORKFLOW EXAMPLE")
   print("="*60)
   
   # Create workflow
   workflow = AdvancedLangGraphWorkflow(config)
   
   # Run experiment
   result = workflow.run_experiment(max_iterations=max_iterations)
   
   print(f"\nAdvanced workflow completed!")
   print(f"Result: {result}")
   
   return result


def run_hyperparam_agent_example(config: dict):
   """Run a standalone hyperparameter agent example."""
   print("\n" + "="*60)
   print("RUNNING HYPERPARAMETER AGENT EXAMPLE")
   print("="*60)
   
   # Create agent
   agent = LangChainHyperparameterAgent(
      sophia_url=config["sophia"]["url"]
   )
   
   # Example context
   context = {
      "iteration": 1,
      "completed_jobs": [
         {
            "hyperparams": {
               "model_type": "resnet18",
               "learning_rate": 0.001,
               "batch_size": 128,
               "num_epochs": 50,
               "hidden_size": 1024,
               "num_layers": 3,
               "dropout_rate": 0.2,
               "weight_decay": 1e-4
            },
            "results": {
               "final_accuracy": 0.75,
               "auc": 0.82,
               "training_time": 120.5
            }
         },
         {
            "hyperparams": {
               "model_type": "resnet34",
               "learning_rate": 0.0005,
               "batch_size": 64,
               "num_epochs": 75,
               "hidden_size": 2048,
               "num_layers": 4,
               "dropout_rate": 0.3,
               "weight_decay": 5e-5
            },
            "results": {
               "final_accuracy": 0.78,
               "auc": 0.85,
               "training_time": 180.2
            }
         }
      ],
      "experiment_id": "example_experiment",
      "dataset": "cifar100",
      "objective": "maximize_accuracy"
   }
   
   # Get hyperparameter suggestions
   suggestions = agent.suggest_hyperparameters(context)
   
   print(f"\nHyperparameter suggestions:")
   print(f"Model type: {suggestions['model_type']}")
   print(f"Learning rate: {suggestions['learning_rate']}")
   print(f"Batch size: {suggestions['batch_size']}")
   print(f"Number of epochs: {suggestions['num_epochs']}")
   print(f"Hidden size: {suggestions['hidden_size']}")
   print(f"Number of layers: {suggestions['num_layers']}")
   print(f"Dropout rate: {suggestions['dropout_rate']}")
   print(f"Weight decay: {suggestions['weight_decay']}")
   
   return suggestions


def main():
   """Main entry point for the example script."""
   parser = argparse.ArgumentParser(description="LangGraph workflow examples")
   parser.add_argument("--config", "-c", required=True, 
                      help="Path to configuration file")
   parser.add_argument("--workflow", "-w", choices=["basic", "advanced", "agent", "all"], 
                      default="all", help="Which workflow to run")
   parser.add_argument("--max-iterations", "-m", type=int, default=3,
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
   
   # Create results directory
   Path("results").mkdir(exist_ok=True)
   
   try:
      if args.workflow in ["basic", "all"]:
         run_basic_workflow_example(config, args.max_iterations)
      
      if args.workflow in ["advanced", "all"]:
         run_advanced_workflow_example(config, args.max_iterations)
      
      if args.workflow in ["agent", "all"]:
         run_hyperparam_agent_example(config)
      
      print("\n" + "="*60)
      print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
      print("="*60)
      
   except Exception as e:
      logger.error(f"Error running examples: {e}")
      sys.exit(1)


if __name__ == "__main__":
   main() 