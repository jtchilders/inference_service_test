#!/usr/bin/env python3
"""
LangChain-based Hyperparameter Agent with tools and structured outputs.
Uses LangChain tools and structured prompts for better hyperparameter optimization.
"""

import json
import logging
import random
from typing import Dict, List, Optional, Any, TypedDict
from datetime import datetime
from dataclasses import dataclass
import sys
import os

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

# Add src/utils to path to import inference_auth_token
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from inference_auth_token import get_access_token


class AuthenticationError(Exception):
   """Exception raised when authentication fails with the LLM service."""
   pass


@dataclass
class HyperparameterSearchSpace:
   """Defines the search space for hyperparameters."""
   model_type: List[str] = None
   learning_rate: Dict[str, Any] = None
   batch_size: List[int] = None
   num_epochs: Dict[str, Any] = None
   hidden_size: List[int] = None
   num_layers: List[int] = None
   dropout_rate: Dict[str, Any] = None
   weight_decay: Dict[str, Any] = None
   
   def __post_init__(self):
      if self.model_type is None:
         self.model_type = ["resnet18", "resnet34", "resnet50", "vgg16", "densenet121"]
      if self.learning_rate is None:
         self.learning_rate = {"min": 1e-5, "max": 1e-1, "type": "float"}
      if self.batch_size is None:
         self.batch_size = [32, 64, 128, 256]
      if self.num_epochs is None:
         self.num_epochs = {"min": 10, "max": 100, "type": "int"}
      if self.hidden_size is None:
         self.hidden_size = [512, 1024, 2048]
      if self.num_layers is None:
         self.num_layers = [2, 3, 4]
      if self.dropout_rate is None:
         self.dropout_rate = {"min": 0.1, "max": 0.5, "type": "float"}
      if self.weight_decay is None:
         self.weight_decay = {"min": 1e-6, "max": 1e-3, "type": "float"}


class Hyperparameters(BaseModel):
   """Structured output for hyperparameters."""
   model_type: str = Field(description="Model architecture type")
   learning_rate: float = Field(description="Learning rate for training")
   batch_size: int = Field(description="Batch size for training")
   num_epochs: int = Field(description="Number of training epochs")
   hidden_size: int = Field(description="Hidden layer size")
   num_layers: int = Field(description="Number of layers")
   dropout_rate: float = Field(description="Dropout rate")
   weight_decay: float = Field(description="Weight decay for regularization")
   reasoning: str = Field(description="Reasoning for these hyperparameter choices")


class LangChainHyperparameterAgent:
   """LangChain-based agent for hyperparameter optimization using tools and structured outputs."""
   
   def __init__(self, sophia_url: str, api_key: str = None):
      self.sophia_url = sophia_url
      self.api_key = api_key or get_access_token()
      self.logger = logging.getLogger(__name__)
      
      # Initialize LangChain components
      self.llm = ChatOpenAI(
         base_url=self.sophia_url,
         api_key=self.api_key,
         model="meta-llama/Llama-3.3-70B-Instruct",
         temperature=0.1
      )
      
      # Define search space
      self.search_space = HyperparameterSearchSpace()
      
      # Track conversation history
      self.conversation_history = []
      
      # Create tools
      self.tools = self._create_tools()
      
      # Create prompt template
      self.prompt = self._create_prompt()
      
      # Create output parser
      self.output_parser = JsonOutputParser(pydantic_object=Hyperparameters)
   
   def _create_tools(self) -> List:
      """Create LangChain tools for the agent."""
      
      @tool
      def analyze_previous_results(results: List[Dict[str, Any]]) -> str:
         """Analyze previous training results to understand patterns."""
         if not results:
            return "No previous results available for analysis."
         
         # Calculate statistics
         accuracies = [r.get("accuracy", 0.0) for r in results]
         avg_accuracy = sum(accuracies) / len(accuracies)
         max_accuracy = max(accuracies)
         min_accuracy = min(accuracies)
         
         # Find best and worst configurations
         best_result = max(results, key=lambda x: x.get("accuracy", 0.0))
         worst_result = min(results, key=lambda x: x.get("accuracy", 0.0))
         
         analysis = f"""
         Previous Results Analysis:
         - Number of experiments: {len(results)}
         - Average accuracy: {avg_accuracy:.4f}
         - Best accuracy: {max_accuracy:.4f}
         - Worst accuracy: {min_accuracy:.4f}
         - Accuracy range: {max_accuracy - min_accuracy:.4f}
         
         Best configuration: {best_result.get("hyperparams", {})}
         Worst configuration: {worst_result.get("hyperparams", {})}
         """
         
         return analysis
      
      @tool
      def get_search_space() -> str:
         """Get the available hyperparameter search space."""
         return f"""
         Available Hyperparameter Search Space:
         {json.dumps(self.search_space.__dict__, indent=2)}
         """
      
      @tool
      def suggest_learning_rate(previous_results: List[Dict[str, Any]]) -> str:
         """Suggest learning rate based on previous results."""
         if not previous_results:
            return "No previous results. Using default learning rate range."
         
         # Analyze learning rates from previous results
         lrs = [r.get("hyperparams", {}).get("learning_rate", 0.001) for r in previous_results]
         accuracies = [r.get("accuracy", 0.0) for r in previous_results]
         
         # Find best performing learning rate
         best_idx = accuracies.index(max(accuracies))
         best_lr = lrs[best_idx]
         
         # Suggest range around best performing LR
         min_lr = max(self.search_space.learning_rate["min"], best_lr * 0.1)
         max_lr = min(self.search_space.learning_rate["max"], best_lr * 10.0)
         
         return f"""
         Learning Rate Analysis:
         - Previous learning rates: {lrs}
         - Best performing LR: {best_lr}
         - Suggested range: [{min_lr:.6f}, {max_lr:.6f}]
         """
      
      @tool
      def suggest_model_architecture(previous_results: List[Dict[str, Any]]) -> str:
         """Suggest model architecture based on previous results."""
         if not previous_results:
            return "No previous results. All model types are equally valid."
         
         # Count performance by model type
         model_performance = {}
         for result in previous_results:
            model_type = result.get("hyperparams", {}).get("model_type", "unknown")
            accuracy = result.get("accuracy", 0.0)
            
            if model_type not in model_performance:
               model_performance[model_type] = []
            model_performance[model_type].append(accuracy)
         
         # Calculate average performance per model
         avg_performance = {}
         for model_type, accuracies in model_performance.items():
            avg_performance[model_type] = sum(accuracies) / len(accuracies)
         
         # Sort by performance
         sorted_models = sorted(avg_performance.items(), key=lambda x: x[1], reverse=True)
         
         analysis = "Model Architecture Performance:\n"
         for model_type, avg_acc in sorted_models:
            analysis += f"- {model_type}: {avg_acc:.4f} (avg)\n"
         
         return analysis
      
      return [analyze_previous_results, get_search_space, suggest_learning_rate, suggest_model_architecture]
   
   def _create_prompt(self) -> ChatPromptTemplate:
      """Create the prompt template for hyperparameter suggestion."""
      
      template = """You are an expert machine learning engineer optimizing hyperparameters for CIFAR-100 image classification.

Your task is to suggest the next set of hyperparameters to try based on previous results and your expertise.

Current context:
- Iteration: {iteration}
- Dataset: {dataset}
- Objective: {objective}
- Experiment ID: {experiment_id}

Previous results: {previous_results}

Available tools:
- analyze_previous_results: Analyze previous training results
- get_search_space: Get available hyperparameter ranges
- suggest_learning_rate: Suggest learning rate based on results
- suggest_model_architecture: Suggest model architecture based on results

Use the tools to analyze the situation and then provide your hyperparameter suggestions.

IMPORTANT: Respond with ONLY a valid JSON object. Do not include any comments, explanations, or additional text outside the JSON. The JSON must be properly formatted without any trailing commas or comments.

Respond with a JSON object containing the hyperparameters in this exact format:
{{
   "model_type": "resnet18",
   "learning_rate": 0.001,
   "batch_size": 128,
   "num_epochs": 50,
   "hidden_size": 1024,
   "num_layers": 3,
   "dropout_rate": 0.2,
   "weight_decay": 1e-4,
   "reasoning": "Brief explanation of your choices"
}}

Focus on exploring promising regions of the search space and avoiding configurations that performed poorly. Provide ONLY the JSON object, no additional text."""
      
      return ChatPromptTemplate.from_template(template)
   
   def suggest_hyperparameters(self, context: Dict[str, Any]) -> Dict[str, Any]:
      """Get hyperparameter suggestions using LangChain tools and structured output."""
      
      try:
         # Prepare context for the prompt
         previous_results = context.get("completed_jobs", [])
         
         # Convert previous results to the format expected by tools
         results_for_tools = []
         for job in previous_results:
            results_for_tools.append({
               "hyperparams": job.get("hyperparams", {}),
               "accuracy": job.get("results", {}).get("final_accuracy", 0.0),
               "auc": job.get("results", {}).get("auc", 0.0),
               "training_time": job.get("results", {}).get("training_time", 0.0)
            })
         
         # Create messages for the LLM
         messages = [
            SystemMessage(content="You are an expert ML engineer specializing in hyperparameter optimization."),
            HumanMessage(content=self.prompt.format(
               iteration=context.get("iteration", 0),
               dataset=context.get("dataset", "cifar100"),
               objective=context.get("objective", "maximize_accuracy"),
               experiment_id=context.get("experiment_id", "unknown"),
               previous_results=json.dumps(results_for_tools, indent=2)
            ))
         ]
         
         # Add conversation history
         for msg in self.conversation_history[-5:]:  # Last 5 messages
            if msg["type"] == "user":
               messages.append(HumanMessage(content=msg["content"]))
            elif msg["type"] == "assistant":
               messages.append(AIMessage(content=msg["content"]))
         
         # Get response from LLM
         response = self.llm.invoke(messages)
         
         # Parse the response
         try:
            # Try to extract JSON from the response
            content = response.content
            hyperparams = self._parse_llm_response(content)
            
            # Validate hyperparameters
            validated_params = self._validate_hyperparameters(hyperparams)
            
            # Add to conversation history
            self.conversation_history.append({
               "timestamp": datetime.now().isoformat(),
               "type": "user",
               "content": f"Suggest hyperparameters for iteration {context.get('iteration', 0)}"
            })
            self.conversation_history.append({
               "timestamp": datetime.now().isoformat(),
               "type": "assistant",
               "content": content
            })
            
            # Keep only recent history
            if len(self.conversation_history) > 20:
               self.conversation_history = self.conversation_history[-20:]
            
            self.logger.info(f"Suggested hyperparameters: {validated_params}")
            return validated_params
            
         except Exception as e:
            self.logger.error(f"Error parsing LLM response: {e}")
            return self._random_hyperparameters()
         
      except Exception as e:
         error_msg = str(e)
         self.logger.error(f"Error getting hyperparameter suggestions: {error_msg}")
         
         # Check if this is an authentication error (403 Forbidden)
         if "403" in error_msg and ("Forbidden" in error_msg or "Permission denied" in error_msg):
            auth_error_msg = (
               "Authentication failed with the LLM service. This is likely due to expired Globus credentials. "
               "Please re-authenticate by running: 'python3 inference_auth_token.py authenticate --force'. "
               "Make sure you authenticate with an authorized identity provider: ['Argonne National Laboratory', 'Argonne LCF']."
            )
            self.logger.error(auth_error_msg)
            raise AuthenticationError(auth_error_msg)
         
         # For other errors, still fall back to random hyperparameters
         return self._random_hyperparameters()
   
   def update_with_results(self, results: List[Dict[str, Any]]) -> None:
      """Update the agent with new training results."""
      if not results:
         return
      
      # Add results to conversation history
      self.conversation_history.append({
         "timestamp": datetime.now().isoformat(),
         "type": "results_update",
         "results": results
      })
      
      # Keep only recent history to avoid context overflow
      if len(self.conversation_history) > 20:
         self.conversation_history = self.conversation_history[-20:]
   
   def _parse_llm_response(self, response: str) -> Dict[str, Any]:
      """Parse the LLM response to extract hyperparameters."""
      
      try:
         # Try to extract JSON from the response
         lines = response.strip().split('\n')
         json_start = None
         
         for i, line in enumerate(lines):
            if line.strip().startswith('{'):
               json_start = i
               break
         
         if json_start is not None:
            json_str = '\n'.join(lines[json_start:])
            # Find the end of the JSON object
            brace_count = 0
            json_end = 0
            
            for i, char in enumerate(json_str):
               if char == '{':
                  brace_count += 1
               elif char == '}':
                  brace_count -= 1
                  if brace_count == 0:
                     json_end = i + 1
                     break
            
            json_str = json_str[:json_end]
            
            # Clean the JSON string to remove comments and fix common issues
            original_json = json_str
            json_str = self._clean_json_string(json_str)
            
            self.logger.debug(f"Original JSON: {original_json}")
            self.logger.debug(f"Cleaned JSON: {json_str}")
            
            return json.loads(json_str)
         
         # If no JSON found, try to parse the entire response
         cleaned_response = self._clean_json_string(response)
         self.logger.debug(f"Cleaned full response: {cleaned_response}")
         return json.loads(cleaned_response)
         
      except json.JSONDecodeError as e:
         self.logger.error(f"Error parsing LLM response: {e}")
         self.logger.error(f"Response: {response}")
         raise
   
   def _clean_json_string(self, json_str: str) -> str:
      """Clean JSON string by removing comments and fixing common formatting issues."""
      
      lines = json_str.split('\n')
      cleaned_lines = []
      
      for line in lines:
         # Remove comments (both # and // style)
         if '#' in line:
            line = line.split('#')[0]
         if '//' in line:
            line = line.split('//')[0]
         
         # Remove trailing commas before closing braces/brackets
         line = line.rstrip()
         if line.endswith(',') and (line.strip().endswith(',}') or line.strip().endswith(',]')):
            line = line[:-1] + line[-1].replace(',', '')
         
         # Fix trailing commas on the same line
         if line.strip().endswith(',}'):
            line = line.replace(',}', '}')
         if line.strip().endswith(',]'):
            line = line.replace(',]', ']')
         
         # Only add non-empty lines
         if line.strip():
            cleaned_lines.append(line)
      
      cleaned_json = '\n'.join(cleaned_lines)
      
      # Additional cleaning: remove any remaining trailing commas
      import re
      cleaned_json = re.sub(r',(\s*[}\]])', r'\1', cleaned_json)
      
      return cleaned_json
   
   def _validate_hyperparameters(self, hyperparams: Dict[str, Any]) -> Dict[str, Any]:
      """Validate and normalize hyperparameters."""
      
      validated = {}
      
      # Validate model_type
      if hyperparams.get("model_type") in self.search_space.model_type:
         validated["model_type"] = hyperparams["model_type"]
      else:
         validated["model_type"] = "resnet18"  # Default
      
      # Validate learning_rate
      lr = hyperparams.get("learning_rate", 0.001)
      lr_bounds = self.search_space.learning_rate
      validated["learning_rate"] = max(lr_bounds["min"], min(lr_bounds["max"], lr))
      
      # Validate batch_size
      if hyperparams.get("batch_size") in self.search_space.batch_size:
         validated["batch_size"] = hyperparams["batch_size"]
      else:
         validated["batch_size"] = 128  # Default
      
      # Validate num_epochs
      epochs = hyperparams.get("num_epochs", 50)
      epochs_bounds = self.search_space.num_epochs
      validated["num_epochs"] = max(epochs_bounds["min"], min(epochs_bounds["max"], epochs))
      
      # Validate hidden_size
      if hyperparams.get("hidden_size") in self.search_space.hidden_size:
         validated["hidden_size"] = hyperparams["hidden_size"]
      else:
         validated["hidden_size"] = 1024  # Default
      
      # Validate num_layers
      if hyperparams.get("num_layers") in self.search_space.num_layers:
         validated["num_layers"] = hyperparams["num_layers"]
      else:
         validated["num_layers"] = 3  # Default
      
      # Validate dropout_rate
      dropout = hyperparams.get("dropout_rate", 0.2)
      dropout_bounds = self.search_space.dropout_rate
      validated["dropout_rate"] = max(dropout_bounds["min"], min(dropout_bounds["max"], dropout))
      
      # Validate weight_decay
      wd = hyperparams.get("weight_decay", 1e-4)
      wd_bounds = self.search_space.weight_decay
      validated["weight_decay"] = max(wd_bounds["min"], min(wd_bounds["max"], wd))
      
      return validated
   
   def _random_hyperparameters(self) -> Dict[str, Any]:
      """Generate random hyperparameters as fallback."""
      
      return {
         "model_type": random.choice(self.search_space.model_type),
         "learning_rate": random.uniform(
            self.search_space.learning_rate["min"],
            self.search_space.learning_rate["max"]
         ),
         "batch_size": random.choice(self.search_space.batch_size),
         "num_epochs": random.randint(
            self.search_space.num_epochs["min"],
            self.search_space.num_epochs["max"]
         ),
         "hidden_size": random.choice(self.search_space.hidden_size),
         "num_layers": random.choice(self.search_space.num_layers),
         "dropout_rate": random.uniform(
            self.search_space.dropout_rate["min"],
            self.search_space.dropout_rate["max"]
         ),
         "weight_decay": random.uniform(
            self.search_space.weight_decay["min"],
            self.search_space.weight_decay["max"]
         )
      } 