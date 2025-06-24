#!/usr/bin/env python3
"""
Hyperparameter Agent that communicates with Sophia's LLM service.
Suggests hyperparameters based on previous training results.
"""

import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import sys
import os

# Add src/utils to path to import inference_auth_token
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from inference_auth_token import get_access_token


class AuthenticationError(Exception):
   """Exception raised when authentication fails with the LLM service."""
   pass


class HyperparameterAgent:
   """Agent that uses LLM on Sophia to suggest hyperparameters."""
   
   def __init__(self, sophia_url: str, api_key: str = None):
      self.sophia_url = sophia_url
      self.api_key = api_key or get_access_token()
      self.logger = logging.getLogger(__name__)
      
      # Initialize LangChain ChatOpenAI client
      self.llm = ChatOpenAI(
         model='meta-llama/Llama-3.3-70B-Instruct',
         base_url=self.sophia_url,
         api_key=self.api_key,
         temperature=0.1,  # Low temperature for more consistent results
         max_tokens=1000
      )
      
      # Track conversation history for context
      self.conversation_history = []
      
      # Define hyperparameter search space
      self.search_space = {
         "model_type": ["resnet18", "resnet34", "resnet50", "vgg16", "densenet121"],
         "learning_rate": {"min": 1e-5, "max": 1e-1, "type": "float"},
         "batch_size": [32, 64, 128, 256],
         "num_epochs": {"min": 10, "max": 100, "type": "int"},
         "hidden_size": [512, 1024, 2048],
         "num_layers": [2, 3, 4],
         "dropout_rate": {"min": 0.1, "max": 0.5, "type": "float"},
         "weight_decay": {"min": 1e-6, "max": 1e-3, "type": "float"}
      }
   
   def suggest_hyperparameters(self, context: Dict[str, Any]) -> Dict[str, Any]:
      """Get hyperparameter suggestions from Sophia LLM."""
      
      # Prepare the prompt for the LLM
      prompt = self._build_prompt(context)
      
      try:
         # Send request to Sophia LLM service using LangChain
         response = self._call_sophia_llm(prompt)
         
         # Parse the response
         hyperparams = self._parse_llm_response(response)
         
         # Validate hyperparameters
         validated_params = self._validate_hyperparameters(hyperparams)
         
         # Add to conversation history
         self.conversation_history.append({
            "timestamp": datetime.now().isoformat(),
            "context": context,
            "suggested_params": validated_params,
            "response": response
         })
         
         self.logger.info(f"Suggested hyperparameters: {validated_params}")
         return validated_params
         
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
   
   def _build_prompt(self, context: Dict[str, Any]) -> str:
      """Build a prompt for the LLM based on context and history."""
      
      prompt = f"""You are an expert machine learning engineer optimizing hyperparameters for CIFAR-100 image classification.

Current context:
- Iteration: {context.get('iteration', 0)}
- Dataset: {context.get('dataset', 'cifar100')}
- Objective: {context.get('objective', 'maximize_accuracy')}
- Experiment ID: {context.get('experiment_id', 'unknown')}

Previous results:"""

      if context.get('completed_jobs'):
         prompt += "\n"
         for job in context['completed_jobs'][-5:]:  # Last 5 jobs
            results = job.get('results', {})
            prompt += f"""
- Hyperparameters: {job.get('hyperparams', {})}
  - Final Accuracy: {results.get('final_accuracy', 'N/A'):.4f}
  - AUC: {results.get('auc', 'N/A'):.4f}
  - Training Time: {results.get('training_time', 'N/A'):.2f}s"""

      prompt += f"""

Available hyperparameter search space:
{json.dumps(self.search_space, indent=2)}

Based on the previous results and your expertise, suggest the next set of hyperparameters to try. 
Focus on exploring promising regions of the search space and avoiding configurations that performed poorly.

Respond with a JSON object containing the hyperparameters in this exact format:
{{
   "model_type": "resnet18",
   "learning_rate": 0.001,
   "batch_size": 128,
   "num_epochs": 50,
   "hidden_size": 1024,
   "num_layers": 3,
   "dropout_rate": 0.2,
   "weight_decay": 1e-4
}}

Explain your reasoning briefly before the JSON response."""
      
      return prompt
   
   def _call_sophia_llm(self, prompt: str) -> str:
      """Make API call to Sophia LLM service using LangChain."""
      
      try:
         # Create messages for the LLM
         system_message = SystemMessage(content="You are an expert ML engineer specializing in hyperparameter optimization.")
         human_message = HumanMessage(content=prompt)
         
         # Invoke the LLM
         response = self.llm.invoke([system_message, human_message])
         
         # Extract the content from the response
         return response.content
         
      except Exception as e:
         self.logger.error(f"Error calling Sophia LLM: {e}")
         raise
   
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
            return json.loads(json_str)
         
         # If no JSON found, try to parse the entire response
         return json.loads(response)
         
      except json.JSONDecodeError as e:
         self.logger.error(f"Error parsing LLM response: {e}")
         self.logger.error(f"Response: {response}")
         raise
   
   def _validate_hyperparameters(self, hyperparams: Dict[str, Any]) -> Dict[str, Any]:
      """Validate and normalize hyperparameters."""
      
      validated = {}
      
      # Validate model_type
      if hyperparams.get("model_type") in self.search_space["model_type"]:
         validated["model_type"] = hyperparams["model_type"]
      else:
         validated["model_type"] = "resnet18"  # Default
      
      # Validate learning_rate
      lr = hyperparams.get("learning_rate", 0.001)
      lr_bounds = self.search_space["learning_rate"]
      validated["learning_rate"] = max(lr_bounds["min"], min(lr_bounds["max"], lr))
      
      # Validate batch_size
      if hyperparams.get("batch_size") in self.search_space["batch_size"]:
         validated["batch_size"] = hyperparams["batch_size"]
      else:
         validated["batch_size"] = 128  # Default
      
      # Validate num_epochs
      epochs = hyperparams.get("num_epochs", 50)
      epochs_bounds = self.search_space["num_epochs"]
      validated["num_epochs"] = max(epochs_bounds["min"], min(epochs_bounds["max"], epochs))
      
      # Validate hidden_size
      if hyperparams.get("hidden_size") in self.search_space["hidden_size"]:
         validated["hidden_size"] = hyperparams["hidden_size"]
      else:
         validated["hidden_size"] = 1024  # Default
      
      # Validate num_layers
      if hyperparams.get("num_layers") in self.search_space["num_layers"]:
         validated["num_layers"] = hyperparams["num_layers"]
      else:
         validated["num_layers"] = 3  # Default
      
      # Validate dropout_rate
      dropout = hyperparams.get("dropout_rate", 0.2)
      dropout_bounds = self.search_space["dropout_rate"]
      validated["dropout_rate"] = max(dropout_bounds["min"], min(dropout_bounds["max"], dropout))
      
      # Validate weight_decay
      wd = hyperparams.get("weight_decay", 1e-4)
      wd_bounds = self.search_space["weight_decay"]
      validated["weight_decay"] = max(wd_bounds["min"], min(wd_bounds["max"], wd))
      
      return validated
   
   def _random_hyperparameters(self) -> Dict[str, Any]:
      """Generate random hyperparameters as fallback."""
      import random
      
      return {
         "model_type": random.choice(self.search_space["model_type"]),
         "learning_rate": random.uniform(
            self.search_space["learning_rate"]["min"],
            self.search_space["learning_rate"]["max"]
         ),
         "batch_size": random.choice(self.search_space["batch_size"]),
         "num_epochs": random.randint(
            self.search_space["num_epochs"]["min"],
            self.search_space["num_epochs"]["max"]
         ),
         "hidden_size": random.choice(self.search_space["hidden_size"]),
         "num_layers": random.choice(self.search_space["num_layers"]),
         "dropout_rate": random.uniform(
            self.search_space["dropout_rate"]["min"],
            self.search_space["dropout_rate"]["max"]
         ),
         "weight_decay": random.uniform(
            self.search_space["weight_decay"]["min"],
            self.search_space["weight_decay"]["max"]
         )
      } 