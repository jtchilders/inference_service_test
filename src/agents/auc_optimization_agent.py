#!/usr/bin/env python3
"""
AUC-Optimized Hyperparameter Agent with Exploration-Exploitation Strategy.
Focuses on maximizing AUC with intelligent parameter space exploration.
"""

import json
import logging
import random
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass
from collections import defaultdict
import sys
import os

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from scipy import stats
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern
from sklearn.preprocessing import StandardScaler

# Add src/utils to path to import inference_auth_token
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from inference_auth_token import get_access_token


class AuthenticationError(Exception):
   """Exception raised when authentication fails with the LLM service."""
   pass


@dataclass
class ParameterRegion:
   """Represents a region in hyperparameter space with performance statistics."""
   center: Dict[str, Any]
   radius: Dict[str, float]
   auc_scores: List[float]
   sample_count: int
   
   @property
   def mean_auc(self) -> float:
      return np.mean(self.auc_scores) if self.auc_scores else 0.0
   
   @property  
   def std_auc(self) -> float:
      return np.std(self.auc_scores) if len(self.auc_scores) > 1 else 1.0
   
   @property
   def confidence_interval(self) -> Tuple[float, float]:
      """95% confidence interval for mean AUC."""
      if len(self.auc_scores) < 2:
         return (self.mean_auc - 1.0, self.mean_auc + 1.0)
      
      sem = stats.sem(self.auc_scores)
      ci = stats.t.interval(0.95, len(self.auc_scores)-1, 
                           loc=self.mean_auc, scale=sem)
      return ci


class AUCOptimizationAgent:
   """Enhanced hyperparameter agent optimized for AUC maximization."""
   
   def __init__(self, sophia_url: str, config: Dict[str, Any], api_key: str = None):
      self.sophia_url = sophia_url
      self.config = config
      self.api_key = api_key or get_access_token()
      self.logger = logging.getLogger(__name__)
      
      # Initialize LangChain components
      self.llm = ChatOpenAI(
         base_url=self.sophia_url,
         api_key=self.api_key,
         model=config.get("sophia", {}).get("model", "meta-llama/Llama-3.3-70B-Instruct"),
         temperature=config.get("sophia", {}).get("temperature", 0.1)
      )
      
      # Optimization configuration
      self.optimization_config = config.get("optimization", {})
      self.search_space_config = config.get("search_space", {})
      self.strategy_config = self.optimization_config.get("strategy", {})
      
      # Search strategy parameters
      self.initial_samples = self.strategy_config.get("initial_samples", 20)
      self.base_exploration_factor = self.strategy_config.get("exploration_factor", 0.7)
      self.exploitation_ramp = self.strategy_config.get("exploitation_ramp", 0.05)
      self.min_exploration = self.strategy_config.get("min_exploration", 0.2)
      
      # Track experiment state
      self.completed_jobs = []
      self.iteration_count = 0
      self.best_auc = 0.0
      self.best_hyperparams = None
      
      # Region-based exploration
      self.promising_regions = []
      self.region_radius_factor = 0.2  # Size of regions relative to parameter range
      
      # Gaussian Process for surrogate modeling
      self.gp_model = None
      self.parameter_scaler = StandardScaler()
      self.use_gp_suggestions = True
      
      # Performance tracking
      self.auc_history = []
      self.exploration_history = []
      
      self.logger.info("Initialized AUC Optimization Agent")
   
   def suggest_hyperparameters(self, context: Dict[str, Any]) -> Dict[str, Any]:
      """
      Suggest next hyperparameters using exploration-exploitation strategy.
      
      Args:
         context: Current experiment context with completed jobs
         
      Returns:
         Dict containing suggested hyperparameters
      """
      
      # Update internal state with latest results
      self._update_state(context)
      
      # Determine current exploration factor
      exploration_factor = self._calculate_exploration_factor()
      
      try:
         if self.iteration_count < self.initial_samples:
            # Initial broad sampling phase
            hyperparams = self._generate_initial_sample()
            reasoning = f"Initial broad sampling (iteration {self.iteration_count + 1}/{self.initial_samples})"
         
         elif exploration_factor > 0.5:
            # High exploration: use LLM with broad suggestions
            hyperparams = self._get_llm_exploration_suggestion(context, exploration_factor)
            reasoning = f"LLM exploration (factor: {exploration_factor:.2f})"
         
         else:
            # High exploitation: focus on promising regions
            hyperparams = self._get_exploitation_suggestion(exploration_factor)
            reasoning = f"Region exploitation (factor: {exploration_factor:.2f})"
         
         # Validate and log suggestion
         validated_params = self._validate_hyperparameters(hyperparams)
         validated_params["reasoning"] = reasoning
         validated_params["exploration_factor"] = exploration_factor
         
         self.iteration_count += 1
         self.exploration_history.append(exploration_factor)
         
         self.logger.info(f"Suggested hyperparameters (iteration {self.iteration_count}): {validated_params}")
         return validated_params
         
      except Exception as e:
         self.logger.error(f"Error in hyperparameter suggestion: {e}")
         # Fallback to random sampling
         return self._generate_random_hyperparameters()
   
   def _update_state(self, context: Dict[str, Any]) -> None:
      """Update internal state with latest experiment results."""
      
      completed_jobs = context.get("completed_jobs", [])
      
      # Process new results
      for job in completed_jobs:
         if job not in self.completed_jobs:
            results = job.get("results", {})
            auc = results.get("auc", 0.0)
            
            self.auc_history.append(auc)
            
            # Track best result
            if auc > self.best_auc:
               self.best_auc = auc
               self.best_hyperparams = job.get("hyperparams", {})
               self.logger.info(f"New best AUC: {auc:.6f}")
            
            # Update promising regions
            self._update_promising_regions(job.get("hyperparams", {}), auc)
      
      self.completed_jobs = completed_jobs.copy()
      
      # Update Gaussian Process model
      if len(self.completed_jobs) >= 5:
         self._update_gp_model()
   
   def _calculate_exploration_factor(self) -> float:
      """Calculate current exploration factor based on iteration and performance."""
      
      # Base decay over iterations
      iteration_factor = max(
         self.min_exploration,
         self.base_exploration_factor - (self.iteration_count // 10) * self.exploitation_ramp
      )
      
      # Increase exploration if stuck (no improvement in recent iterations)
      if len(self.auc_history) >= 10:
         recent_improvement = max(self.auc_history[-10:]) - max(self.auc_history[-20:-10]) if len(self.auc_history) >= 20 else 0
         if recent_improvement < 0.001:  # No significant improvement
            iteration_factor = min(0.8, iteration_factor + 0.2)
      
      return iteration_factor
   
   def _generate_initial_sample(self) -> Dict[str, Any]:
      """Generate a random sample for initial broad exploration."""
      
      hyperparams = {}
      
      # Sample from configured initial suggestions or distributions
      for param, config in self.search_space_config.items():
         if "initial_suggestions" in config and random.random() < 0.5:
            # Use predefined good starting points
            hyperparams[param] = random.choice(config["initial_suggestions"])
         else:
            # Random sampling from full space
            hyperparams[param] = self._sample_parameter(param, config)
      
      return hyperparams
   
   def _get_llm_exploration_suggestion(self, context: Dict[str, Any], exploration_factor: float) -> Dict[str, Any]:
      """Get hyperparameter suggestions from LLM for exploration phase."""
      
      # Build context-aware prompt
      prompt = self._build_exploration_prompt(context, exploration_factor)
      
      try:
         # Create messages for the LLM
         messages = [
            SystemMessage(content="You are an expert ML engineer specializing in AUC optimization through intelligent hyperparameter exploration."),
            HumanMessage(content=prompt)
         ]
         
         response = self.llm.invoke(messages)
         hyperparams = self._parse_llm_response(response.content)
         
         # Add some controlled randomness for exploration
         if exploration_factor > 0.6:
            hyperparams = self._add_exploration_noise(hyperparams, exploration_factor)
         
         return hyperparams
         
      except Exception as e:
         self.logger.error(f"LLM suggestion failed: {e}")
         return self._generate_random_hyperparameters()
   
   def _get_exploitation_suggestion(self, exploration_factor: float) -> Dict[str, Any]:
      """Generate suggestions focused on promising regions (exploitation)."""
      
      if not self.promising_regions:
         return self._generate_random_hyperparameters()
      
      # Choose region based on upper confidence bound
      best_region = max(self.promising_regions, 
                       key=lambda r: r.mean_auc + 1.96 * r.std_auc / np.sqrt(r.sample_count))
      
      # Sample from promising region with some exploration
      hyperparams = {}
      for param, center_value in best_region.center.items():
         if param in self.search_space_config:
            config = self.search_space_config[param]
            radius = best_region.radius.get(param, 0.1)
            
            # Sample around the center with controlled noise
            noise_factor = exploration_factor * radius
            hyperparams[param] = self._sample_around_value(param, config, center_value, noise_factor)
      
      # Optionally use Gaussian Process suggestions
      if self.gp_model is not None and random.random() < 0.3:
         gp_suggestion = self._get_gp_suggestion()
         if gp_suggestion:
            # Blend GP suggestion with region-based suggestion
            for param in hyperparams:
               if param in gp_suggestion:
                  weight = 0.7  # Favor region-based approach
                  hyperparams[param] = self._blend_parameter_values(
                     param, hyperparams[param], gp_suggestion[param], weight
                  )
      
      return hyperparams
   
   def _update_promising_regions(self, hyperparams: Dict[str, Any], auc: float) -> None:
      """Update promising regions based on new results."""
      
      # Find if this point belongs to an existing region
      region_found = False
      for region in self.promising_regions:
         if self._point_in_region(hyperparams, region):
            region.auc_scores.append(auc)
            region.sample_count += 1
            region_found = True
            break
      
      # Create new region if high-performing and isolated
      if not region_found and auc > np.percentile(self.auc_history, 70):
         new_region = ParameterRegion(
            center=hyperparams.copy(),
            radius=self._calculate_region_radii(hyperparams),
            auc_scores=[auc],
            sample_count=1
         )
         self.promising_regions.append(new_region)
      
      # Prune poor-performing regions
      min_mean_auc = np.percentile(self.auc_history, 50) if len(self.auc_history) > 10 else 0
      self.promising_regions = [r for r in self.promising_regions if r.mean_auc >= min_mean_auc]
      
      # Limit number of regions
      if len(self.promising_regions) > 10:
         self.promising_regions.sort(key=lambda r: r.mean_auc, reverse=True)
         self.promising_regions = self.promising_regions[:10]
   
   def _update_gp_model(self) -> None:
      """Update Gaussian Process surrogate model."""
      
      try:
         # Prepare training data
         X, y = self._prepare_gp_data()
         
         if len(X) < 5:
            return
         
         # Initialize GP if needed
         if self.gp_model is None:
            kernel = Matern(length_scale=1.0, nu=2.5)
            self.gp_model = GaussianProcessRegressor(
               kernel=kernel,
               alpha=1e-6,
               normalize_y=True,
               n_restarts_optimizer=3
            )
         
         # Fit the model
         X_scaled = self.parameter_scaler.fit_transform(X)
         self.gp_model.fit(X_scaled, y)
         
         self.logger.debug(f"Updated GP model with {len(X)} samples")
         
      except Exception as e:
         self.logger.warning(f"Failed to update GP model: {e}")
         self.gp_model = None
   
   def _build_exploration_prompt(self, context: Dict[str, Any], exploration_factor: float) -> str:
      """Build context-aware prompt for LLM exploration."""
      
      # Analyze recent trends
      recent_results = self.completed_jobs[-10:] if len(self.completed_jobs) >= 10 else self.completed_jobs
      
      if recent_results:
         recent_aucs = [job.get("results", {}).get("auc", 0) for job in recent_results]
         avg_recent_auc = np.mean(recent_aucs)
         std_recent_auc = np.std(recent_aucs)
         
         # Find best performing configurations
         best_recent = max(recent_results, key=lambda x: x.get("results", {}).get("auc", 0))
         best_hyperparams = best_recent.get("hyperparams", {})
         best_auc = best_recent.get("results", {}).get("auc", 0)
      else:
         avg_recent_auc = 0.0
         std_recent_auc = 0.0
         best_hyperparams = {}
         best_auc = 0.0
      
      prompt = f"""You are optimizing hyperparameters for CIFAR-100 classification to maximize AUC (Area Under the Learning Curve).

CURRENT OPTIMIZATION STATUS:
- Iteration: {self.iteration_count + 1}
- Best AUC so far: {self.best_auc:.6f}
- Recent average AUC: {avg_recent_auc:.6f} ± {std_recent_auc:.6f}
- Exploration factor: {exploration_factor:.2f} (higher = more exploration)

BEST CONFIGURATION SO FAR:
{json.dumps(self.best_hyperparams, indent=2) if self.best_hyperparams else "None yet"}

RECENT HIGH-PERFORMING CONFIGURATION:
{json.dumps(best_hyperparams, indent=2) if best_hyperparams else "None yet"}
AUC: {best_auc:.6f}

SEARCH SPACE:
{json.dumps(self.search_space_config, indent=2)}

OPTIMIZATION STRATEGY:
- AUC includes both accuracy and training time (more epochs at same accuracy = higher AUC)
- With exploration_factor={exploration_factor:.2f}, balance exploration vs exploitation
- If exploration_factor > 0.5: Try diverse parameters to find new promising regions
- If exploration_factor ≤ 0.5: Focus on refining around successful configurations

RECENT PERFORMANCE ANALYSIS:"""

      if len(self.auc_history) >= 5:
         recent_trend = "improving" if self.auc_history[-1] > np.mean(self.auc_history[-5:-1]) else "plateauing"
         prompt += f"\n- Recent trend: {recent_trend}"
         
         # Parameter analysis
         if len(recent_results) >= 3:
            # Analyze which parameters correlate with high AUC
            param_analysis = self._analyze_parameter_performance(recent_results)
            prompt += f"\n- Parameter insights: {param_analysis}"

      prompt += f"""

Based on this analysis, suggest the next hyperparameters to try. Focus on maximizing AUC.

Respond with ONLY a valid JSON object:
{{
   "model_type": "resnet18",
   "learning_rate": 0.001,
   "batch_size": 128,
   "num_epochs": 50,
   "hidden_size": 1024,
   "num_layers": 3,
   "dropout_rate": 0.2,
   "weight_decay": 1e-4,
   "reasoning": "Brief explanation focusing on AUC optimization strategy"
}}"""
      
      return prompt
   
   def _analyze_parameter_performance(self, recent_results: List[Dict]) -> str:
      """Analyze which parameters correlate with high AUC in recent results."""
      
      if len(recent_results) < 3:
         return "Insufficient data for parameter analysis"
      
      # Extract parameter values and AUCs
      param_values = defaultdict(list)
      aucs = []
      
      for job in recent_results:
         auc = job.get("results", {}).get("auc", 0)
         hyperparams = job.get("hyperparams", {})
         
         aucs.append(auc)
         for param, value in hyperparams.items():
            param_values[param].append(value)
      
      # Find correlations
      insights = []
      median_auc = np.median(aucs)
      
      for param, values in param_values.items():
         if len(set(values)) > 1:  # Parameter varies across results
            # Split into high and low AUC groups
            high_auc_indices = [i for i, auc in enumerate(aucs) if auc > median_auc]
            low_auc_indices = [i for i, auc in enumerate(aucs) if auc <= median_auc]
            
            if high_auc_indices and low_auc_indices:
               high_values = [values[i] for i in high_auc_indices]
               low_values = [values[i] for i in low_auc_indices]
               
               # Analyze categorical parameters
               if isinstance(values[0], str):
                  high_mode = max(set(high_values), key=high_values.count) if high_values else None
                  if high_mode:
                     insights.append(f"{param}={high_mode} performed well")
               
               # Analyze numerical parameters
               elif isinstance(values[0], (int, float)):
                  high_mean = np.mean(high_values)
                  low_mean = np.mean(low_values) 
                  if abs(high_mean - low_mean) > 0.1 * abs(high_mean):
                     direction = "higher" if high_mean > low_mean else "lower"
                     insights.append(f"{direction} {param} values performed better")
      
      return "; ".join(insights) if insights else "No clear parameter patterns observed"
   
   def _sample_parameter(self, param: str, config: Dict[str, Any]) -> Any:
      """Sample a parameter value from its configuration."""
      
      param_type = config.get("type", "categorical")
      
      if param_type == "categorical" or "values" in config:
         values = config["values"]
         weights = config.get("weights")
         return np.random.choice(values, p=weights)
      
      elif param_type == "log_uniform":
         min_val = self._safe_float_convert(config["min"])
         max_val = self._safe_float_convert(config["max"])
         log_min = np.log10(min_val)
         log_max = np.log10(max_val)
         return 10 ** np.random.uniform(log_min, log_max)
      
      elif param_type == "uniform":
         min_val = self._safe_float_convert(config["min"])
         max_val = self._safe_float_convert(config["max"])
         return np.random.uniform(min_val, max_val)
      
      elif param_type == "int_uniform":
         min_val = self._safe_int_convert(config["min"])
         max_val = self._safe_int_convert(config["max"])
         return np.random.randint(min_val, max_val + 1)
      
      else:
         # Fallback to first value if available
         if "values" in config:
            return config["values"][0]
         return 0.001  # Default fallback
   
   def _sample_around_value(self, param: str, config: Dict[str, Any], center_value: Any, noise_factor: float) -> Any:
      """Sample a parameter value around a center point with controlled noise."""
      
      param_type = config.get("type", "categorical")
      
      if param_type == "categorical" or "values" in config:
         # For categorical, occasionally sample nearby values
         values = config["values"]
         if random.random() < noise_factor:
            return random.choice(values)
         else:
            return center_value
      
      elif param_type == "log_uniform":
         # Add noise in log space
         log_center = np.log10(center_value)
         min_val = self._safe_float_convert(config["min"])
         max_val = self._safe_float_convert(config["max"])
         log_range = np.log10(max_val) - np.log10(min_val)
         noise = np.random.normal(0, noise_factor * log_range)
         new_log_val = np.clip(log_center + noise, np.log10(min_val), np.log10(max_val))
         return 10 ** new_log_val
      
      elif param_type == "uniform":
         min_val = self._safe_float_convert(config["min"])
         max_val = self._safe_float_convert(config["max"])
         param_range = max_val - min_val
         noise = np.random.normal(0, noise_factor * param_range)
         return np.clip(center_value + noise, min_val, max_val)
      
      elif param_type == "int_uniform":
         min_val = self._safe_int_convert(config["min"])
         max_val = self._safe_int_convert(config["max"])
         param_range = max_val - min_val
         noise = np.random.normal(0, noise_factor * param_range)
         new_val = round(center_value + noise)
         return np.clip(new_val, min_val, max_val)
      
      return center_value
   
   def _validate_hyperparameters(self, hyperparams: Dict[str, Any]) -> Dict[str, Any]:
      """Validate and normalize hyperparameters based on search space."""
      
      validated = {}
      
      for param, config in self.search_space_config.items():
         if param in hyperparams:
            value = hyperparams[param]
            
            # Validate based on parameter type
            if "values" in config:
               if value in config["values"]:
                  validated[param] = value
               else:
                  validated[param] = config["values"][0]  # Default to first value
            
            elif config.get("type") == "log_uniform":
               min_val = self._safe_float_convert(config["min"])
               max_val = self._safe_float_convert(config["max"])
               safe_value = self._safe_float_convert(value)
               validated[param] = np.clip(safe_value, min_val, max_val)
            
            elif config.get("type") == "uniform":
               min_val = self._safe_float_convert(config["min"])
               max_val = self._safe_float_convert(config["max"])
               safe_value = self._safe_float_convert(value)
               validated[param] = np.clip(safe_value, min_val, max_val)
            
            elif config.get("type") == "int_uniform":
               min_val = self._safe_int_convert(config["min"])
               max_val = self._safe_int_convert(config["max"])
               safe_value = self._safe_float_convert(value)  # Convert to float first for clipping
               validated[param] = int(np.clip(safe_value, min_val, max_val))
            
            else:
               validated[param] = value
         
         else:
            # Use default value
            if "values" in config:
               validated[param] = config["values"][0]
            elif "initial_suggestions" in config:
               # Safely convert initial suggestions which might be strings like "1e-4"
               initial_value = config["initial_suggestions"][0]
               if config.get("type") in ["log_uniform", "uniform"]:
                  validated[param] = self._safe_float_convert(initial_value)
               elif config.get("type") == "int_uniform":
                  validated[param] = self._safe_int_convert(initial_value)
               else:
                  validated[param] = initial_value
            else:
               validated[param] = self._get_default_value(param)
      
      return validated
   
   def _get_default_value(self, param: str) -> Any:
      """Get default value for a parameter."""
      defaults = {
         "model_type": "resnet18",
         "learning_rate": 0.001,
         "batch_size": 128,
         "num_epochs": 50,
         "hidden_size": 1024,
         "num_layers": 3,
         "dropout_rate": 0.2,
         "weight_decay": 1e-4
      }
      return defaults.get(param, 0.001)
   
   def _safe_float_convert(self, value: Any) -> float:
      """Safely convert a value to float, handling string scientific notation."""
      if isinstance(value, str):
         try:
            return float(value)
         except ValueError:
            self.logger.warning(f"Failed to convert '{value}' to float, using 0.0")
            return 0.0
      elif isinstance(value, (int, float)):
         return float(value)
      else:
         self.logger.warning(f"Unexpected type for numeric conversion: {type(value)}, using 0.0")
         return 0.0
   
   def _safe_int_convert(self, value: Any) -> int:
      """Safely convert a value to int."""
      if isinstance(value, str):
         try:
            return int(float(value))  # Convert through float to handle scientific notation
         except ValueError:
            self.logger.warning(f"Failed to convert '{value}' to int, using 0")
            return 0
      elif isinstance(value, (int, float)):
         return int(value)
      else:
         self.logger.warning(f"Unexpected type for integer conversion: {type(value)}, using 0")
         return 0
   
   def _generate_random_hyperparameters(self) -> Dict[str, Any]:
      """Generate random hyperparameters as fallback."""
      
      hyperparams = {}
      for param, config in self.search_space_config.items():
         hyperparams[param] = self._sample_parameter(param, config)
      
      return self._validate_hyperparameters(hyperparams)
   
   def _parse_llm_response(self, response: str) -> Dict[str, Any]:
      """Parse LLM response to extract hyperparameters."""
      
      try:
         # Extract JSON from response
         lines = response.strip().split('\n')
         json_start = None
         
         for i, line in enumerate(lines):
            if line.strip().startswith('{'):
               json_start = i
               break
         
         if json_start is not None:
            json_str = '\n'.join(lines[json_start:])
            
            # Find matching braces
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
         
         # Try parsing entire response
         return json.loads(response)
         
      except json.JSONDecodeError as e:
         self.logger.error(f"Failed to parse LLM response: {e}")
         self.logger.error(f"Response: {response}")
         raise
   
   def get_optimization_summary(self) -> Dict[str, Any]:
      """Get current optimization status summary."""
      
      summary = {
         "iteration_count": self.iteration_count,
         "best_auc": self.best_auc,
         "best_hyperparams": self.best_hyperparams,
         "total_jobs": len(self.completed_jobs),
         "promising_regions": len(self.promising_regions),
         "current_exploration_factor": self._calculate_exploration_factor(),
      }
      
      if len(self.auc_history) >= 5:
         summary["recent_auc_stats"] = {
            "mean": float(np.mean(self.auc_history[-10:])),
            "std": float(np.std(self.auc_history[-10:])),
            "trend": "improving" if len(self.auc_history) >= 2 and self.auc_history[-1] > self.auc_history[-2] else "stable"
         }
      
      return summary
   
   # Additional helper methods for GP and region management...
   
   def _prepare_gp_data(self) -> Tuple[np.ndarray, np.ndarray]:
      """Prepare data for Gaussian Process model."""
      
      X = []
      y = []
      
      for job in self.completed_jobs:
         hyperparams = job.get("hyperparams", {})
         auc = job.get("results", {}).get("auc", 0.0)
         
         # Convert hyperparams to numerical vector
         x_vector = []
         for param in sorted(self.search_space_config.keys()):
            value = hyperparams.get(param, self._get_default_value(param))
            x_vector.append(self._param_to_numeric(param, value))
         
         X.append(x_vector)
         y.append(auc)
      
      return np.array(X), np.array(y)
   
   def _param_to_numeric(self, param: str, value: Any) -> float:
      """Convert parameter value to numeric for GP model."""
      
      config = self.search_space_config.get(param, {})
      
      if isinstance(value, str):
         # Categorical: map to index
         values = config.get("values", [value])
         return float(values.index(value) if value in values else 0)
      
      elif isinstance(value, (int, float)):
         # Numerical: use as-is or log-transform
         if config.get("type") == "log_uniform":
            return np.log10(max(value, 1e-10))
         return float(value)
      
      return 0.0
   
   def _get_gp_suggestion(self) -> Optional[Dict[str, Any]]:
      """Get suggestion from Gaussian Process model."""
      
      if self.gp_model is None:
         return None
      
      try:
         # Generate candidate points and evaluate with GP
         n_candidates = 100
         candidates = []
         
         for _ in range(n_candidates):
            candidate = []
            for param in sorted(self.search_space_config.keys()):
               config = self.search_space_config[param]
               value = self._sample_parameter(param, config)
               candidate.append(self._param_to_numeric(param, value))
            candidates.append(candidate)
         
         candidates = np.array(candidates)
         candidates_scaled = self.parameter_scaler.transform(candidates)
         
         # Predict with uncertainty
         means, stds = self.gp_model.predict(candidates_scaled, return_std=True)
         
         # Upper confidence bound acquisition
         acquisition = means + 1.96 * stds
         best_idx = np.argmax(acquisition)
         
         # Convert back to hyperparameters
         best_candidate = candidates[best_idx]
         hyperparams = {}
         
         for i, param in enumerate(sorted(self.search_space_config.keys())):
            numeric_value = best_candidate[i]
            hyperparams[param] = self._numeric_to_param(param, numeric_value)
         
         return hyperparams
         
      except Exception as e:
         self.logger.warning(f"GP suggestion failed: {e}")
         return None
   
   def _numeric_to_param(self, param: str, numeric_value: float) -> Any:
      """Convert numeric value back to parameter value."""
      
      config = self.search_space_config.get(param, {})
      
      if "values" in config:
         # Categorical: map from index
         values = config["values"]
         idx = int(round(numeric_value)) % len(values)
         return values[idx]
      
      elif config.get("type") == "log_uniform":
         return 10 ** numeric_value
      
      elif config.get("type") == "int_uniform":
         return int(round(numeric_value))
      
      else:
         return numeric_value
   
   def _point_in_region(self, hyperparams: Dict[str, Any], region: ParameterRegion) -> bool:
      """Check if a point falls within a parameter region."""
      
      for param, value in hyperparams.items():
         if param in region.center:
            center_val = region.center[param]
            radius = region.radius.get(param, 0.1)
            
            if isinstance(value, str):
               if value != center_val:
                  return False
            else:
               relative_distance = abs(value - center_val) / max(abs(center_val), 1e-6)
               if relative_distance > radius:
                  return False
      
      return True
   
   def _calculate_region_radii(self, hyperparams: Dict[str, Any]) -> Dict[str, float]:
      """Calculate appropriate radii for a new region."""
      
      radii = {}
      for param, value in hyperparams.items():
         if param in self.search_space_config:
            config = self.search_space_config[param]
            
            if isinstance(value, str):
               radii[param] = 0.0  # Exact match for categorical
            elif config.get("type") == "log_uniform":
               min_val = self._safe_float_convert(config["min"])
               max_val = self._safe_float_convert(config["max"])
               log_range = np.log10(max_val) - np.log10(min_val)
               radii[param] = self.region_radius_factor * log_range
            else:
               min_val = self._safe_float_convert(config.get("min", 0))
               max_val = self._safe_float_convert(config.get("max", 1))
               param_range = max_val - min_val
               radii[param] = self.region_radius_factor * param_range
      
      return radii
   
   def _add_exploration_noise(self, hyperparams: Dict[str, Any], exploration_factor: float) -> Dict[str, Any]:
      """Add controlled noise to hyperparameters for exploration."""
      
      noisy_params = hyperparams.copy()
      
      for param, value in hyperparams.items():
         if param in self.search_space_config and random.random() < exploration_factor * 0.3:
            config = self.search_space_config[param]
            
            # Add appropriate noise based on parameter type
            if isinstance(value, str) and "values" in config:
               if random.random() < 0.2:  # 20% chance to change categorical
                  noisy_params[param] = random.choice(config["values"])
            
            elif isinstance(value, (int, float)):
               noise_std = 0.1 * exploration_factor  # Proportional to exploration factor
               
               if config.get("type") == "log_uniform":
                  log_val = np.log10(value)
                  min_val = self._safe_float_convert(config["min"])
                  max_val = self._safe_float_convert(config["max"])
                  log_range = np.log10(max_val) - np.log10(min_val)
                  noise = np.random.normal(0, noise_std * log_range)
                  new_log_val = np.clip(log_val + noise, np.log10(min_val), np.log10(max_val))
                  noisy_params[param] = 10 ** new_log_val
               
               elif config.get("type") in ["uniform", "int_uniform"]:
                  min_val = self._safe_float_convert(config["min"])
                  max_val = self._safe_float_convert(config["max"])
                  param_range = max_val - min_val
                  noise = np.random.normal(0, noise_std * param_range)
                  new_val = value + noise
                  new_val = np.clip(new_val, min_val, max_val)
                  
                  if config.get("type") == "int_uniform":
                     new_val = int(round(new_val))
                  
                  noisy_params[param] = new_val
      
      return noisy_params
   
   def _blend_parameter_values(self, param: str, value1: Any, value2: Any, weight1: float) -> Any:
      """Blend two parameter values with given weight."""
      
      if isinstance(value1, str):
         return value1 if random.random() < weight1 else value2
      
      elif isinstance(value1, (int, float)):
         blended = weight1 * value1 + (1 - weight1) * value2
         
         config = self.search_space_config.get(param, {})
         if config.get("type") == "int_uniform":
            return int(round(blended))
         
         return blended
      
      return value1 