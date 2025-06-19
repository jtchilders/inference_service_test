#!/usr/bin/env python3
"""
Test script to verify Sophia integration with LangChain.
Tests the authentication and basic LLM functionality.
"""

import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# Add src/utils to path to import inference_auth_token
sys.path.append(str(Path(__file__).parent / "src" / "utils"))
from inference_auth_token import get_access_token


def test_sophia_authentication():
   """Test Sophia authentication."""
   print("Testing Sophia authentication...")
   
   try:
      access_token = get_access_token()
      print(f"✓ Successfully obtained access token: {access_token[:20]}...")
      return access_token
   except Exception as e:
      print(f"✗ Authentication failed: {e}")
      return None


def test_sophia_llm_integration(access_token):
   """Test Sophia LLM integration with LangChain."""
   print("\nTesting Sophia LLM integration...")
   
   try:
      # Sophia configuration
      sophia_url = "https://data-portal-dev.cels.anl.gov/resource_server/sophia/vllm/v1"
      model = "meta-llama/Llama-3.3-70B-Instruct"
      
      # Initialize LangChain ChatOpenAI client
      llm = ChatOpenAI(
         model=model,
         base_url=sophia_url,
         api_key=access_token,
         temperature=0.1,
         max_tokens=100
      )
      
      print(f"✓ Successfully initialized LangChain client with model: {model}")
      
      # Test simple message
      message = "Tell me about Argonne National Laboratory in one sentence."
      
      print(f"Sending test message: {message}")
      
      # Create messages for the LLM
      system_message = SystemMessage(content="You are a helpful assistant.")
      human_message = HumanMessage(content=message)
      
      # Invoke the LLM
      response = llm.invoke([system_message, human_message])
      
      print(f"✓ Successfully received response: {response.content}")
      
      return True
      
   except Exception as e:
      print(f"✗ LLM integration failed: {e}")
      return False


def test_hyperparameter_agent():
   """Test the hyperparameter agent integration."""
   print("\nTesting hyperparameter agent...")
   
   try:
      from src.agents.hyperparam_agent import HyperparameterAgent
      
      # Create agent
      sophia_url = "https://data-portal-dev.cels.anl.gov/resource_server/sophia/vllm/v1"
      agent = HyperparameterAgent(sophia_url=sophia_url)
      
      print("✓ Successfully created HyperparameterAgent")
      
      # Test context
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
            }
         ],
         "experiment_id": "test_experiment",
         "dataset": "cifar100",
         "objective": "maximize_accuracy"
      }
      
      # Get hyperparameter suggestions
      suggestions = agent.suggest_hyperparameters(context)
      
      print(f"✓ Successfully got hyperparameter suggestions: {suggestions}")
      
      return True
      
   except Exception as e:
      print(f"✗ Hyperparameter agent test failed: {e}")
      return False


def main():
   """Main test function."""
   print("="*60)
   print("SOPHIA INTEGRATION TEST")
   print("="*60)
   
   # Test authentication
   access_token = test_sophia_authentication()
   if not access_token:
      print("\n❌ Authentication test failed. Exiting.")
      sys.exit(1)
   
   # Test LLM integration
   llm_success = test_sophia_llm_integration(access_token)
   if not llm_success:
      print("\n❌ LLM integration test failed. Exiting.")
      sys.exit(1)
   
   # Test hyperparameter agent
   agent_success = test_hyperparameter_agent()
   if not agent_success:
      print("\n❌ Hyperparameter agent test failed. Exiting.")
      sys.exit(1)
   
   print("\n" + "="*60)
   print("✅ ALL TESTS PASSED!")
   print("✅ Sophia integration with LangChain is working correctly.")
   print("="*60)


if __name__ == "__main__":
   main() 