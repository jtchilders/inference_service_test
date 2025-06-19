# LangGraph/LangChain Implementation

This document describes the refactored agentic workflow implementation using LangGraph and LangChain for better structure, state management, and tool integration.

## Overview

The original implementation has been refactored to use LangGraph for workflow orchestration and LangChain for agent interactions. This provides:

- **Structured State Management**: LangGraph's state management ensures consistent workflow state across nodes
- **Conditional Routing**: Dynamic workflow paths based on conditions and results
- **Tool Integration**: LangChain tools for enhanced agent capabilities
- **Better Error Handling**: Structured error handling and recovery mechanisms
- **Extensibility**: Easy to add new nodes, tools, and workflow patterns

## Architecture

### Components

1. **LangGraph Workflows**
   - `LangGraphWorkflow`: Basic workflow with essential nodes
   - `AdvancedLangGraphWorkflow`: Advanced workflow with additional features

2. **LangChain Agents**
   - `LangChainHyperparameterAgent`: Agent with tools and structured outputs

3. **State Management**
   - `WorkflowState`: Basic state for simple workflows
   - `AdvancedWorkflowState`: Enhanced state with additional tracking

## Workflow Structure

### Basic LangGraph Workflow

```
initialize → suggest_hyperparams → submit_job → collect_results → check_convergence
                                                                    ↓
                                                              [continue/stop]
                                                                    ↓
                                                              generate_report
```

### Advanced LangGraph Workflow

```
initialize → suggest_hyperparams → submit_job → collect_results → check_convergence
                                                                    ↓
                                                              [continue/stop]
                                                                    ↓
                                                              generate_report
```

## Key Features

### 1. State Management

LangGraph provides automatic state management through the `StateGraph` class:

```python
@dataclass
class WorkflowState:
   iteration: int = 0
   max_iterations: int = 10
   experiment_id: str = ""
   completed_jobs: List[Dict[str, Any]] = field(default_factory=list)
   pending_jobs: List[Dict[str, Any]] = field(default_factory=list)
   current_hyperparams: Optional[Dict[str, Any]] = None
   should_stop: bool = False
   error_message: Optional[str] = None
   final_report: Optional[Dict[str, Any]] = None
```

### 2. Conditional Routing

Workflows can dynamically route based on conditions:

```python
workflow.add_conditional_edges(
   "check_convergence",
   self._should_continue,
   {
      "continue": "suggest_hyperparams",
      "stop": "generate_report"
   }
)
```

### 3. LangChain Tools

The hyperparameter agent uses LangChain tools for enhanced capabilities:

```python
@tool
def analyze_previous_results(results: List[Dict[str, Any]]) -> str:
   """Analyze previous training results to understand patterns."""
   # Implementation...

@tool
def suggest_learning_rate(previous_results: List[Dict[str, Any]]) -> str:
   """Suggest learning rate based on previous results."""
   # Implementation...
```

### 4. Structured Outputs

Using Pydantic models for structured outputs:

```python
class Hyperparameters(BaseModel):
   model_type: str = Field(description="Model architecture type")
   learning_rate: float = Field(description="Learning rate for training")
   batch_size: int = Field(description="Batch size for training")
   # ... other fields
```

## Usage Examples

### Running Basic Workflow

```bash
python src/agents/langgraph_workflow.py \
   --config config.yaml \
   --max-iterations 10 \
   --log-level INFO
```

### Running Advanced Workflow

```bash
python src/agents/advanced_langgraph_workflow.py \
   --config config.yaml \
   --max-iterations 10 \
   --log-level INFO
```

### Running Examples

```bash
python examples/langgraph_example.py \
   --config config.yaml \
   --workflow all \
   --max-iterations 3
```

## Configuration

The configuration remains the same as the original implementation:

```yaml
# Sophia LLM service configuration
sophia:
  url: "https://sophia.alcf.anl.gov/v1"
  api_key: "${SOPHIA_API_KEY}"

# Aurora PBS job configuration
aurora:
  host: "aurora.alcf.anl.gov"
  user: "${AURORA_USER}"
  key_path: "${AURORA_SSH_KEY}"
  pbs_template: "jobs/train_cifar100.pbs.template"

# Data and results configuration
data:
  dir: "data/cifar-100-python"

results:
  dir: "results"
```

## Benefits of LangGraph/LangChain

### 1. Better State Management
- Automatic state persistence across nodes
- Type-safe state definitions
- Clear state transitions

### 2. Enhanced Debugging
- Visual workflow graphs
- State inspection at each node
- Better error tracing

### 3. Extensibility
- Easy to add new nodes
- Simple to modify workflow logic
- Tool integration for agents

### 4. Production Readiness
- Built-in error handling
- Retry mechanisms
- Monitoring and logging

## Migration from Original Implementation

### Before (Original)
```python
class AgentWorkflow:
   def run_experiment(self, max_iterations: int = 10):
      for iteration in range(max_iterations):
         hyperparams = self._get_next_hyperparameters(iteration)
         job_id = self._submit_training_job(hyperparams, iteration)
         self._collect_completed_jobs()
         self._update_agent_with_results()
         if self._should_stop_experiment():
            break
```

### After (LangGraph)
```python
class LangGraphWorkflow:
   def _create_workflow(self) -> StateGraph:
      workflow = StateGraph(WorkflowState)
      workflow.add_node("suggest_hyperparams", self._suggest_hyperparams_node)
      workflow.add_node("submit_job", self._submit_job_node)
      workflow.add_node("collect_results", self._collect_results_node)
      workflow.add_conditional_edges("check_convergence", self._should_continue, {...})
      return workflow.compile()
   
   def run_experiment(self, max_iterations: int = 10):
      initial_state = WorkflowState(max_iterations=max_iterations)
      final_state = self.workflow.invoke(initial_state)
      return final_state.final_report
```

## Dependencies

The new implementation requires additional dependencies:

```txt
# LangGraph and LangChain for agentic workflows
langgraph>=0.2.0
langchain>=0.2.0
langchain-core>=0.2.0
langchain-community>=0.2.0
langchain-openai>=0.1.0
```

## Future Enhancements

1. **More Tools**: Add tools for resource monitoring, cost analysis, etc.
2. **Parallel Execution**: Support for parallel job submission
3. **Advanced Routing**: More sophisticated conditional logic
4. **Visualization**: Workflow visualization and monitoring
5. **Distributed Execution**: Support for distributed workflow execution

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all LangGraph/LangChain dependencies are installed
2. **State Issues**: Check that state fields are properly initialized
3. **Tool Errors**: Verify tool function signatures and return types
4. **Workflow Loops**: Ensure conditional edges have proper exit conditions

### Debug Mode

Run with debug logging to see detailed workflow execution:

```bash
python src/agents/langgraph_workflow.py \
   --config config.yaml \
   --log-level DEBUG
```

## Conclusion

The LangGraph/LangChain implementation provides a more robust, maintainable, and extensible foundation for agentic workflows. The structured approach makes it easier to add new features, debug issues, and scale the system. 