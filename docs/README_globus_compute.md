# Globus Compute Integration for Agentic Workflow

This document provides an overview of the Globus Compute integration for the agentic hyperparameter optimization workflow.

## Overview

The agentic workflow now supports two job scheduling methods:

1. **Direct PBS Submission** (default): Direct SSH-based PBS job submission to Aurora
2. **Globus Compute** (new): Service-based job submission using Globus Compute endpoints

## Benefits of Globus Compute

- **Simplified Authentication**: No need to manage SSH keys or MFA sessions
- **Better Error Handling**: Automatic retries and robust error recovery
- **Service Management**: Jobs are managed by the Globus Compute service
- **Scalability**: Better support for concurrent job submission
- **Monitoring**: Enhanced job monitoring and status tracking

## Quick Start

### 1. Setup Globus Compute Endpoint

Follow the detailed instructions in [`docs/globus_compute_setup.md`](globus_compute_setup.md) to:

- Install `globus-compute-endpoint` on Aurora
- Configure the endpoint using the provided configuration
- Start the endpoint and get the endpoint ID

### 2. Configure the Workflow

Update your `config.yaml` to enable Globus Compute:

```yaml
# Enable Globus Compute
globus_compute:
  enabled: true
  endpoint_id: "your-endpoint-id-here"  # Replace with actual endpoint ID
  auth_method: "native_client"
  function_timeout: 3600
  max_retries: 2
```

### 3. Install Dependencies

```bash
pip install globus-compute-sdk
```

### 4. Run the Workflow

The workflow will automatically use Globus Compute when enabled:

```bash
python -m src.agents.main --config config.yaml
```

## Configuration Options

### Globus Compute Configuration

```yaml
globus_compute:
  enabled: false  # Set to true to enable Globus Compute
  endpoint_id: "your-endpoint-id"  # Required: Globus Compute endpoint ID
  endpoint_config_path: "config/globus_endpoint.yaml"  # Path to endpoint config
  auth_method: "native_client"  # Authentication method
  function_timeout: 3600  # Function timeout in seconds
  max_retries: 2  # Maximum retry attempts
```

### Authentication Methods

1. **native_client** (recommended): Interactive browser-based authentication
2. **confidential_client**: For automated environments with client credentials
3. **client_credentials**: For service-to-service authentication

## File Structure

```
├── config/
│   └── globus_endpoint.yaml         # Globus Compute endpoint configuration
├── docs/
│   ├── globus_compute_setup.md      # Setup instructions
│   └── README_globus_compute.md     # This file
├── examples/
│   ├── globus_compute_example.py    # Example usage
│   ├── globus_compute_diagnostics.py # Diagnostic testing script
│   ├── torch_test.py                # PyTorch test functions
│   └── user_test_pattern.py         # User's exact test pattern
└── src/agents/
    └── globus_job_scheduler.py      # Globus Compute job scheduler
```

## Testing the Setup

After setting up the endpoint, you can test it in several ways:

### Option 1: Using the Diagnostic Script

Run the comprehensive diagnostic script:

```bash
# Test basic connectivity
python examples/globus_compute_diagnostics.py -e your-endpoint-id -t connectivity

# Test PyTorch and Intel GPU
python examples/globus_compute_diagnostics.py -e your-endpoint-id -t torch

# Test MPI and GPU tile information
python examples/globus_compute_diagnostics.py -e your-endpoint-id -t mpi

# Test parallel execution
python examples/globus_compute_diagnostics.py -e your-endpoint-id -t parallel -n 12

# Run comprehensive tests
python examples/globus_compute_diagnostics.py -e your-endpoint-id -t comprehensive
```

### Option 2: Using the GlobusJobScheduler

```python
from src.agents.globus_job_scheduler import GlobusJobScheduler

# Create scheduler
scheduler = GlobusJobScheduler(endpoint_id="your-endpoint-id")

# Run diagnostic tests
simple_result = scheduler.run_diagnostic_test("simple")
torch_result = scheduler.run_diagnostic_test("torch")
parallel_result = scheduler.run_diagnostic_test("parallel")
```

### Option 3: Direct Globus Compute SDK Usage

```python
from globus_compute_sdk import Executor
from examples.torch_test import train_wrapper

# Test with your endpoint
ENDPOINT_ID = "your-endpoint-id"
hparams = {'lr': 0.01, 'batch': 32, 'optimizer': 'adam'}

with Executor(endpoint_id=ENDPOINT_ID) as executor:
    future = executor.submit(train_wrapper, hparams)
    result = future.result()
    print(result)
```

### Option 4: User Test Pattern

Run the exact pattern provided by the user:

```bash
# Update the endpoint ID in the script first
python examples/user_test_pattern.py
```

The diagnostic tests verify:
- **Basic connectivity** to the endpoint
- **PyTorch availability** and version
- **Intel GPU access** and device information
- **MPI rank and GPU tile** information
- **Parallel execution** capabilities
- **Tensor computation** performance

## Usage Examples

### Basic Job Submission

```python
from src.agents.globus_job_scheduler import GlobusJobScheduler

# Initialize scheduler
scheduler = GlobusJobScheduler(
    endpoint_id="your-endpoint-id",
    working_dir_config=config["working_dirs"]
)

# Submit job
job_config = {
    "job_id": "test_job_001",
    "model_type": "resnet18",
    "learning_rate": 0.001,
    "batch_size": 64,
    "num_epochs": 10
}

job_id = scheduler.submit_job(job_config)
```

### Running the Example

```bash
# Update the endpoint ID in the example
python examples/globus_compute_example.py
```

## Workflow Integration

The Globus Compute integration is fully integrated with existing workflows:

- **Main Workflow** (`src/agents/main.py`): Supports both PBS and Globus Compute
- **LangGraph Workflow** (`src/agents/langgraph_workflow.py`): Automatic scheduler selection
- **Advanced Workflow** (`src/agents/advanced_langgraph_workflow.py`): Full compatibility

## Comparison: PBS vs Globus Compute

| Feature | Direct PBS | Globus Compute |
|---------|------------|----------------|
| Authentication | SSH + MFA | Globus Auth |
| Error Handling | Manual | Automatic retry |
| Scalability | Limited | High |
| Monitoring | Basic | Enhanced |
| Setup Complexity | Medium | Low |
| Dependencies | SSH client | globus-compute-sdk |

## Troubleshooting

### Common Issues

1. **Endpoint Not Found**
   - Verify endpoint ID is correct
   - Check if endpoint is running: `globus-compute-endpoint list`

2. **Authentication Errors**
   - Run `globus auth login` to refresh credentials
   - Check client ID and secret for confidential clients

3. **Job Submission Failures**
   - Verify endpoint is active and healthy
   - Check Aurora resource availability
   - Review function serialization issues

4. **Environment Issues**
   - Ensure correct modules are loaded in `worker_init`
   - Verify Python environment and PYTHONPATH

### Debugging

Enable debug logging in your workflow:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Check endpoint logs:

```bash
globus-compute-endpoint logs aurora-training
```

## Performance Considerations

- **Function Registration**: Functions are registered once per scheduler instance
- **Serialization**: Large objects may impact performance
- **Concurrency**: Globus Compute handles multiple jobs efficiently
- **Timeouts**: Adjust `function_timeout` based on your job duration

## Migration from PBS

To migrate from direct PBS to Globus Compute:

1. **Setup Endpoint**: Follow the setup guide
2. **Update Configuration**: Enable Globus Compute in config.yaml
3. **Test Integration**: Run the example script
4. **Gradual Migration**: Keep PBS as fallback initially

## Best Practices

1. **Endpoint Management**: Monitor endpoint health regularly
2. **Error Handling**: Implement proper error handling in your functions
3. **Resource Management**: Configure appropriate resource limits
4. **Security**: Use appropriate authentication methods
5. **Monitoring**: Log job submission and completion events

## Support

For issues related to:
- **Globus Compute**: https://globus-compute.readthedocs.io/
- **Aurora Access**: Contact ALCF support
- **Workflow Issues**: Check project documentation

## Future Enhancements

Planned improvements for Globus Compute integration:

- **Multi-endpoint Support**: Support for multiple endpoints
- **Dynamic Scaling**: Automatic endpoint scaling based on workload
- **Advanced Monitoring**: Integration with monitoring services
- **Result Caching**: Caching of training results
- **Batch Operations**: Support for batch job operations

## References

- [Globus Compute Documentation](https://globus-compute.readthedocs.io/)
- [Aurora User Guide](https://docs.alcf.anl.gov/aurora/getting-started/)
- [Project Documentation](../README.md) 