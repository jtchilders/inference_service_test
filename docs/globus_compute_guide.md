# Globus Compute Guide for Aurora Training

This guide provides complete instructions for setting up and using Globus Compute to execute CIFAR-100 training jobs on Aurora as part of the agentic hyperparameter optimization workflow.

## Overview

### What is Globus Compute?

Globus Compute provides an alternative to direct PBS job submission for running training jobs on Aurora. The agentic workflow supports two job scheduling methods:

1. **Direct PBS Submission** (default): Direct SSH-based PBS job submission to Aurora
2. **Globus Compute** (recommended): Service-based job submission using Globus Compute endpoints

### Benefits of Globus Compute

- **Simplified Authentication**: No need to manage SSH keys or MFA sessions
- **Better Error Handling**: Automatic retries and robust error recovery
- **Service Management**: Jobs are managed by the Globus Compute service
- **Scalability**: Better support for concurrent job submission
- **Monitoring**: Enhanced job monitoring and status tracking
- **Reliability**: Robust API for job management with automatic queuing

### When to Use Each Approach

| Feature | Direct PBS | Globus Compute |
|---------|------------|----------------|
| Authentication | SSH + MFA | Globus Auth |
| Error Handling | Manual | Automatic retry |
| Scalability | Limited | High |
| Monitoring | Basic | Enhanced |
| Setup Complexity | Medium | Low |
| Dependencies | SSH client | globus-compute-sdk |

## Prerequisites

Before setting up Globus Compute, ensure you have:

1. **Aurora Account**: Active account on Aurora with access to compute resources
2. **Globus Account**: Free account at https://www.globus.org
3. **Python Environment**: Python 3.8+ environment on Aurora
4. **Project Access**: Access to your project directory on Aurora

## Step 1: Repository Setup on Aurora

### 1.1 Clone the Repository

SSH into Aurora and set up the repository:

```bash
# Connect to Aurora
ssh aurora

# Navigate to your project directory (adjust path as needed)
cd /lus/flare/projects/datascience/parton/

# Clone the repository
git clone https://github.com/your-org/inference_service_test.git
cd inference_service_test
```

### 1.2 Set Up Python Environment

```bash
# Load the frameworks module
module load frameworks

# Create or activate virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install globus-compute-endpoint
pip install globus-compute-endpoint
```

### 1.3 Download CIFAR-100 Data

```bash
# Make the download script executable and run it
chmod +x scripts/download_cifar100.sh
./scripts/download_cifar100.sh
```

This script will:
- Create the `data/` directory
- Download the CIFAR-100 dataset
- Extract it to `data/cifar-100-python/`
- Verify the download

## Step 2: Globus Compute Endpoint Setup

### 2.1 Configure the Endpoint

Create and configure a new Globus Compute endpoint:

```bash
# Create endpoint configuration
globus-compute-endpoint configure aurora-training

# This creates ~/.globus_compute/aurora-training/
```

### 2.2 Copy Configuration

Replace the generated configuration with the project's optimized configuration:

```bash
# Copy the project configuration
cp config/globus_endpoint.yaml ~/.globus_compute/aurora-training/config.yaml
```

### 2.3 Customize Configuration (if needed)

Edit the configuration file to match your environment:

```bash
nano ~/.globus_compute/aurora-training/config.yaml
```

Key settings to verify/adjust:

```yaml
# Account and queue settings
provider:
  account: datascience          # Your Aurora account
  queue: debug                  # Queue to use (debug, prod, etc.)

# Working directory setup
worker_init: |
  module load frameworks
  cd /lus/flare/projects/datascience/parton/  # Adjust to your project path
  source venv/bin/activate
  export PYTHONPATH=/lus/flare/projects/datascience/parton/inference_service_test:$PYTHONPATH
  cd inference_service_test
```

### 2.4 Start the Endpoint

```bash
# Start the endpoint
globus-compute-endpoint start aurora-training

# Check status
globus-compute-endpoint list

# Get the endpoint ID (save this for configuration)
globus-compute-endpoint show aurora-training
```

The output will show your endpoint ID, which looks like: `xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx`

## Step 3: Authentication Setup

### Option 1: Native Client Authentication (Recommended)

This is the default and most user-friendly option:

```bash
# The first time you run a job, you'll be prompted to authenticate
# Follow the browser-based authentication flow
```

### Option 2: Confidential Client Authentication

For automated/production environments:

```bash
# Set up client credentials (requires registration with Globus)
export GLOBUS_COMPUTE_CLIENT_ID="your-client-id"
export GLOBUS_COMPUTE_CLIENT_SECRET="your-client-secret"
```

## Step 4: Configuration Integration

### 4.1 Update Main Configuration

Create or update your `config.yaml` to enable Globus Compute:

```yaml
globus_compute:
  enabled: true  # Set to true to enable Globus Compute
  endpoint_id: "your-endpoint-id-here"  # Replace with actual endpoint ID from Step 2.4
  endpoint_config_path: "config/globus_endpoint.yaml"
  auth_method: "native_client"
  function_timeout: 3600  # 1 hour timeout
  max_retries: 2

# Working directories configuration
working_dirs:
  aurora_base: "/lus/flare/projects/datascience/parton/workflows"
  local_base: "results"
  launch_iteration_pattern: "{timestamp}_{experiment_id}"
  job_pattern: "job_{job_id}"

# Data configuration
data:
  dir: "data/cifar-100-python"
```

### 4.2 Install Client Dependencies

On your local machine (where you'll run the workflow):

```bash
pip install globus-compute-sdk
```

## Step 5: Testing Your Setup

### 5.1 Basic Connectivity Test

Test basic endpoint connectivity:

```python
from globus_compute_sdk import Client

# Create a client
client = Client()

# Define a simple test function
def hello_world():
    import socket
    return f"Hello from {socket.gethostname()}!"

# Register and run
function_id = client.register_function(hello_world)
task_id = client.run(function_id=function_id, endpoint_id="your-endpoint-id")
result = client.get_result(task_id)
print(result)
```

### 5.2 Training Validation Test

Use the training validation script to test the complete pipeline:

```bash
# Run parallel training test with 5 jobs
python examples/test_globus_training.py --endpoint-id your-endpoint-id --num-jobs 5 --epochs 3
```

This will:
- Generate random hyperparameter combinations
- Submit parallel training jobs
- Monitor progress and display results
- Validate that training completed successfully

### 5.3 Comprehensive Diagnostics

For thorough testing, use the diagnostic script:

```bash
# Run comprehensive diagnostics
python examples/globus_compute_diagnostics.py -e your-endpoint-id -t comprehensive
```

## Step 6: Running the Agentic Workflow

Once Globus Compute is set up and tested:

```bash
# Run the main agentic workflow
python -m src.agents.main --config config.yaml
```

The workflow will automatically use Globus Compute when enabled in the configuration.

## Endpoint Management

### Starting and Stopping

```bash
# Start the endpoint
globus-compute-endpoint start aurora-training

# Stop the endpoint
globus-compute-endpoint stop aurora-training

# Restart the endpoint
globus-compute-endpoint restart aurora-training
```

### Monitoring

```bash
# Check endpoint status
globus-compute-endpoint list

# View endpoint logs
globus-compute-endpoint logs aurora-training

# Check endpoint details and get ID
globus-compute-endpoint show aurora-training
```

### Debugging

```bash
# Enable debug logging
globus-compute-endpoint start aurora-training --log-level DEBUG

# Check for configuration issues
globus-compute-endpoint validate aurora-training
```

## Troubleshooting

### Common Issues

#### 1. Authentication Failures
- Ensure you're logged in to Globus: `globus auth login`
- Check that your credentials haven't expired
- Verify client ID and secret (if using confidential client)

#### 2. Endpoint Not Starting
- Check PBS queue availability
- Verify account permissions and resource limits
- Review endpoint logs: `globus-compute-endpoint logs aurora-training`
- Check that all modules load correctly in `worker_init`

#### 3. Job Submission Failures
- Verify endpoint is running: `globus-compute-endpoint list`
- Check Aurora resource availability
- Review function serialization issues
- Ensure CIFAR-100 data is available

#### 4. Environment Issues
- Verify all modules are loaded correctly in `worker_init`
- Check Python environment activation
- Ensure PYTHONPATH includes the project directory
- Verify data directory exists: `data/cifar-100-python/`

#### 5. Data Issues
- Re-run the download script: `./scripts/download_cifar100.sh`
- Check data directory permissions
- Verify CIFAR-100 files are present in `data/cifar-100-python/`

### Logs and Debugging

- **Endpoint Logs**: `~/.globus_compute/aurora-training/endpoint.log`
- **Worker Logs**: Check PBS job output files in Aurora's scratch space
- **Client Logs**: Enable debug logging in your Python client

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Performance Optimization

1. **Prefetch Capacity**: Increase `prefetch_capacity` in endpoint config for better throughput
2. **Worker Scaling**: Adjust `max_workers_per_node` based on your workload
3. **Block Management**: Tune `init_blocks` and `max_blocks` for your needs
4. **Timeout Settings**: Adjust timeouts based on your job duration

## Security Considerations

1. **Endpoint Security**: The endpoint runs with your Aurora credentials
2. **Function Security**: Only submit trusted functions to the endpoint
3. **Data Security**: Ensure sensitive data is properly handled
4. **Network Security**: Globus Compute uses secure HTTPS connections

## Maintenance

1. **Regular Restarts**: Restart the endpoint periodically to clear any issues
2. **Log Rotation**: Monitor and rotate logs to prevent disk space issues
3. **Updates**: Keep Globus Compute updated: `pip install --upgrade globus-compute-endpoint`
4. **Monitoring**: Set up monitoring for endpoint health and performance

## Integration with Agents

The Globus Compute integration is fully integrated with existing workflows:

- **Main Workflow** (`src/agents/main.py`): Supports both PBS and Globus Compute
- **LangGraph Workflow** (`src/agents/langgraph_workflow.py`): Automatic scheduler selection
- **Advanced Workflow** (`src/agents/advanced_langgraph_workflow.py`): Full compatibility

The workflow automatically chooses between PBS and Globus Compute based on your configuration.

## Reference

### Configuration Options

```yaml
globus_compute:
  enabled: false               # Set to true to enable
  endpoint_id: "endpoint-id"   # Required: your endpoint ID
  endpoint_config_path: "config/globus_endpoint.yaml"
  auth_method: "native_client" # Authentication method
  function_timeout: 3600       # Function timeout in seconds
  max_retries: 2              # Maximum retry attempts
```

### Authentication Methods

1. **native_client** (recommended): Interactive browser-based authentication
2. **confidential_client**: For automated environments with client credentials
3. **client_credentials**: For service-to-service authentication

## Support and Documentation

For additional help:
- **Globus Compute**: https://globus-compute.readthedocs.io/
- **Aurora User Guide**: https://docs.alcf.anl.gov/aurora/getting-started/
- **ALCF Support**: Contact ALCF support for Aurora-specific issues
- **Project Issues**: Check project documentation and GitHub issues

## Next Steps

After completing this setup:

1. **Test the pipeline** with `examples/test_globus_training.py`
2. **Run diagnostics** to verify everything works
3. **Configure your agentic workflow** with the new endpoint ID
4. **Monitor performance** and adjust configuration as needed 