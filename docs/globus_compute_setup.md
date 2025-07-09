# Globus Compute Setup on Aurora

This document provides instructions for setting up a Globus Compute endpoint on Aurora to execute training jobs as part of the agentic workflow.

## Overview

Globus Compute provides an alternative to direct PBS job submission by:
- Managing job submission and monitoring through the Globus Compute service
- Handling authentication and job queuing automatically
- Providing better error handling and retry mechanisms
- Offering a more robust API for job management

## Prerequisites

1. **Aurora Account**: You must have an active account on Aurora
2. **Globus Account**: You need a Globus account (free at https://www.globus.org)
3. **Python Environment**: Access to Python 3.8+ on Aurora

## Installation Steps

### 1. Install Globus Compute on Aurora

SSH into Aurora and install the Globus Compute endpoint:

```bash
# Connect to Aurora
ssh aurora

# Navigate to your project directory
cd /lus/flare/projects/datascience/parton/globus_compute/

# Load the frameworks module (if not already loaded)
module load frameworks

# Activate your virtual environment
source venvFrameworks/bin/activate

# Install globus-compute-endpoint
pip install globus-compute-endpoint
```

### 2. Configure the Endpoint

```bash
# Create a new endpoint configuration
globus-compute-endpoint configure aurora-training

# This creates a config directory at ~/.globus_compute/aurora-training/
```

### 3. Update Endpoint Configuration

Replace the generated configuration with our custom configuration:

```bash
# Copy our configuration to the endpoint directory
cp /path/to/your/repo/config/globus_endpoint.yaml ~/.globus_compute/aurora-training/config.yaml
```

### 4. Start the Endpoint

```bash
# Start the endpoint
globus-compute-endpoint start aurora-training

# Check endpoint status
globus-compute-endpoint list

# Get the endpoint ID (you'll need this for the main configuration)
globus-compute-endpoint show aurora-training
```

### 5. Update Main Configuration

Copy the endpoint ID from the previous step and update your main `config.yaml`:

```yaml
globus_compute:
  enabled: true  # Enable Globus Compute
  endpoint_id: "your-endpoint-id-here"  # Replace with actual endpoint ID
  endpoint_config_path: "config/globus_endpoint.yaml"
  auth_method: "native_client"
  function_timeout: 3600
  max_retries: 2
```

## Authentication Setup

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

# Check endpoint details
globus-compute-endpoint show aurora-training
```

### Debugging

```bash
# Enable debug logging
globus-compute-endpoint start aurora-training --log-level DEBUG

# Check for configuration issues
globus-compute-endpoint validate aurora-training
```

## Environment Setup

The endpoint configuration includes environment setup in the `worker_init` section:

```yaml
worker_init: |
  module load frameworks
  cd /lus/flare/projects/datascience/parton/globus_compute/
  source venvFrameworks/bin/activate
  export PYTHONPATH=/lus/flare/projects/datascience/parton/globus_compute/:$PYTHONPATH
  cd workflows
```

Modify this section to match your specific environment setup:

1. **Module Loading**: Ensure all required modules are loaded
2. **Virtual Environment**: Activate the correct Python environment
3. **Path Setup**: Set PYTHONPATH to include your project
4. **Working Directory**: Set the correct working directory

## Resource Configuration

The endpoint is configured with the following resources:

- **Nodes**: 1 node per block
- **CPUs**: 64 CPUs per node
- **GPUs**: 1 GPU per node
- **Queue**: debug queue
- **Walltime**: 1 hour
- **Account**: datascience

Modify these settings in `config/globus_endpoint.yaml` as needed for your requirements.

## Testing the Setup

After setting up the endpoint, test it with a simple function:

```python
from globus_compute_sdk import Client

# Create a client
client = Client()

# Define a simple test function
def hello_world():
    return "Hello from Aurora!"

# Register the function
function_id = client.register_function(hello_world)

# Submit a task
task_id = client.run(function_id, endpoint_id="your-endpoint-id")

# Get results
result = client.get_result(task_id)
print(result)
```

## Integration with Agentic Workflow

Once the endpoint is set up and tested:

1. **Enable Globus Compute** in your `config.yaml`
2. **Update Endpoint ID** with your actual endpoint ID
3. **Run the Workflow** using the GlobusJobScheduler

The workflow will automatically use Globus Compute instead of direct PBS submission.

## Troubleshooting

### Common Issues

1. **Authentication Failures**
   - Ensure you're logged in to Globus
   - Check that your credentials haven't expired
   - Verify client ID and secret (if using confidential client)

2. **Endpoint Not Starting**
   - Check PBS queue availability
   - Verify account permissions
   - Review endpoint logs for errors

3. **Job Submission Failures**
   - Verify endpoint is running
   - Check resource availability
   - Review function serialization issues

4. **Environment Issues**
   - Ensure all modules are loaded correctly
   - Verify Python environment activation
   - Check PYTHONPATH configuration

### Logs and Debugging

- **Endpoint Logs**: `~/.globus_compute/aurora-training/endpoint.log`
- **Worker Logs**: Check PBS job output files
- **Client Logs**: Enable debug logging in your Python client

## Security Considerations

1. **Endpoint Security**: The endpoint runs with your Aurora credentials
2. **Function Security**: Only submit trusted functions to the endpoint
3. **Data Security**: Ensure sensitive data is properly handled
4. **Network Security**: Globus Compute uses secure HTTPS connections

## Performance Optimization

1. **Prefetch Capacity**: Increase `prefetch_capacity` for better throughput
2. **Worker Scaling**: Adjust `max_workers_per_node` based on your workload
3. **Block Management**: Tune `init_blocks` and `max_blocks` for your needs
4. **Timeout Settings**: Adjust timeouts based on your job duration

## Maintenance

1. **Regular Restarts**: Restart the endpoint periodically to clear any issues
2. **Log Rotation**: Monitor and rotate logs to prevent disk space issues
3. **Updates**: Keep Globus Compute updated to the latest version
4. **Monitoring**: Set up monitoring for endpoint health and performance

For more information, consult the [Globus Compute Documentation](https://globus-compute.readthedocs.io/). 