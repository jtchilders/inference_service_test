# Inference Service Test

Testing Space for the Inference Service running on Sophia to demonstrate running Agents on one system (e.g. Crux), which get LLM inference from another system (e.g. Sophia), to make decisions about what to launch on another system (e.g. Aurora).

## Project Overview

This project implements an agentic workflow for hyperparameter optimization on the CIFAR-100 dataset using three ALCF systems:

- **Crux** (ALCF CPU supercomputer): Runs the main agent orchestrator
- **Sophia** (ALCF Nvidia DGX supercomputer): Provides LLM inference for hyperparameter suggestions  
- **Aurora** (ALCF GPU supercomputer): Runs the actual training jobs

The workflow uses an LLM agent to intelligently suggest hyperparameters based on previous training results, creating an automated hyperparameter optimization system.

## 🆕 LangGraph/LangChain Implementation

The project now includes a modern LangGraph/LangChain implementation that provides:

- **Structured State Management**: Automatic state persistence across workflow nodes
- **Conditional Routing**: Dynamic workflow paths based on conditions and results
- **Tool Integration**: LangChain tools for enhanced agent capabilities
- **Better Error Handling**: Structured error handling and recovery mechanisms
- **Extensibility**: Easy to add new nodes, tools, and workflow patterns

### New Workflow Options

1. **Basic LangGraph Workflow** (`src/agents/langgraph_workflow.py`)
   - Essential workflow with state management
   - Conditional routing based on convergence

2. **Advanced LangGraph Workflow** (`src/agents/advanced_langgraph_workflow.py`)
   - Enhanced workflow with additional features
   - Better error handling and monitoring

3. **LangChain Hyperparameter Agent** (`src/agents/langchain_hyperparam_agent.py`)
   - Agent with tools and structured outputs
   - Enhanced analysis capabilities

### Quick Start with LangGraph

```bash
# Run basic LangGraph workflow
python src/agents/langgraph_workflow.py --config config.yaml --max-iterations 10

# Run advanced LangGraph workflow
python src/agents/advanced_langgraph_workflow.py --config config.yaml --max-iterations 10

# Run examples
python examples/langgraph_example.py --config config.yaml --workflow all
```

For detailed documentation, see [LangGraph Implementation Guide](docs/langgraph_implementation.md).

## Architecture

```
┌─────────┐    ┌──────────┐    ┌─────────┐
│  Crux   │───▶│  Sophia  │───▶│ Aurora  │
│(Agent)  │    │  (LLM)   │    │(Training)│
└─────────┘    └──────────┘    └─────────┘
     │              │              │
     │              │              │
     ▼              ▼              ▼
  Orchestrates  Suggests      Runs Training
  Workflow      Hyperparams   Jobs
```

## Quick Start

### Prerequisites

- Python 3.8+
- Access to ALCF systems (Crux, Sophia, Aurora)
- Globus authentication for Sophia LLM service

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd inference_service_test
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download CIFAR-100 dataset:
```bash
./scripts/download_cifar100.sh
```

### Authentication Setup

The project uses Globus authentication for accessing the Sophia LLM service. Before running the workflows, you need to authenticate:

1. **First-time authentication**:
```bash
python src/utils/inference_auth_token.py authenticate
```

2. **Get access token** (for testing):
```bash
python src/utils/inference_auth_token.py get_access_token
```

The authentication tokens are automatically managed and refreshed as needed.

### Configuration

1. Create a configuration file `config.yaml`:
```yaml
# Sophia LLM service configuration
sophia:
  url: "https://data-portal-dev.cels.anl.gov/resource_server/sophia/vllm/v1"

# Aurora PBS job configuration
aurora:
  host: "aurora.alcf.anl.gov"
  user: "${AURORA_USER}"
  # Aurora uses MFA via mobile app, not SSH keys
  # SSH connections use ControlMaster to reuse existing authenticated sessions
  control_master: true
  pbs_template: "jobs/train_cifar100.pbs.template"

# Working directory configuration
# Hierarchical structure: {working_dir_base}/{launch_iteration_base}/{job_base}
working_dirs:
  # Base directory on Aurora for all workflow runs
  aurora_base: "/lus/eagle/projects/datascience/${AURORA_USER}/workflows"
  # Base directory for local results and logs
  local_base: "results"
  # Directory naming patterns
  launch_iteration_pattern: "{timestamp}_{experiment_id}"  # e.g., "20241201_143022_exp_001"
  job_pattern: "job_{job_id}"  # e.g., "job_exp_001_iter_0"

# Data and results configuration
data:
  dir: "data/cifar-100-python"

results:
  dir: "results"

# Workflow configuration
polling_interval: 60  # seconds
```

2. Set environment variables:
```bash
export AURORA_USER="your_aurora_username"
# Note: Aurora uses MFA via mobile app, not SSH keys
# SSH connections will use ControlMaster to reuse existing authenticated sessions
```

### Aurora Authentication Setup

Aurora uses MFA via mobile app instead of SSH keys. To enable automated job submission, you need to set up SSH ControlMaster:

1. **Establish initial connection** (requires MFA):
```bash
ssh -M aurora.alcf.anl.gov
```

2. **Keep this connection open** - subsequent SSH calls will reuse this authenticated session

3. **Configure SSH ControlMaster** (optional, for automatic setup):
```bash
# Add to ~/.ssh/config
Host aurora.alcf.anl.gov
    ControlMaster auto
    ControlPersist 10m
```

### Aurora Configuration

The project has been specifically configured for Aurora's Intel GPU architecture. Key features include:

- **Intel GPU Support**: Optimized for Aurora's Ponte Vecchio GPUs using Intel oneAPI
- **Module Configuration**: Proper Intel module loading (intel, intel-mpi, intel-mkl, pytorch)
- **Environment Variables**: Intel-specific settings for optimal performance
- **Device Detection**: Automatic detection and fallback between Intel GPU → CUDA → CPU

#### Testing Aurora Configuration

Before running training jobs, test the Aurora setup:

```bash
# On Aurora login node or in a test job
python test_aurora_config.py
```

This verifies:
- Intel module loading
- PyTorch Intel extension availability
- GPU device detection
- Environment variable configuration

#### Aurora-Specific Considerations

- **Intel GPUs**: Aurora uses Intel Ponte Vecchio GPUs, not NVIDIA
- **Memory Management**: Different memory characteristics than CUDA GPUs
- **Performance Tuning**: Intel-specific optimizations for best performance

For detailed Aurora configuration information, see [Aurora Configuration Guide](docs/aurora_configuration.md).

### Deployment to Aurora

The workflow now includes automatic deployment of the software repository and dataset to Aurora. This ensures that the training scripts and data are always up-to-date before running experiments.

#### Automatic Deployment (Recommended)

The LangGraph workflow automatically handles deployment:

```bash
# The workflow will automatically deploy before starting experiments
python src/agents/langgraph_workflow.py --config config.yaml --max-iterations 10
```

The workflow includes a `deploy_environment` node that:
1. **Checks repository status** on Aurora
2. **Clones or updates** the git repository
3. **Downloads CIFAR-100 dataset** if not present
4. **Verifies deployment** before proceeding

#### Manual Deployment

You can also deploy manually using the deployment agent:

```bash
# Deploy everything (repository + dataset)
python src/agents/deployment_agent.py --config config.yaml --action all

# Deploy only repository
python src/agents/deployment_agent.py --config config.yaml --action repo

# Deploy only dataset
python src/agents/deployment_agent.py --config config.yaml --action dataset

# Verify deployment
python src/agents/deployment_agent.py --config config.yaml --action verify
```

#### Bash Script Deployment

For quick deployment without Python dependencies:

```bash
# Make script executable (first time only)
chmod +x scripts/deploy_to_aurora.sh

# Run deployment
./scripts/deploy_to_aurora.sh
```

This script:
- Automatically detects git repository and branch
- Creates base directory on Aurora
- Clones/updates the repository
- Downloads CIFAR-100 dataset
- Verifies the deployment

#### Repository Configuration

Configure repository settings in `config.yaml`:

```yaml
# Repository configuration for deployment
# If not specified, will use current git repository
repository:
  url: "https://github.com/your-org/inference_service_test.git"  # Optional: auto-detected
  branch: "main"  # Optional: auto-detected
```

If not specified, the deployment agent will automatically detect:
- Current git repository URL
- Current branch name

#### Deployment Verification

The deployment process verifies:
- ✅ Repository exists and is up-to-date
- ✅ Source code structure is intact
- ✅ Training script is present
- ✅ CIFAR-100 dataset is downloaded
- ✅ Python environment is available
- ✅ Requirements.txt exists

#### Deployment Logs

Deployment results are included in the final experiment report:

```json
{
  "experiment_id": "20241201_143022_exp_001",
  "deployment_status": {
    "repository": {
      "status": "success",
      "action": "updated",
      "repo_url": "https://github.com/your-org/inference_service_test.git",
      "repo_branch": "main"
    },
    "dataset": {
      "status": "success", 
      "action": "exists",
      "data_dir": "/lus/eagle/projects/datascience/user/workflows/inference_service_test/data/cifar-100-python"
    },
    "verification": {
      "status": "success",
      "repository": "ok",
      "source_code": "ok",
      "training_script": "ok",
      "dataset": "ok"
    }
  }
}
```

### Testing the Integration

Before running the full workflow, test the Sophia integration:

```bash
python test_sophia_integration.py
```

This will verify that:
- Authentication is working correctly
- LangChain integration with Sophia is functional
- Hyperparameter agent can communicate with the LLM service

### Running the Workflow

#### Original Implementation
```bash
python src/agents/main.py --config config.yaml --max-iterations 10
```

#### New LangGraph Implementation
```bash
# Basic workflow
python src/agents/langgraph_workflow.py --config config.yaml --max-iterations 10

# Advanced workflow
python src/agents/advanced_langgraph_workflow.py --config config.yaml --max-iterations 10
```

2. Monitor progress:
```bash
tail -f logs/agent_workflow.log
# or for LangGraph workflows:
tail -f logs/langgraph_workflow.log
```

3. View results:
```bash
ls results/
cat results/experiment_report_*.json
```

### Hierarchical Working Directory Structure

The system now uses a hierarchical directory structure to organize workflow runs:

```
{working_dir_base}/{launch_iteration_base}/{job_base}
```

#### Local Structure (on Crux)
```
results/
├── 20241201_143022_exp_001/          # Launch iteration directory
│   ├── logs/                         # Experiment-level logs
│   ├── checkpoints/                  # Experiment-level checkpoints
│   ├── tensorboard/                  # Experiment-level TensorBoard logs
│   ├── reports/                      # Experiment reports
│   ├── job_exp_001_iter_0/           # Job-specific directory
│   │   ├── output/                   # Training outputs
│   │   ├── logs/                     # Job-specific logs
│   │   ├── checkpoints/              # Model checkpoints
│   │   └── tensorboard/              # Training curves
│   └── job_exp_001_iter_1/           # Another job
│       └── ...
└── 20241201_150045_exp_002/          # Another launch iteration
    └── ...
```

#### Remote Structure (on Aurora)
```
/lus/eagle/projects/datascience/{user}/workflows/
├── 20241201_143022_exp_001/          # Launch iteration directory
│   ├── job_exp_001_iter_0/           # Job-specific directory
│   │   ├── job.pbs                   # PBS script
│   │   ├── results.json              # Training results
│   │   ├── job_exp_001_iter_0.out    # PBS stdout
│   │   └── job_exp_001_iter_0.err    # PBS stderr
│   └── job_exp_001_iter_1/           # Another job
│       └── ...
└── 20241201_150045_exp_002/          # Another launch iteration
    └── ...
```

#### Configuration

The directory structure is configurable in `config.yaml`:

```yaml
working_dirs:
  aurora_base: "/lus/eagle/projects/datascience/${AURORA_USER}/workflows"
  local_base: "results"
  launch_iteration_pattern: "{timestamp}_{experiment_id}"
  job_pattern: "job_{job_id}"
```

#### Benefits

- **Organization**: Each workflow run gets its own timestamped directory
- **Isolation**: Jobs within the same experiment are grouped together
- **Cleanup**: Easy to identify and remove old experiments
- **Scalability**: Supports multiple concurrent experiments
- **Debugging**: Clear separation of logs and outputs by job

#### Testing the Directory Structure

Test the new directory structure:

```bash
python test_working_dirs.py
```

This will create sample directories and demonstrate the hierarchical structure.

## Project Structure

```
inference_service_test/
├── README.md                 # Project overview and quickstart
├── .gitignore               # Ignore data, logs, .pyc, etc.
├── requirements.txt         # Python dependencies
├── cursor_rules.txt         # Cursor IDE rules and context
├── data/                    # CIFAR-100 dataset storage
├── scripts/
│   ├── download_cifar100.sh # Download/unpack CIFAR-100
│   └── deploy_to_aurora.sh  # 🆕 Aurora deployment script
├── jobs/
│   └── train_cifar100.pbs.template  # PBS script template
├── src/
│   ├── agents/
│   │   ├── main.py                    # Original orchestrator (runs on Crux)
│   │   ├── hyperparam_agent.py        # Original LLM agent (talks to Sophia)
│   │   ├── langgraph_workflow.py      # 🆕 Basic LangGraph workflow
│   │   ├── advanced_langgraph_workflow.py  # 🆕 Advanced LangGraph workflow
│   │   ├── langchain_hyperparam_agent.py   # 🆕 LangChain hyperparameter agent
│   │   ├── deployment_agent.py        # 🆕 Aurora deployment agent
│   │   └── job_scheduler.py           # PBS job manager (SSH to Aurora)
│   ├── training/
│   │   ├── train.py         # Training entry point
│   │   ├── model.py         # PyTorch model definitions
│   │   └── trainer.py       # Training loop and logging
│   └── utils/
│       ├── data_utils.py    # CIFAR-100 dataset loading, result parsing, and analysis utilities
│       ├── metrics.py       # AUC calculation, learning curve analysis, and other evaluation metrics
│       ├── working_dirs.py  # 🆕 Hierarchical working directory management for local and remote storage
│       └── inference_auth_token.py  # Authentication token management for Sophia LLM service
├── examples/
│   └── langgraph_example.py # 🆕 LangGraph usage examples
├── docs/
│   └── langgraph_implementation.md  # 🆕 LangGraph documentation
└── notebooks/
    └── exploration.ipynb    # Local prototyping and testing
```

## Components

### Agents (`src/agents/`)

#### Original Implementation
- **`main.py`**: Orchestrates the entire workflow, coordinates between hyperparameter agent and job scheduler
- **`hyperparam_agent.py`**: Communicates with Sophia's LLM service to suggest hyperparameters based on previous results
- **`job_scheduler.py`**: Manages SSH connections to Aurora and PBS job submission/monitoring

#### 🆕 LangGraph Implementation
- **`langgraph_workflow.py`**: Basic LangGraph workflow with state management and conditional routing
- **`advanced_langgraph_workflow.py`**: Advanced workflow with enhanced error handling and monitoring
- **`langchain_hyperparam_agent.py`**: LangChain-based agent with tools and structured outputs
- **`deployment_agent.py`**: 🆕 Aurora deployment agent for repository and dataset setup

### Training (`src/training/`)

- **`train.py`**: Entry point for individual training runs, loads hyperparameters and kicks off training
- **`model.py`**: PyTorch model definitions including ResNet, VGG, DenseNet, and custom CNN architectures
- **`trainer.py`**: Training loop with validation, logging, and checkpointing

### Utils (`src/utils/`)

- **`data_utils.py`**: CIFAR-100 dataset loading, result parsing, and analysis utilities
- **`metrics.py`**: AUC calculation, learning curve analysis, and other evaluation metrics
- **`working_dirs.py`**: 🆕 Hierarchical working directory management for local and remote storage
- **`inference_auth_token.py`**: Authentication token management for Sophia LLM service

## Usage Examples

### Local Testing

Use the Jupyter notebook for local prototyping:
```bash
jupyter notebook notebooks/exploration.ipynb
```

### Single Training Run

Test a single training job:
```bash
python src/training/train.py \
  --model_type resnet18 \
  --learning_rate 0.001 \
  --batch_size 128 \
  --num_epochs 50 \
  --output_dir test_output \
  --job_id test_run
```

### Hyperparameter Optimization

#### Original Implementation
```bash
python src/agents/main.py \
  --config config.yaml \
  --max-iterations 20 \
  --log-level INFO
```

#### 🆕 LangGraph Implementation
```bash
# Basic workflow
python src/agents/langgraph_workflow.py \
  --config config.yaml \
  --max-iterations 20 \
  --log-level INFO

# Advanced workflow
python src/agents/advanced_langgraph_workflow.py \
  --config config.yaml \
  --max-iterations 20 \
  --log-level INFO

# Run examples
python examples/langgraph_example.py \
  --config config.yaml \
  --workflow all \
  --max-iterations 5
```

## Model Architectures

The project supports multiple model architectures:

- **ResNet**: ResNet18, ResNet34, ResNet50
- **VGG**: VGG16
- **DenseNet**: DenseNet121
- **Custom CNN**: Configurable architecture with custom layers

## Hyperparameters

The LLM agent optimizes the following hyperparameters:

- **Model**: Architecture type, hidden size, number of layers
- **Training**: Learning rate, batch size, number of epochs, weight decay
- **Regularization**: Dropout rate

## Metrics

The system tracks and optimizes:

- **Final Accuracy**: Test set accuracy
- **AUC**: Area under the accuracy curve
- **Training Time**: Time efficiency
- **Convergence**: How quickly the model converges

## Monitoring and Debugging

### Logs

- `logs/agent_workflow.log`: Main workflow logs
- Job-specific logs in `results/job_*/`

### TensorBoard

Training progress is logged to TensorBoard:
```bash
tensorboard --logdir results/job_*/tensorboard
```

### Results Analysis

Analyze all results:
```python
from src.utils.data_utils import create_data_summary
create_data_summary("results", "analysis_report.json")
```

## Troubleshooting

### Common Issues

1. **SSH Connection Failed**: 
   - Ensure you have an active SSH session to Aurora with ControlMaster
   - Run: `ssh -M -S ~/.ssh/aurora_control_%h_%p_%r aurora.alcf.anl.gov`
   - Keep this connection open while running the workflow

2. **ControlMaster Socket Not Found**:
   - Check if the ControlMaster socket exists: `ls -la ~/.ssh/control*`
   - If not found, establish a new ControlMaster connection
   - Verify SSH config has correct ControlMaster settings

3. **LLM Service Unavailable**: Verify Sophia API key and service status
4. **PBS Job Failed**: Check PBS script template and Aurora queue settings
5. **Out of Memory**: Reduce batch size or model complexity

### Aurora Authentication Issues

If you encounter authentication problems with Aurora:

1. **Check ControlMaster Status**:
   ```bash
   # Check if ControlMaster socket exists
   ls -la ~/.ssh/control*
   
   # Test SSH connection
   ssh -o ControlMaster=no aurora.alcf.anl.gov "echo 'Connection successful'"
   ```

2. **Re-establish ControlMaster Connection**:
   ```bash
   # Close existing connection if needed
   ssh -O exit aurora.alcf.anl.gov
   
   # Create new ControlMaster connection (requires MFA)
   ssh -M aurora.alcf.anl.gov
   ```

3. **Configure SSH for Automatic ControlMaster** (optional):
   ```bash
   # Add to ~/.ssh/config
   Host aurora.alcf.anl.gov
       ControlMaster auto
       ControlPersist 10m
   ```

## Contributing

1. Follow the code style guidelines in `cursor_rules.txt`
2. Use 3-space indentation
3. Include type hints and comprehensive docstrings
4. Test changes locally before deployment

## License

[Add your license information here]

## Acknowledgments

- ALCF for providing access to Crux, Sophia, and Aurora
- PyTorch and torchvision for the deep learning framework
- The CIFAR-100 dataset creators

### Debug Mode

Run with debug logging:
```bash
python src/agents/main.py --config config.yaml --log-level DEBUG
```
