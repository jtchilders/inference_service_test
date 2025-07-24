# 🚀 Agentic Hyperparameter Optimization - Quick Start Guide

**Intelligent AI-driven hyperparameter optimization for CIFAR-100 AUC maximization using Globus Compute on Aurora**

---

## 📋 Overview

This system uses AI agents (LLMs) to intelligently explore hyperparameter space and find the best configurations for maximizing AUC (Area Under the Learning Curve) on CIFAR-100 image classification. The workflow combines:

- 🤖 **AI Agent Intelligence**: LLM analyzes results and suggests promising hyperparameters
- 🚀 **Parallel Execution**: Multiple training jobs run simultaneously on Aurora GPUs
- 📊 **Real-time Monitoring**: Live statistics, plots, and progress tracking
- 🎯 **AUC Optimization**: Focus on maximizing accuracy × training efficiency

---

## ⚡ Quick Start (5 minutes)

### 1. **Prerequisites**

```bash
# Ensure you have access to:
# - Aurora supercomputer
# - Sophia LLM service  
# - Working Globus Compute endpoint on Aurora

# Check your Globus Compute endpoint
python examples/test_globus_training.py --endpoint-id YOUR_ENDPOINT_ID --num-jobs 2 --epochs 3
```

### 2. **Install Dependencies**

```bash
pip install globus-compute-sdk langchain-openai langchain-core langgraph matplotlib seaborn scikit-learn scipy
```

### 3. **Authenticate with Sophia**

```bash
python src/utils/inference_auth_token.py authenticate --force
```

### 4. **Run Simple Example**

```bash
# Quick test (fast, 20 iterations)
python examples/agentic_hyperparameter_search.py --endpoint YOUR_ENDPOINT_ID --quick-test

# Standard optimization (100 iterations)
python examples/agentic_hyperparameter_search.py --endpoint YOUR_ENDPOINT_ID
```

### 5. **Check Results**

Results are saved in `results/agentic_optimization_YYYYMMDD_HHMMSS/`:
- 📊 **Plots**: AUC progression, parameter correlations, exploration heatmaps
- 📝 **Logs**: Detailed optimization progress and job status
- 🏆 **Final Report**: Best hyperparameters and performance summary

---

## 🎯 Production Usage

### Configuration Setup

1. **Copy and edit the production config:**
```bash
cp config_agentic_production.yaml my_config.yaml
# Edit my_config.yaml:
# - Set your Globus Compute endpoint ID
# - Adjust max_iterations, parallel jobs, etc.
```

2. **Run production optimization:**
```bash
python run_agentic_optimization.py --config my_config.yaml
```

### Key Configuration Options

```yaml
optimization:
  max_iterations: 100           # Number of optimization rounds
  convergence_patience: 15      # Stop if no improvement for N iterations

globus_compute:
  parallel_jobs:
    max_concurrent: 12          # Aurora has 6 GPUs × 2 tiles = 12 max
    initial_batch_size: 8       # Start with 8 parallel jobs
    adaptive_scaling: true      # Adjust based on system performance
```

---

## 🤖 How the AI Agent Works

### Exploration → Exploitation Strategy

1. **Initial Sampling (20 iterations)**: Random exploration across parameter space
2. **AI-Guided Exploration (70% exploration)**: LLM suggests diverse parameters
3. **Progressive Exploitation (→ 20% exploration)**: Focus on high-AUC regions
4. **Convergence Detection**: Stop when no significant AUC improvement

### LLM Intelligence Features

- 📈 **Result Analysis**: Analyzes previous training outcomes
- 🎯 **Pattern Recognition**: Identifies which parameters correlate with high AUC
- 🧠 **Context Awareness**: Considers training time, convergence patterns
- 📊 **Statistical Guidance**: Uses confidence intervals and trend analysis

---

## 📊 Real-time Monitoring

### Console Output
```
📊 OPTIMIZATION PROGRESS UPDATE 📊
Time Elapsed: 0:45:23
Iteration: 15/100

🎯 Job Statistics:
   Active: 8
   Completed: 67
   Success Rate: 94.0%

🏆 Performance:
   Best AUC: 0.847291
   Recent AUC: 0.823456 ± 0.034567
   Exploration Factor: 0.45
```

### Generated Plots
- **AUC Progression**: Best AUC over time
- **Parameter Correlation**: Which parameters affect AUC
- **Exploration Heatmap**: Parameter space coverage
- **Convergence Analysis**: Training efficiency patterns

---

## 🛠 Advanced Usage

### Command Line Options

```bash
# Production optimization
python run_agentic_optimization.py \
  --config config_agentic_production.yaml \
  --max-iterations 50 \
  --max-concurrent 8 \
  --log-level DEBUG

# Resume interrupted experiment
python run_agentic_optimization.py \
  --config config_agentic_production.yaml \
  --resume experiment_20241201_143022

# Validation only (no training)
python run_agentic_optimization.py \
  --config config_agentic_production.yaml \
  --dry-run
```

### Custom Parameter Spaces

Edit `search_space` in your config:

```yaml
search_space:
  model_type:
    values: ["resnet18", "resnet34", "resnet50"]
    weights: [0.5, 0.3, 0.2]  # Prefer simpler models
  
  learning_rate:
    type: "log_uniform"
    min: 1e-5
    max: 1e-2
    initial_suggestions: [1e-4, 1e-3, 1e-2]
  
  num_epochs:
    type: "int_uniform"
    min: 10
    max: 100
    adaptive: true  # LLM can suggest based on convergence
```

---

## 🔧 Troubleshooting

### Common Issues

**❌ Authentication Failed**
```bash
# Re-authenticate with Sophia
python src/utils/inference_auth_token.py authenticate --force
```

**❌ Globus Compute Connection Issues**
```bash
# Test your endpoint
python examples/test_globus_training.py --endpoint-id YOUR_ENDPOINT_ID --num-jobs 1 --epochs 3
```

**❌ Jobs Failing on Aurora**
```bash
# Check Aurora access and data paths
# Verify CIFAR-100 data is available at: /lus/flare/projects/datascience/parton/data
```

**❌ Missing Dependencies**
```bash
# Install all required packages
pip install -r requirements.txt
```

### Debug Mode

```bash
# Enable detailed logging
python run_agentic_optimization.py --config my_config.yaml --log-level DEBUG
```

---

## 📈 Performance Tips

### Optimize for Speed
- Use `quick_test` mode for fast iterations
- Reduce `max_concurrent` if jobs are failing
- Set shorter `max_epochs` for initial exploration

### Optimize for Quality
- Increase `max_iterations` to 200+
- Use broader parameter ranges
- Enable all plot types for detailed analysis

### Optimize for Efficiency
- Enable `adaptive_scaling` 
- Use early stopping in training
- Set appropriate convergence criteria

---

## 📁 Output Structure

```
results/agentic_optimization_20241201_143022/
├── plots/                          # Real-time updated plots
│   ├── auc_progress_143045.png     # AUC over time
│   ├── hyperparam_correlation_143105.png
│   └── exploration_heatmap_143125.png
├── checkpoint.json                 # Resumable state
├── final_report.json              # Complete results
└── logs/                          # Detailed logs
    ├── agentic_optimization_20241201_143022.log
    ├── hyperparameter_agent.log
    └── job_scheduler.log
```

---

## 🎯 Example Results

After a successful optimization, you'll see:

```
🎉 OPTIMIZATION COMPLETED SUCCESSFULLY! 🎉

🏆 BEST RESULTS:
   Best AUC: 0.876543
   Mean AUC: 0.834567 ± 0.045123
   AUC Improvement: +0.142891

🎯 BEST HYPERPARAMETERS:
{
   "model_type": "resnet34",
   "learning_rate": 0.003456,
   "batch_size": 128,
   "num_epochs": 47,
   "dropout_rate": 0.234,
   "weight_decay": 0.0001234
}

📊 JOB STATISTICS:
   Total Jobs: 127
   Successful: 119
   Success Rate: 93.7%
```

---

## 🚀 Next Steps

1. **Use Best Parameters**: Apply the discovered hyperparameters to your own training
2. **Analyze Patterns**: Study the plots to understand what worked
3. **Experiment**: Try different parameter ranges or model architectures
4. **Scale Up**: Run longer optimizations for even better results

---

## 📞 Support

- **Issues**: Check logs in `results/*/logs/` for error details
- **Configuration**: Review `config_agentic_production.yaml` for options
- **Examples**: Run `examples/agentic_hyperparameter_search.py --help`
- **Testing**: Use `--dry-run` to validate setup without training

---

**🎉 Happy optimizing! The AI agent is ready to find your best CIFAR-100 hyperparameters! 🎉** 