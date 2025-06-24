# Aurora Configuration Guide

This document describes the Aurora-specific configuration for running PyTorch training jobs on the Aurora supercomputer at ALCF.

## Overview

Aurora uses Intel GPUs (Ponte Vecchio) and Intel oneAPI, which requires specific configuration for optimal PyTorch performance. The PBS template and training script have been updated to properly handle Aurora's architecture with Intel PyTorch Extension (IPEX) for maximum performance.

## Key Changes

### PBS Template Updates (`jobs/train_cifar100.pbs.template`)

1. **Module Loading**: Updated to load Aurora-specific modules:
   ```bash
   module load frameworks
   ```

2. **Environment Variables**: Added Intel oneAPI and GPU configuration:
   ```bash
   # Intel oneAPI settings
   export OMP_NUM_THREADS=1
   export OMP_PLACES=cores
   export OMP_PROC_BIND=close
   export I_MPI_PIN_DOMAIN=auto:compact
   export I_MPI_PIN_ORDER=compact
   
   # Intel GPU settings
   export ZE_AFFINITY_MASK=0.0
   export SYCL_DEVICE_FILTER=level_zero:gpu
   export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
   export PYTORCH_DEVICE=level_zero:gpu
   ```

3. **GPU Detection**: Added SYCL device listing for debugging:
   ```bash
   echo "GPU Info:"
   sycl-ls 2>/dev/null || echo "SYCL devices not available"
   ```

### Training Script Updates (`src/training/train.py`)

1. **Intel PyTorch Extension (IPEX) Integration**: Added proper IPEX import and optimization:
   ```python
   # Import Intel PyTorch Extension (IPEX) for Aurora optimization
   try:
      import intel_extension_for_pytorch as ipex
      IPEX_AVAILABLE = True
   except ImportError:
      IPEX_AVAILABLE = False
   ```

2. **Model and Optimizer Optimization**: Automatic IPEX optimization for Intel GPUs:
   ```python
   def optimize_model_for_intel(model, optimizer, device_type: str):
      if not IPEX_AVAILABLE or device_type != "xpu":
         return model, optimizer
      
      try:
         model, optimizer = ipex.optimize(model, optimizer=optimizer)
         return model, optimizer
      except Exception as e:
         logging.warning(f"Failed to optimize with IPEX: {e}")
         return model, optimizer
   ```

3. **Enhanced Device Detection**: Prioritizes Intel GPUs with IPEX support:
   - First checks for Intel GPU with IPEX support
   - Falls back to Intel GPU without IPEX
   - Falls back to CUDA if available
   - Uses CPU as final fallback

4. **IPEX Configuration**: Added `--use_ipex` argument to control IPEX usage

### Trainer Updates (`src/training/trainer.py`)

1. **IPEX Integration**: Added IPEX optimization in Trainer constructor:
   ```python
   def _optimize_with_ipex(self):
      if not IPEX_AVAILABLE:
         return self.model, self.optimizer
      
      try:
         model, optimizer = ipex.optimize(self.model, optimizer=self.optimizer)
         return model, optimizer
      except Exception as e:
         logging.warning(f"Failed to optimize with IPEX: {e}")
         return self.model, self.optimizer
   ```

2. **Device Setup**: Updated to prioritize Intel GPUs with IPEX support

3. **Fallback Logic**: Robust fallback with IPEX awareness

## Intel PyTorch Extension (IPEX) Benefits

IPEX provides significant performance improvements on Aurora:

1. **Automatic Optimization**: Optimizes model and optimizer for Intel GPUs
2. **Memory Efficiency**: Better memory management for Intel GPU architecture
3. **Kernel Fusion**: Fuses operations for improved performance
4. **Mixed Precision**: Automatic mixed precision training
5. **Graph Optimization**: Optimizes computation graphs for Intel GPUs

## Usage

### Testing the Configuration

Before running full training jobs, test the Aurora configuration:

```bash
# On Aurora login node or in a test job
python test_aurora_config.py
```

This will verify:
- Environment variable setup
- PyTorch installation and Intel extension
- IPEX import and functionality
- Device detection and availability
- Basic tensor operations
- Training component integration

### Submitting Jobs

Jobs are submitted using the existing workflow:

```bash
python src/agents/main.py --config config.yaml
```

The job scheduler will automatically use the updated PBS template with Aurora-specific configuration and IPEX optimization.

### Manual Job Submission

To submit a job manually:

```bash
# Set required environment variables
export JOB_ID="test_job_001"
export OUTPUT_DIR="/lus/eagle/projects/datascience/$USER/workflows/test"
export DATA_DIR="data/cifar-100-python"
export QUEUE="debug"
export MODEL_TYPE="resnet18"
export LEARNING_RATE="0.001"
export BATCH_SIZE="128"
export NUM_EPOCHS="10"

# Submit job
qsub jobs/train_cifar100.pbs.template
```

### IPEX Control

Control IPEX usage with command-line arguments:

```bash
# Enable IPEX (default)
python src/training/train.py --use_ipex ...

# Disable IPEX
python src/training/train.py --no-use_ipex ...
```

## Aurora-Specific Considerations

### Intel GPU Memory

Aurora's Intel GPUs have different memory characteristics than NVIDIA GPUs:
- Monitor memory usage with `sycl-ls` and Intel tools
- Adjust batch sizes if needed for memory constraints
- Use Intel's memory profiling tools for optimization

### Performance Optimization

1. **IPEX Optimization**: Always use IPEX for Intel GPU training
2. **Threading**: Set `OMP_NUM_THREADS=1` to avoid oversubscription
3. **Memory Pinning**: Use Intel MPI settings for optimal memory access
4. **Device Selection**: Ensure proper GPU device selection with SYCL settings

### Debugging

If jobs fail or performance is poor:

1. **Check IPEX**: Verify IPEX is imported and working
2. **Check SYCL devices**: `sycl-ls`
3. **Verify Intel modules**: `module list`
4. **Check environment variables**: `env | grep -E "(OMP|I_MPI|ZE|SYCL)"`
5. **Review job logs**: Check `.out` and `.err` files

## Troubleshooting

### Common Issues

1. **IPEX Import Failed**:
   - Verify Intel PyTorch Extension is installed
   - Check module loading (`module load frameworks`)
   - Ensure proper Intel environment

2. **Intel GPU not detected**:
   - Verify Intel PyTorch Extension is installed
   - Check SYCL device availability
   - Ensure proper module loading

3. **Memory errors**:
   - Reduce batch size
   - Check GPU memory usage
   - Verify Intel GPU memory allocation

4. **Performance issues**:
   - Ensure IPEX optimization is enabled
   - Check thread pinning settings
   - Verify Intel MKL configuration
   - Monitor GPU utilization

### Getting Help

For Aurora-specific issues:
- Check ALCF Aurora documentation
- Contact ALCF support
- Review Intel oneAPI documentation
- Check Intel PyTorch Extension documentation

## Configuration Files

The following files have been updated for Aurora:

- `jobs/train_cifar100.pbs.template`: PBS job template
- `src/training/train.py`: Main training script with IPEX integration
- `src/training/trainer.py`: Trainer class with IPEX optimization
- `test_aurora_config.py`: Configuration test script with IPEX testing

## Performance Expectations

With IPEX optimization on Aurora:

- **Training Speed**: 2-5x faster than CPU training
- **Memory Efficiency**: Better memory utilization
- **Scalability**: Improved multi-GPU performance
- **Stability**: More stable training on Intel GPUs

## References

- [ALCF Aurora Documentation](https://docs.alcf.anl.gov/aurora/)
- [Intel oneAPI Documentation](https://www.intel.com/content/www/us/en/developer/tools/oneapi/overview.html)
- [Intel PyTorch Extension](https://github.com/intel/intel-extension-for-pytorch)
- [IPEX Optimization Guide](https://intel.github.io/intel-extension-for-pytorch/)
- [SYCL Documentation](https://www.khronos.org/sycl/) 