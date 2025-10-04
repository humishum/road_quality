# Waymo Open Dataset Environment Setup

## Environment Details

- **Conda Environment Name**: `waymo`
- **Python Version**: 3.10
- **TensorFlow Version**: 2.13.0 (with GPU support)
- **Waymo Open Dataset Version**: 1.6.7
- **GPU**: NVIDIA GeForce GTX 1070 (detected and configured)

## Quick Start

### Activate the Environment
```bash
conda activate waymo
```

### Verify GPU Detection
```bash
python -c "import tensorflow as tf; print('GPUs:', tf.config.list_physical_devices('GPU'))"
```

### Run Your Existing Script
```bash
python womd_lidar_dump.py
```

## Installed Key Packages

- `tensorflow==2.13.0` - Deep learning framework with GPU support
- `waymo-open-dataset-tf-2-12-0==1.6.7` - Waymo dataset tools
- `nvidia-cudnn-cu11==8.6.0.163` - CUDA Deep Neural Network library
- `jax==0.4.13` / `jaxlib==0.4.13` - Numerical computing
- `numpy==1.23.5`
- `pandas==1.5.3`
- `matplotlib==3.6.1`
- `scikit-learn==1.2.2`
- `scikit-image==0.20.0`
- `plotly==5.13.1`

## Environment Configuration

The environment is configured with automatic CUDA library path setup via:
`/home/ape/miniconda3/envs/waymo/etc/conda/activate.d/env_vars.sh`

This script automatically sets `LD_LIBRARY_PATH` to include the cuDNN libraries when you activate the environment.

## Testing the Setup

### Test Waymo Import
```python
from waymo_open_dataset import dataset_pb2
from waymo_open_dataset.utils import frame_utils
print("Waymo Open Dataset imported successfully!")
```

### Test GPU Availability
```python
import tensorflow as tf
print(f"TensorFlow version: {tf.__version__}")
print(f"GPUs available: {tf.config.list_physical_devices('GPU')}")
```

## Hardware Info

- **GPU Model**: NVIDIA GeForce GTX 1070
- **VRAM**: 8192 MiB
- **CUDA Version**: 12.6 (driver)
- **Compatible CUDA**: 11.x (libraries installed)

## Troubleshooting

### GPU Not Detected
If GPU is not detected after activating the environment:
```bash
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
```

### Check NVIDIA Driver
```bash
nvidia-smi
```

### Reinstall Environment (if needed)
```bash
conda env remove -n waymo
conda create -n waymo python=3.10 -y
conda activate waymo
pip install tensorflow==2.13.0
pip install nvidia-cudnn-cu11==8.6.0.163
pip install waymo-open-dataset-tf-2-12-0
```

## Notes

- The GTX 1070 works best with CUDA 11.x libraries (installed)
- TensorRT warnings can be ignored - not required for basic operations
- The environment uses TensorFlow 2.13 (compatible with Waymo dataset library)
