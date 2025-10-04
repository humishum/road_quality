# Waymo LiDAR Data Setup

## Problem Solved

The original `womd_lidar_dump.py` script had an import error:
```
AttributeError: module 'waymo_open_dataset.dataset_pb2' has no attribute 'Scenario'
```

This was because the `Scenario` class is in `waymo_open_dataset.protos.scenario_pb2`, not in `dataset_pb2`.

## Current Status

✅ **Fixed the import issue** - Updated `womd_lidar_dump.py` to use the correct imports
✅ **Created working LiDAR reader** - `waymo_lidar_reader.py` can handle both Frame and Scenario data
✅ **Discovered data limitation** - The current file (`training_20s.tfrecord-00000-of-01000`) is a Motion Prediction dataset without sensor data
✅ **Created synthetic examples** - `lidar_example.py` demonstrates LiDAR data processing with synthetic data

## Files Created

1. **`waymo_lidar_reader.py`** - Robust script that can read both Frame-based and Scenario-based Waymo data
2. **`lidar_example.py`** - Demonstrates LiDAR data processing with synthetic road scene data
3. **`LIDAR_README.md`** - This documentation file

## How to Use

### With Current Data (Motion Prediction Dataset)
```bash
conda activate waymo
python waymo_lidar_reader.py data/training_20s.tfrecord-00000-of-01000
```
*Note: This will show that the file contains no LiDAR data*

### With Synthetic Data
```bash
conda activate waymo
python lidar_example.py
```
*This creates synthetic LiDAR data and saves it as .npy files*

### With Real Waymo LiDAR Data (when you get it)
```bash
conda activate waymo
python waymo_lidar_reader.py path/to/perception_dataset.tfrecord
```

## Getting Real LiDAR Data

To get actual Waymo LiDAR data:

1. Visit: https://waymo.com/open/data/
2. Sign up and agree to terms
3. Download the **"Perception"** dataset (NOT "Motion Prediction")
4. The Perception dataset contains:
   - Frame-based data with LiDAR, camera, and radar
   - Files named like: `training_0000.tfrecord`
   - ~150k points per frame from multiple LiDAR sensors

## Data Format

### Synthetic Data (from lidar_example.py)
- Points: `(N, 3)` numpy array with X, Y, Z coordinates
- Intensities: `(N,)` numpy array with reflection intensities
- Combined: `(N, 4)` numpy array with [X, Y, Z, Intensity]

### Real Waymo Data (when available)
- Multiple LiDAR sensors (top, front, side, rear)
- Higher point density (~150k points per frame)
- Temporal sequences of frames
- More realistic intensity values

## Next Steps

1. **Download Perception Dataset** - Get real LiDAR data from Waymo
2. **Extend Analysis** - Add filtering, segmentation, object detection
3. **Temporal Analysis** - Work with sequences of frames
4. **Machine Learning** - Apply ML models for road quality assessment

## Environment

- **Conda Environment**: `waymo`
- **Python**: 3.10
- **TensorFlow**: 2.13.0 (with GPU support)
- **Waymo Dataset**: 1.6.7
- **GPU**: NVIDIA GeForce GTX 1070

## Quick Test

```bash
conda activate waymo
python -c "
from waymo_open_dataset import dataset_pb2
from waymo_open_dataset.protos import scenario_pb2
print('✅ Waymo imports working!')
print('Frame fields:', list(dataset_pb2.Frame().DESCRIPTOR.fields_by_name.keys()))
print('Scenario fields:', list(scenario_pb2.Scenario().DESCRIPTOR.fields_by_name.keys()))
"
```

You now have a working setup for Waymo LiDAR data processing! 🎉
