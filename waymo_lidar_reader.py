#!/usr/bin/env python3
"""
Simple Waymo Open Dataset LiDAR Data Reader

This script can read LiDAR data from Waymo Open Dataset files.
It handles both Frame-based and Scenario-based data formats.

Usage:
    python waymo_lidar_reader.py <path_to_tfrecord_file>

Note: The current data file (training_20s.tfrecord-00000-of-01000) appears to be
a motion prediction dataset without sensor data. For LiDAR data, you'll need
to download the "Perception" dataset from Waymo Open Dataset.
"""

import sys
import zlib
import numpy as np
import tensorflow as tf
from waymo_open_dataset import dataset_pb2
from waymo_open_dataset.protos import scenario_pb2
from waymo_open_dataset.utils import frame_utils


def read_frame_data(file_path):
    """Read LiDAR data from a Frame-based Waymo dataset file."""
    print(f"Reading Frame-based data from: {file_path}")
    
    dataset = tf.data.TFRecordDataset(file_path, compression_type='')
    
    for i, raw_data in enumerate(dataset):
        try:
            frame = dataset_pb2.Frame()
            frame.ParseFromString(bytes(raw_data.numpy()))
            
            print(f"Frame {i}:")
            print(f"  Timestamp: {frame.timestamp_micros}")
            print(f"  Number of lasers: {len(frame.lasers)}")
            
            if len(frame.lasers) > 0:
                # Extract point cloud data
                points, intensities = extract_lidar_points(frame)
                print(f"  Point cloud shape: {points.shape}")
                print(f"  Intensity range: {intensities.min():.3f} - {intensities.max():.3f}")
                
                # Save first frame as example
                if i == 0:
                    np.save("lidar_points_frame0.npy", points)
                    np.save("lidar_intensities_frame0.npy", intensities)
                    print(f"  Saved point cloud to lidar_points_frame0.npy")
                    print(f"  Saved intensities to lidar_intensities_frame0.npy")
                
                return points, intensities
            else:
                print(f"  No laser data found in frame {i}")
                
        except Exception as e:
            print(f"Error reading frame {i}: {e}")
            continue
    
    print("No valid frames with LiDAR data found")
    return None, None


def read_scenario_data(file_path):
    """Read LiDAR data from a Scenario-based Waymo dataset file."""
    print(f"Reading Scenario-based data from: {file_path}")
    
    dataset = tf.data.TFRecordDataset(file_path, compression_type='')
    
    for i, raw_data in enumerate(dataset):
        try:
            scenario = scenario_pb2.Scenario()
            scenario.ParseFromString(bytes(raw_data.numpy()))
            
            print(f"Scenario {i}:")
            print(f"  Scenario ID: {scenario.scenario_id}")
            print(f"  Number of timestamps: {len(scenario.timestamps_seconds)}")
            print(f"  Number of compressed frame laser data: {len(scenario.compressed_frame_laser_data)}")
            
            if len(scenario.compressed_frame_laser_data) > 0:
                # Decompress and extract LiDAR data
                lidar_data = scenario.compressed_frame_laser_data[0]
                frame_bytes = zlib.decompress(lidar_data)
                frame = dataset_pb2.Frame()
                frame.ParseFromString(frame_bytes)
                
                points, intensities = extract_lidar_points(frame)
                print(f"  Point cloud shape: {points.shape}")
                print(f"  Intensity range: {intensities.min():.3f} - {intensities.max():.3f}")
                
                # Save first frame as example
                if i == 0:
                    np.save("lidar_points_scenario0.npy", points)
                    np.save("lidar_intensities_scenario0.npy", intensities)
                    print(f"  Saved point cloud to lidar_points_scenario0.npy")
                    print(f"  Saved intensities to lidar_intensities_scenario0.npy")
                
                return points, intensities
            else:
                print(f"  No compressed laser data found in scenario {i}")
                
        except Exception as e:
            print(f"Error reading scenario {i}: {e}")
            continue
    
    print("No valid scenarios with LiDAR data found")
    return None, None


def extract_lidar_points(frame):
    """Extract 3D points and intensities from a Waymo frame."""
    try:
        # Parse range images and camera projections
        (range_images, camera_projections, range_image_top_pose) = \
            frame_utils.parse_range_image_and_camera_projection(frame)
        
        # Get first return points
        points_3d, intensities = frame_utils.convert_range_image_to_point_cloud(
            frame, range_images, camera_projections, range_image_top_pose,
            ri_index=frame_utils.LaserReturnType.RANGE_IMAGE_RETURN_TYPE_FIRST_RETURN,
            keep_polar=False
        )
        
        # Combine points from all lidars
        all_points = []
        all_intensities = []
        
        for lidar_id, points in points_3d.items():
            all_points.append(points)
            
            # Get corresponding intensities
            ri = range_images[lidar_id][frame_utils.LaserReturnType.RANGE_IMAGE_RETURN_TYPE_FIRST_RETURN]
            ri_t = tf.reshape(tf.convert_to_tensor(ri.data), ri.shape.dims)
            valid = ri_t[..., 0] > 0
            intensity = tf.boolean_mask(ri_t[..., 1], valid).numpy().astype(np.float32)
            all_intensities.append(intensity)
        
        if all_points:
            combined_points = np.concatenate(all_points, axis=0).astype(np.float32)
            combined_intensities = np.concatenate(all_intensities, axis=0).astype(np.float32)
            return combined_points, combined_intensities
        else:
            return np.array([]), np.array([])
            
    except Exception as e:
        print(f"Error extracting LiDAR points: {e}")
        return np.array([]), np.array([])


def main():
    if len(sys.argv) != 2:
        print("Usage: python waymo_lidar_reader.py <path_to_tfrecord_file>")
        print("\nExample:")
        print("  python waymo_lidar_reader.py data/training_20s.tfrecord-00000-of-01000")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    print("Waymo Open Dataset LiDAR Reader")
    print("=" * 40)
    
    # Try to read as Frame data first
    points, intensities = read_frame_data(file_path)
    
    if points is None or len(points) == 0:
        print("\nFrame-based reading failed, trying Scenario-based...")
        points, intensities = read_scenario_data(file_path)
    
    if points is not None and len(points) > 0:
        print(f"\nSuccessfully extracted LiDAR data!")
        print(f"Point cloud shape: {points.shape}")
        print(f"Intensity array shape: {intensities.shape}")
        print(f"Point cloud bounds:")
        print(f"  X: {points[:, 0].min():.2f} to {points[:, 0].max():.2f}")
        print(f"  Y: {points[:, 1].min():.2f} to {points[:, 1].max():.2f}")
        print(f"  Z: {points[:, 2].min():.2f} to {points[:, 2].max():.2f}")
    else:
        print("\nNo LiDAR data found in the file.")
        print("\nNote: The current file appears to be a motion prediction dataset")
        print("without sensor data. To get LiDAR data, you need to download")
        print("the 'Perception' dataset from Waymo Open Dataset, not the")
        print("'Motion Prediction' dataset.")


if __name__ == "__main__":
    main()
