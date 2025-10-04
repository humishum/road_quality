#!/usr/bin/env python3
"""
Waymo LiDAR Data Example

This script demonstrates how to work with Waymo LiDAR data.
Since the current dataset file doesn't contain sensor data, this example
shows how to create and visualize synthetic LiDAR data similar to what
you'd get from the Waymo Perception dataset.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def create_synthetic_lidar_data():
    """Create synthetic LiDAR data similar to Waymo's format."""
    print("Creating synthetic LiDAR data...")
    
    # Create a simple 3D scene with some objects
    np.random.seed(42)
    
    # Ground plane
    x_ground = np.linspace(-50, 50, 100)
    y_ground = np.linspace(-50, 50, 100)
    X_ground, Y_ground = np.meshgrid(x_ground, y_ground)
    Z_ground = np.zeros_like(X_ground)
    
    ground_points = np.column_stack([
        X_ground.flatten(),
        Y_ground.flatten(), 
        Z_ground.flatten()
    ])
    ground_intensities = np.random.uniform(0.1, 0.3, len(ground_points))
    
    # Add some buildings/objects
    objects_points = []
    objects_intensities = []
    
    # Building 1
    x_building = np.random.uniform(10, 20, 500)
    y_building = np.random.uniform(10, 20, 500)
    z_building = np.random.uniform(0, 15, 500)
    building1_points = np.column_stack([x_building, y_building, z_building])
    building1_intensities = np.random.uniform(0.4, 0.8, len(building1_points))
    
    # Building 2
    x_building = np.random.uniform(-25, -15, 400)
    y_building = np.random.uniform(-10, 0, 400)
    z_building = np.random.uniform(0, 20, 400)
    building2_points = np.column_stack([x_building, y_building, z_building])
    building2_intensities = np.random.uniform(0.3, 0.7, len(building2_points))
    
    # Car
    x_car = np.random.uniform(-5, 5, 200)
    y_car = np.random.uniform(-5, 5, 200)
    z_car = np.random.uniform(0, 2, 200)
    car_points = np.column_stack([x_car, y_car, z_car])
    car_intensities = np.random.uniform(0.6, 0.9, len(car_points))
    
    # Combine all points
    all_points = np.vstack([ground_points, building1_points, building2_points, car_points])
    all_intensities = np.hstack([ground_intensities, building1_intensities, building2_intensities, car_intensities])
    
    # Add some noise
    noise = np.random.normal(0, 0.1, all_points.shape)
    all_points += noise
    
    print(f"Created synthetic point cloud with {len(all_points)} points")
    print(f"Point cloud bounds:")
    print(f"  X: {all_points[:, 0].min():.2f} to {all_points[:, 0].max():.2f}")
    print(f"  Y: {all_points[:, 1].min():.2f} to {all_points[:, 1].max():.2f}")
    print(f"  Z: {all_points[:, 2].min():.2f} to {all_points[:, 2].max():.2f}")
    print(f"  Intensity range: {all_intensities.min():.3f} to {all_intensities.max():.3f}")
    
    return all_points, all_intensities


def visualize_lidar_data(points, intensities, max_points=10000):
    """Visualize LiDAR point cloud data."""
    print(f"Visualizing point cloud (showing up to {max_points} points)...")
    
    # Subsample if too many points
    if len(points) > max_points:
        indices = np.random.choice(len(points), max_points, replace=False)
        points = points[indices]
        intensities = intensities[indices]
    
    # Create 3D plot
    fig = plt.figure(figsize=(15, 5))
    
    # 3D scatter plot
    ax1 = fig.add_subplot(131, projection='3d')
    scatter = ax1.scatter(points[:, 0], points[:, 1], points[:, 2], 
                         c=intensities, cmap='viridis', s=1, alpha=0.6)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('3D Point Cloud')
    plt.colorbar(scatter, ax=ax1, shrink=0.5, label='Intensity')
    
    # Top view (X-Y plane)
    ax2 = fig.add_subplot(132)
    ax2.scatter(points[:, 0], points[:, 1], c=intensities, cmap='viridis', s=1, alpha=0.6)
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('Top View (X-Y)')
    ax2.set_aspect('equal')
    
    # Side view (X-Z plane)
    ax3 = fig.add_subplot(133)
    ax3.scatter(points[:, 0], points[:, 2], c=intensities, cmap='viridis', s=1, alpha=0.6)
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Z (m)')
    ax3.set_title('Side View (X-Z)')
    ax3.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('lidar_visualization.png', dpi=150, bbox_inches='tight')
    print("Saved visualization to 'lidar_visualization.png'")
    plt.show()


def analyze_lidar_data(points, intensities):
    """Analyze LiDAR data statistics."""
    print("\nLiDAR Data Analysis:")
    print("=" * 30)
    
    # Basic statistics
    print(f"Total points: {len(points):,}")
    print(f"Point density: {len(points) / 10000:.1f} points per 100m²")
    
    # Distance analysis
    distances = np.sqrt(np.sum(points**2, axis=1))
    print(f"Distance range: {distances.min():.1f}m to {distances.max():.1f}m")
    print(f"Mean distance: {distances.mean():.1f}m")
    
    # Height analysis
    print(f"Height range: {points[:, 2].min():.1f}m to {points[:, 2].max():.1f}m")
    print(f"Mean height: {points[:, 2].mean():.1f}m")
    
    # Intensity analysis
    print(f"Intensity range: {intensities.min():.3f} to {intensities.max():.3f}")
    print(f"Mean intensity: {intensities.mean():.3f}")
    
    # Distance vs intensity correlation
    correlation = np.corrcoef(distances, intensities)[0, 1]
    print(f"Distance-Intensity correlation: {correlation:.3f}")


def save_lidar_data(points, intensities, filename="synthetic_lidar_data"):
    """Save LiDAR data to files."""
    print(f"\nSaving LiDAR data to {filename}.*")
    
    # Save as numpy arrays
    np.save(f"{filename}_points.npy", points.astype(np.float32))
    np.save(f"{filename}_intensities.npy", intensities.astype(np.float32))
    
    # Save as combined array
    combined_data = np.column_stack([points, intensities])
    np.save(f"{filename}_combined.npy", combined_data.astype(np.float32))
    
    print(f"Saved files:")
    print(f"  - {filename}_points.npy ({points.shape})")
    print(f"  - {filename}_intensities.npy ({intensities.shape})")
    print(f"  - {filename}_combined.npy ({combined_data.shape})")


def main():
    print("Waymo LiDAR Data Example")
    print("=" * 40)
    
    # Create synthetic data
    points, intensities = create_synthetic_lidar_data()
    
    # Analyze the data
    analyze_lidar_data(points, intensities)
    
    # Save the data
    save_lidar_data(points, intensities)
    
    # Visualize (comment out if running headless)
    try:
        visualize_lidar_data(points, intensities)
    except Exception as e:
        print(f"Visualization failed (probably headless): {e}")
        print("You can load the saved .npy files to visualize later.")
    
    print("\n" + "=" * 40)
    print("HOW TO GET REAL WAYMO LIDAR DATA:")
    print("=" * 40)
    print("1. Visit: https://waymo.com/open/data/")
    print("2. Sign up and agree to the terms")
    print("3. Download the 'Perception' dataset (not 'Motion Prediction')")
    print("4. The Perception dataset contains:")
    print("   - Frame-based data with LiDAR, camera, and radar")
    print("   - Files named like: training_0000.tfrecord")
    print("5. Use the waymo_lidar_reader.py script with those files")
    print("\nExample command once you have real data:")
    print("  python waymo_lidar_reader.py path/to/training_0000.tfrecord")


if __name__ == "__main__":
    main()
