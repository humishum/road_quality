import zlib
import numpy as np
import tensorflow as tf

from waymo_open_dataset import dataset_pb2 as _dataset_pb2
from waymo_open_dataset.protos import scenario_pb2
from waymo_open_dataset.utils import frame_utils

def scenario_to_first_frame(path):
    dataset = tf.data.TFRecordDataset(path, compression_type='')
    raw = next(iter(dataset))
    scenario = scenario_pb2.Scenario()
    scenario.ParseFromString(bytes(raw.numpy()))
    
    # LiDAR data lives in scenario.lidar_data (list, one per timestamp).
    # Take the first lidar_data entry:
    lidar_record = scenario.lidar_data[0].lidar_points_compressed

    # Decompress
    frame_bytes = zlib.decompress(lidar_record)
    frame = _dataset_pb2.Frame()
    frame.ParseFromString(frame_bytes)
    return frame

def frame_to_xyz_intensity(frame):
    (range_images, camera_projections, range_image_top_pose) = \
        frame_utils.parse_range_image_and_camera_projection(frame)
    RT_FIRST = frame_utils.LaserReturnType.RANGE_IMAGE_RETURN_TYPE_FIRST_RETURN

    points, intensities = [], []
    pts_by_lidar = frame_utils.convert_range_image_to_point_cloud(
        frame, range_images, camera_projections, range_image_top_pose,
        ri_index=RT_FIRST, keep_polar=False, use_multi_return=True
    )
    for lidar_id, xyz in pts_by_lidar.items():
        ri = range_images[lidar_id][RT_FIRST]
        ri_t = tf.reshape(tf.convert_to_tensor(ri.data), ri.shape.dims)
        valid = ri_t[..., 0] > 0
        inten = tf.boolean_mask(ri_t[..., 1], valid).numpy().astype(np.float32)
        points.append(xyz.astype(np.float32))
        intensities.append(inten)
    return np.concatenate(points), np.concatenate(intensities)

if __name__ == "__main__":
    scenario_file = "data/training_20s.tfrecord-00000-of-01000"  # adjust
    frame = scenario_to_first_frame(scenario_file)
    xyz, inten = frame_to_xyz_intensity(frame)
    print("Got point cloud:", xyz.shape, "points")
    np.save("lidar_frame0.npy", np.c_[xyz, inten])
