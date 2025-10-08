#!/usr/bin/env python
import json
import os
import sys
from os.path import basename
from pathlib import Path

import numpy as np
from PIL import Image
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore

# Create a type store to use if the bag has no message definitions.
default_typestore = get_typestore(Stores.ROS2_FOXY)


# return timestamp list
def save_images(bag_path, edex_folder, left_camera_topic, right_camera_topic):
    left_stamps = []
    right_stamps = []

    with AnyReader([bag_path], default_typestore=default_typestore) as reader:
        print('Topics:')
        for topic in sorted(reader.topics):
            print(f'  {topic}')

        for connection, timestamp, raw in reader.messages(connections=reader.connections):
            if connection.topic == left_camera_topic:
                msg = reader.deserialize(raw, connection.msgtype)
                left_stamps.append(msg.header.stamp.sec*(10**9)+msg.header.stamp.nanosec)
                continue
            if connection.topic == right_camera_topic:
                msg = reader.deserialize(raw, connection.msgtype)
                right_stamps.append(msg.header.stamp.sec*(10**9)+msg.header.stamp.nanosec)
                continue
    common_stamps = np.sort(list(set(right_stamps).intersection(left_stamps)))
    print(f'left frames={len(left_stamps)}')
    print(f'right frames={len(right_stamps)}')
    print(f'pairs={len(common_stamps)}')

    os.makedirs(os.path.join(edex_folder, "images"), exist_ok=True)

    with AnyReader([bag_path], default_typestore=default_typestore) as reader:
        i = 0
        for connection, timestamp, raw in reader.messages(connections=reader.connections):
            if connection.topic == left_camera_topic:
                msg = reader.deserialize(raw, connection.msgtype)
                left_stamp = msg.header.stamp.sec*(10**9)+msg.header.stamp.nanosec
                if left_stamp in common_stamps:
                    cnt = int(np.where(left_stamp == common_stamps)[0][0])
                    image = Image.frombytes('L', (msg.width, msg.height), msg.data)
                    image.save(os.path.join(edex_folder, "images", 'cam0.%05d.png' % cnt), format="png")
                else:
                    print(f'\nmissing left frame. timestamp={left_stamp}')

            if connection.topic == right_camera_topic:
                msg = reader.deserialize(raw, connection.msgtype)
                right_stamp = msg.header.stamp.sec*(10**9)+msg.header.stamp.nanosec
                if right_stamp in common_stamps:
                    cnt = int(np.where(right_stamp == common_stamps)[0][0])
                    image = Image.frombytes('L', (msg.width, msg.height), msg.data)
                    image.save(os.path.join(edex_folder, "images", 'cam1.%05d.png' % cnt), format="png")
                else:
                    print(f'missing right frame. timestamp={left_stamp}')
            i += 1
            if i % 100 == 0:
                print(f'\rprogress {int(100 * i / reader.message_count)}%', end='')
    return common_stamps


def save_frame_metadata(common_stamps, edex_folder):
    with open(os.path.join(edex_folder,  'frame_metadata.jsonl'), mode='a') as writer:
        for i in range(len(common_stamps)):
            line_tmp = {
                "frame_id": i,
                "cams": [
                    {"id": 0, "filename": "images/cam0.%05d.png" % i, "timestamp": int(common_stamps[i])},
                    {"id": 1, "filename": "images/cam1.%05d.png" % i, "timestamp": int(common_stamps[i])}
                ]
            }
            json.dump(line_tmp, writer)
            writer.write('\n')


def extract_intrinsics(projection_matrix):
    fx = projection_matrix[0]
    fy = projection_matrix[5]
    cx = projection_matrix[2]
    cy = projection_matrix[6]
    return fx, fy, cx, cy


def create_edex(fx1, fy1, cx1, cy1, sx1, sy1, fx2, fy2, cx2, cy2, sx2, sy2, baseline, edex_folder):

    # # for Realsense D455
    # rot_mat_imu = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    # translation_vec_imu = np.array([-0.03022, -0.0074, -0.016]).reshape([-1,1])

    # for Realsense D435i
    rot_mat_imu = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    translation_vec_imu = np.array([-0.005520000122487545, -0.005100000184029341, -0.011739999987185001]).reshape([-1,1])

    imu_out_mat = np.hstack([rot_mat_imu, translation_vec_imu])
    imu_out_list = [list(i) for i in imu_out_mat]

    t1 = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ]
    t2 = [
        [1.0, 0.0, 0.0, baseline],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ]
    cam1 = {
        'transform': t1,
        'intrinsics': {
            "distortion_model": "pinhole",
            "distortion_params": [],
            "focal": [fx1, fy1],
            "principal": [cx1, cy1],
            "size": [sx1, sy1]
        }
    }
    cam2 = {
        'transform': t2,
        'intrinsics': {
            "distortion_model": "pinhole",
            "distortion_params": [],
            "focal": [fx2, fy2],
            "principal": [cx2, cy2],
            "size": [sx2, sy2]
        }
    }
    json_str = (json.dumps([{"version": "0.9",
                             "frame_start": 0,
                             "frame_end": len(os.listdir(os.path.join(edex_folder, 'images'))),
                             "imu": {"g": [0.0, -9.81, 0.0],
                                     "measurements": "IMU.jsonl",
                                     "transform": imu_out_list,
                                     # for Realsense D435i with BMI055
                                     "gyro_noise_density": 0.00015182353358040757,
                                     "gyro_random_walk": 1.0784169054875935e-05,
                                     "accel_noise_density": 0.0016706377518817322,
                                     "accel_random_walk": 0.00023972770530160262,
                                     # # for Realsense D455 and HAWK with BMI088
                                     # "gyro_noise_density": 0.00017221024330235052,
                                     # "gyro_random_walk": 9.595089650939567e-06,
                                     # "accel_noise_density": 0.0014855959546680947,
                                     # "accel_random_walk": 0.0001219997738886554,
                                     "frequency": 200},
                             "cameras": [cam1, cam2]
                             },
                            {"frame_metadata": "frame_metadata.jsonl",
                             "points3d": {}, "points2d": {}, "rig_positions": {},
                             "sequence": [["00/l.000000.png"],
                                          ["01/r.000000.png"]]
                             #"fps": int(bag.get_type_and_topic_info()[1][left_cam_info_topic.replace('info/camera_info', 'image/data')][3])
                             }], indent=4))
    with open(os.path.join(edex_folder, 'stereo.edex'), 'w') as outfile:
        outfile.write(json_str)


def convert_bag_to_edex(bag_path, edex_folder,
                        left_cam_info_topic='/camera/infra1/camera_info',
                        right_cam_info_topic='/camera/infra2/camera_info'):
    with AnyReader([bag_path], default_typestore=default_typestore) as reader:
        connections = [x for x in reader.connections if x.topic == left_cam_info_topic]
        for connection, timestamp, rawdata in reader.messages(connections):
            left_cam = reader.deserialize(rawdata, connection.msgtype)
            break

    with AnyReader([bag_path], default_typestore=default_typestore) as reader:
        connections = [x for x in reader.connections if x.topic == right_cam_info_topic]
        for connection, timestamp, rawdata in reader.messages(connections):
            right_cam = reader.deserialize(rawdata, connection.msgtype)
            break

    sx1, sy1 = left_cam.width, left_cam.height
    fx1, fy1, cx1, cy1 = extract_intrinsics(left_cam.p)

    sx2, sy2 = right_cam.width, right_cam.height
    fx2, fy2, cx2, cy2 = extract_intrinsics(right_cam.p)

    # baseline = 0.0950183 # for Realsense D455
    baseline = 0.05 # for Realsense D435i
    create_edex(fx1, fy1, cx1, cy1, sx1, sy1, fx2, fy2, cx2, cy2, sx2, sy2, baseline, edex_folder)


def write_imu_data(bag_path, edex_folder, imu_topic='/camera/imu'):
    with open(os.path.join(edex_folder, "IMU.jsonl"), 'a') as f:
        with AnyReader([bag_path], default_typestore=default_typestore) as reader:
            connections = [x for x in reader.connections if x.topic==imu_topic ]
            for connection, timestamp, rawdata in reader.messages(connections):
                msg = reader.deserialize(rawdata, connection.msgtype)
                imu_line = {"AngularVelocityX": msg.angular_velocity.x,
                            "AngularVelocityY": msg.angular_velocity.y,
                            "AngularVelocityZ": msg.angular_velocity.z,
                            "LinearAccelerationX": msg.linear_acceleration.x,
                            "LinearAccelerationY": msg.linear_acceleration.y,
                            "LinearAccelerationZ": msg.linear_acceleration.z,
                            "timestamp": msg.header.stamp.sec*10**9+msg.header.stamp.nanosec}
                json.dump(imu_line, f)
                f.write('\n')


def bag_to_edex(bag_folder, edex_folder):
    print('bag_folder={}'.format(bag_folder))
    print('edex_folder={}'.format(edex_folder))
    bag_path = Path(bag_folder)
    common_stamps = save_images(bag_path, edex_folder,
                                '/camera/infra1/image_rect_raw',
                                '/camera/infra2/image_rect_raw')
    print('export images: done')
    save_frame_metadata(common_stamps, edex_folder)
    print('create metadata jsonl: done')
    convert_bag_to_edex(bag_path, edex_folder)
    print('create edex file: done')
    write_imu_data(bag_path, edex_folder)
    print('write imu measurements: done')

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Use\n {} path/to/bag_folder path/to/edex_folder".format(basename(sys.argv[0])))
        sys.exit('Wrong parameter number')
    bag_to_edex(os.path.expanduser(sys.argv[1]), os.path.expanduser(sys.argv[2]))
    print('finish success')
