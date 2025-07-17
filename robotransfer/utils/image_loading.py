# Project RoboTransfer
#
# Copyright (c) 2025 Horizon Robotics and GigaAI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied. See the License for the specific language governing
# permissions and limitations under the License.

import os

import numpy as np
import torch
from datasets.arrow_dataset import Dataset
from PIL import Image


def load_images_from_dataset(
    dataset: Dataset,
    target_size: None | tuple[int] = (640, 384),
    frames_start=0,
    frames_end=30,
):
    temporal_depth_images = []
    temporal_normal_images = []
    scene_id = dataset[frames_start]["scene_id"]
    scene_id_has_changed = False
    for i in range(frames_start, frames_end):
        # padding if scene_id has changed
        if dataset[i]["scene_id"] != scene_id:
            temporal_depth_images.append(temporal_depth_images[-1])
            temporal_normal_images.append(temporal_normal_images[-1])
            scene_id_has_changed = True
            continue

        depth_left_image, depth_middle_image, depth_right_image = dataset[i][
            "depth_images"
        ]
        if target_size is not None:
            depth_left_image = depth_left_image.resize(size=target_size)
            depth_middle_image = depth_middle_image.resize(size=target_size)
            depth_right_image = depth_right_image.resize(size=target_size)

        normal_left_image, normal_middle_image, normal_right_image = dataset[
            i
        ]["normal_images"]
        if target_size is not None:
            normal_left_image = normal_left_image.resize(size=target_size)
            normal_middle_image = normal_middle_image.resize(size=target_size)
            normal_right_image = normal_right_image.resize(size=target_size)

        depth_image = np.concatenate(
            [
                np.array(depth_left_image) / 1000,
                np.array(depth_middle_image) / 1000,
                np.array(depth_right_image) / 1000,
            ],
            axis=1,
        )
        normal_image = np.concatenate(
            [
                np.array(normal_left_image) / 255,
                np.array(normal_middle_image) / 255,
                np.array(normal_right_image) / 255,
            ],
            axis=1,
        )
        temporal_depth_images.append(depth_image)
        temporal_normal_images.append(normal_image)
    temporal_depth_image = np.stack(temporal_depth_images, axis=0)
    temporal_depth_image_pt = torch.from_numpy(temporal_depth_image).unsqueeze(
        dim=1
    )
    temporal_normal_image = np.stack(temporal_normal_images, axis=0)
    temporal_normal_image_pt = torch.from_numpy(temporal_normal_image).permute(
        0, 3, 1, 2
    )
    return (
        temporal_depth_image_pt.clip(min=0.1, max=2.0),
        2 * temporal_normal_image_pt - 1,
        scene_id_has_changed,
    )


def get_dataset_length(data_root: str):
    """Get the length of the dataset by checking the number of images in each camera's directory.

    Args:
        data_root (str): The root directory of the dataset.

    Returns:
        int: The length of the dataset, which is the number of images in each camera's directory.
    """

    normal_left_images_path = os.path.join(
        data_root, "left_camera", "mono_normal"
    )
    normal_middle_images_path = os.path.join(
        data_root, "middle_camera", "mono_normal"
    )
    normal_right_images_path = os.path.join(
        data_root, "right_camera", "mono_normal"
    )

    length_left = len(os.listdir(normal_left_images_path))
    length_middle = len(os.listdir(normal_middle_images_path))
    length_right = len(os.listdir(normal_right_images_path))
    if length_left == length_middle == length_right:
        return length_left
    else:
        raise ValueError(
            "The length of the dataset is not equal for all cameras. "
            f"Left: {length_left}, Middle: {length_middle}, Right: {length_right}"
        )


def load_images_from_local(
    data_root: str,
    target_size: None | tuple[int] = (640, 384),
    frames_start=0,
    frames_end=30,
):
    depth_key = (
        "mono_depth"
        if os.path.exists(os.path.join(data_root, "left_camera", "mono_depth"))
        else "depth"
    )
    depth_left_images_path = os.path.join(data_root, "left_camera", depth_key)
    depth_middle_images_path = os.path.join(
        data_root, "middle_camera", depth_key
    )
    depth_right_images_path = os.path.join(
        data_root, "right_camera", depth_key
    )
    normal_left_images_path = os.path.join(
        data_root, "left_camera", "mono_normal"
    )
    normal_middle_images_path = os.path.join(
        data_root, "middle_camera", "mono_normal"
    )
    normal_right_images_path = os.path.join(
        data_root, "right_camera", "mono_normal"
    )

    depth_left_images = sorted(os.listdir(depth_left_images_path))[
        frames_start:frames_end
    ]
    depth_middle_images = sorted(os.listdir(depth_middle_images_path))[
        frames_start:frames_end
    ]
    depth_right_images = sorted(os.listdir(depth_right_images_path))[
        frames_start:frames_end
    ]

    normal_left_images = sorted(os.listdir(normal_left_images_path))[
        frames_start:frames_end
    ]
    normal_middle_images = sorted(os.listdir(normal_middle_images_path))[
        frames_start:frames_end
    ]
    normal_right_images = sorted(os.listdir(normal_right_images_path))[
        frames_start:frames_end
    ]

    temporal_depth_images = []
    temporal_normal_images = []
    for (
        depth_left_image,
        depth_middle_image,
        depth_right_image,
        normal_left_image,
        normal_middle_image,
        normal_right_image,
    ) in zip(
        depth_left_images,
        depth_middle_images,
        depth_right_images,
        normal_left_images,
        normal_middle_images,
        normal_right_images,
        strict=True,
    ):
        depth_left_image = Image.open(
            os.path.join(depth_left_images_path, depth_left_image)
        )
        depth_middle_image = Image.open(
            os.path.join(depth_middle_images_path, depth_middle_image)
        )
        depth_right_image = Image.open(
            os.path.join(depth_right_images_path, depth_right_image)
        )
        if target_size is not None:
            depth_left_image = depth_left_image.resize(size=target_size)
            depth_middle_image = depth_middle_image.resize(size=target_size)
            depth_right_image = depth_right_image.resize(size=target_size)

        normal_left_image = Image.open(
            os.path.join(normal_left_images_path, normal_left_image)
        )
        normal_middle_image = Image.open(
            os.path.join(normal_middle_images_path, normal_middle_image)
        )
        normal_right_image = Image.open(
            os.path.join(normal_right_images_path, normal_right_image)
        )
        if target_size is not None:
            normal_left_image = normal_left_image.resize(size=target_size)
            normal_middle_image = normal_middle_image.resize(size=target_size)
            normal_right_image = normal_right_image.resize(size=target_size)

        depth_image = np.concatenate(
            [
                np.array(depth_left_image) / 1000,
                np.array(depth_middle_image) / 1000,
                np.array(depth_right_image) / 1000,
            ],
            axis=1,
        )
        normal_image = np.concatenate(
            [
                np.array(normal_left_image) / 255,
                np.array(normal_middle_image) / 255,
                np.array(normal_right_image) / 255,
            ],
            axis=1,
        )
        temporal_depth_images.append(depth_image)
        temporal_normal_images.append(normal_image)
    temporal_depth_image = np.stack(temporal_depth_images, axis=0)
    temporal_depth_image_pt = torch.from_numpy(temporal_depth_image).unsqueeze(
        dim=1
    )
    temporal_normal_image = np.stack(temporal_normal_images, axis=0)
    temporal_normal_image_pt = torch.from_numpy(temporal_normal_image).permute(
        0, 3, 1, 2
    )
    return (
        temporal_depth_image_pt.clip(min=0.1, max=2.0),
        2 * temporal_normal_image_pt - 1,
    )
