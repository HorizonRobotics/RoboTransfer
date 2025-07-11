import os
import torch
import numpy as np
from PIL import Image

from datasets.arrow_dataset import Dataset


def load_images_from_dataset(
    dataset: Dataset, target_size: None | tuple[int] = (640, 384), frames_start=0, frames_end=30
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

        normal_left_image, normal_middle_image, normal_right_image = dataset[i][
            "normal_images"
        ]
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
    temporal_depth_image_pt = torch.from_numpy(temporal_depth_image).unsqueeze(dim=1)
    temporal_normal_image = np.stack(temporal_normal_images, axis=0)
    temporal_normal_image_pt = torch.from_numpy(temporal_normal_image).permute(
        0, 3, 1, 2
    )
    return temporal_depth_image_pt.clip(
        min=0.1, max=2.0
    ), 2 * temporal_normal_image_pt - 1, scene_id_has_changed


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
    depth_middle_images_path = os.path.join(data_root, "middle_camera", depth_key)
    depth_right_images_path = os.path.join(data_root, "right_camera", depth_key)
    normal_left_images_path = os.path.join(data_root, "left_camera", "mono_normal")
    normal_middle_images_path = os.path.join(data_root, "middle_camera", "mono_normal")
    normal_right_images_path = os.path.join(data_root, "right_camera", "mono_normal")

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
    temporal_depth_image_pt = torch.from_numpy(temporal_depth_image).unsqueeze(dim=1)
    temporal_normal_image = np.stack(temporal_normal_images, axis=0)
    temporal_normal_image_pt = torch.from_numpy(temporal_normal_image).permute(
        0, 3, 1, 2
    )
    return temporal_depth_image_pt.clip(
        min=0.1, max=2.0
    ), 2 * temporal_normal_image_pt - 1
