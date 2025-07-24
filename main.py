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

import argparse

import torch
from datasets import load_dataset
from PIL import Image
from robotransfer import RoboTransferPipeline
from robotransfer.utils.image_loading import (
    get_dataset_length,
    load_images_from_dataset,
    load_images_from_local,
)
from robotransfer.utils.save_video import save_images_to_mp4

def main():
    parser = argparse.ArgumentParser(description="Run RoboTransfer example.")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="HorizonRobotics/RoboTransfer-RealData",
        help="Path to the dataset.",
    )
    parser.add_argument(
        "--refer_image_path",
        type=str,
        default="assets/example_ref_image/gray_grid_desk.png",
        help="Path to the reference image.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="Path to save the output video.",
    )
    args = parser.parse_args()

    # Set the paths from the arguments
    dataset_path = args.dataset_path
    refer_image_path = args.refer_image_path
    output_dir = args.output_dir

    # Load the dataset
    if dataset_path.startswith("HorizonRobotics"):
        print(f"Loading dataset from Hugging Face: {dataset_path}")
        dataset = load_dataset(dataset_path)
        load_loacal_dataset = False
        length = len(dataset["train"])
    else:
        print(f"Loading local dataset from local path: {dataset_path}")
        load_loacal_dataset = True
        length = get_dataset_length(dataset_path)

    pipe = RoboTransferPipeline.from_pretrained(
        "HorizonRobotics/RoboTransfer",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    pipe.to("cuda")

    frames = []
    for i in range(0, length - 30, 30):
        if load_loacal_dataset:
            depth_guider_images, normal_guider_images = load_images_from_local(
                dataset_path, frames_start=i, frames_end=i + 30
            )
            save_video = False

        else:
            depth_guider_images, normal_guider_images, save_video = (
                load_images_from_dataset(
                    dataset, frames_start=i, frames_end=i + 30
                )
            )

        frames += pipe(
            image=Image.open(refer_image_path),
            depth_guider_images=depth_guider_images,
            normal_guider_images=normal_guider_images,
            min_guidance_scale=1.0,
            max_guidance_scale=3,
            height=384,
            width=640 * 3,
            num_frames=30,
            num_inference_steps=25,
        ).frames[0]

        if save_video:
            save_images_to_mp4(
                frames, f"{output_dir}/output_frames_final.mp4", fps=10
            )
        save_images_to_mp4(frames, f"{output_dir}/output_frames.mp4", fps=10)


if __name__ == "__main__":
    main()
