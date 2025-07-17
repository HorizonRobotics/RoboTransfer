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
import os
import sys

import cv2
import numpy as np
import torch
from huggingface_hub import snapshot_download

sys.path.append("submodule/Video-Depth-Anything")
import re

import imageio
import matplotlib.cm as cm
from utils.dc_utils import read_video_frames, save_video
from video_depth_anything.video_depth import VideoDepthAnything


def natural_sort_key(s):
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split(r"(\d+)", s)
    ]


def read_image_frames(
    image_folder,
    process_length=-1,
    target_fps=-1,
    max_res=-1,
    depth_mode=False,
):
    """Read image sequence and maintain compatibility with video processing interface.

    Args:
        image_folder (str): Path to the input image folder.
        process_length (int): Maximum number of frames to process, -1 means all.
        target_fps (int): Target output frames per second (fps), -1 means original fps.
        max_res (int): Maximum resolution edge length, -1 means keep original size.
        depth_mode (bool): Whether to treat images as depth maps (use nearest neighbor interpolation).

    Returns:
        frames (np.ndarray): Array of frames, shape (NumFrames, H, W, 3) or single channel (NumFrames, H, W).
        target_fps (int): The target fps, same as input.
        image_files (list): List of image files used.
    """

    def ensure_even(value):
        return value // 2 * 2

    valid_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    image_files = [
        os.path.join(image_folder, f)
        for f in sorted(os.listdir(image_folder), key=natural_sort_key)
        if f.lower().endswith(valid_extensions)
    ]

    if process_length > 0:
        image_files = image_files[:process_length]

    frames = []
    for img_path in image_files:
        # loading depth images in grayscale mode
        if depth_mode:
            img = cv2.imread(
                img_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE
            )
        else:
            img = cv2.imread(img_path)

        if img is None:
            print(f"Warning: Failed to read image {img_path}. Skipping.")
            continue

        if not depth_mode:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # keep original size if max_res is -1
        h, w = img.shape[:2]
        if max_res > 0 and max(h, w) > max_res:
            scale = max_res / max(h, w)
            new_h = ensure_even(round(h * scale))
            new_w = ensure_even(round(w * scale))
            interpolation = (
                cv2.INTER_NEAREST if depth_mode else cv2.INTER_LINEAR
            )
            img = cv2.resize(img, (new_w, new_h), interpolation=interpolation)

        frames.append(img)

    # Convert list of frames to numpy array
    frames = np.asarray(frames)
    return frames, target_fps, image_files


def save_muti_video(
    frames,
    output_video_path,
    fps=10,
    depths=None,
    sensor_depths=None,
    grayscale=False,
):
    """Save a video with frames, depths and sensor_depths side by side.

    Args:
        frames (np.ndarray): Array of frames, shape (N, H, W, 3).
        output_video_path (str): Path to save the output video.
        fps (int): Frames per second for the output video.
        depths (np.ndarray, optional): Depths corresponding to the frames, shape (N, H, W).
        sensor_depths (np.ndarray, optional): Sensor depths corresponding to the frames, shape (N, H, W).
        grayscale (bool): If True, visualize depths in grayscale instead of color.

    """
    writer = imageio.get_writer(
        output_video_path,
        fps=fps,
        macro_block_size=1,
        codec="libx264",
        ffmpeg_params=["-crf", "18"],
    )
    colormap = np.array(cm.get_cmap("turbo").colors)
    # d_min, d_max = sensor_depths.min(), sensor_depths.max()
    d_min, d_max = 0, 1500
    for i in range(frames.shape[0]):
        frame = frames[i]
        h, w = frame.shape[:2]
        vis = np.zeros((h, w * 3, 3), dtype=np.uint8)
        vis[:, :w, :] = frame
        if depths is not None:
            depth = depths[i]
            depth_norm = ((depth - d_min) / (d_max - d_min) * 255).astype(
                np.uint8
            )
            depth_vis = (
                (colormap[depth_norm] * 255).astype(np.uint8)
                if not grayscale
                else depth_norm
            )
            vis[:, w : w * 2, :] = depth_vis

        if sensor_depths is not None:
            sensor_depth = sensor_depths[i]
            sensor_depth_norm = (
                (sensor_depth - d_min) / (d_max - d_min) * 255
            ).astype(np.uint8)
            sensor_depth_vis = (
                (colormap[sensor_depth_norm] * 255).astype(np.uint8)
                if not grayscale
                else sensor_depth_norm
            )
            vis[:, w * 2 :, :] = sensor_depth_vis
        writer.append_data(vis)
    writer.close()


def compute_scale_and_shift_total(prediction, target, mask):
    """Compute the scale and shift between prediction and target using a linear regression approach.

    Args:
        prediction (torch.Tensor): Predicted depth values, shape (N, H, W).
        target (torch.Tensor): Ground truth depth values, shape (N, H, W).
        mask (torch.Tensor): Mask indicating valid pixels, shape (N, H, W).

    Returns:
        tuple: Scale and shift values, each of shape (N,).
    """
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(mask * prediction * prediction)
    a_01 = torch.sum(mask * prediction)
    a_11 = torch.sum(mask)

    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target)
    b_1 = torch.sum(mask * target)

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    # A needs to be a positive definite matrix.
    valid = det > 0

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[
        valid
    ]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[
        valid
    ]

    return x_0, x_1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video Depth Anything")
    parser.add_argument(
        "--input_video",
        type=str,
        default="./assets/example_videos/davis_rollercoaster.mp4",
    )
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--input_size", type=int, default=518)
    parser.add_argument("--max_res", type=int, default=1280)
    parser.add_argument(
        "--encoder", type=str, default="vitl", choices=["vits", "vitl"]
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=-1,
        help="maximum length of the input video, -1 means no limit",
    )
    parser.add_argument(
        "--target_fps",
        type=int,
        default=-1,
        help="target fps of the input video, -1 means the original fps",
    )
    parser.add_argument(
        "--fp32",
        action="store_true",
        help="model infer with torch.float32, default is torch.float16",
    )
    parser.add_argument(
        "--grayscale",
        action="store_true",
        help="do not apply colorful palette",
    )
    parser.add_argument(
        "--save_npz", action="store_true", help="save depths as npz"
    )
    parser.add_argument(
        "--save_exr", action="store_true", help="save depths as exr"
    )
    parser.add_argument(
        "--save_png",
        action="store_true",
        help="Save depth visualizations as PNG",
    )
    parser.add_argument(
        "--save_vis",
        action="store_true",
        help="Save depth visualizations as video",
    )
    parser.add_argument(
        "--depth_dir",
        type=str,
        default="",
        help="load sensor depth for scale alignment",
    )
    parser.add_argument(
        "--no_depth_scaled", action="store_true", help="output scaled depth"
    )
    parser.add_argument(
        "--filter_error_persentage",
        type=float,
        default=0.8,
        help="filter error percentage for scale and shift",
    )

    args = parser.parse_args()

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    model_configs = {
        "vits": {
            "encoder": "vits",
            "features": 64,
            "out_channels": [48, 96, 192, 384],
        },
        "vitl": {
            "encoder": "vitl",
            "features": 256,
            "out_channels": [256, 512, 1024, 1024],
        },
    }

    filter_error_percentage = args.filter_error_persentage
    video_depth_anything = VideoDepthAnything(**model_configs[args.encoder])

    if not os.path.exists("./submodule/Video-Depth-Anything/checkpoints"):
        os.makedirs(
            "./submodule/Video-Depth-Anything/checkpoints", exist_ok=True
        )

    checkoutpoint_path = f"./submodule/Video-Depth-Anything/checkpoints/video_depth_anything_{args.encoder}.pth"
    if not os.path.exists(checkoutpoint_path):
        print(f"Checkpoint not found at {checkoutpoint_path}, downloading...")
        if args.encoder == "vits":
            repo_id = "depth-anything/Video-Depth-Anything-Small"
        elif args.encoder == "vitl":
            repo_id = "depth-anything/Video-Depth-Anything-Large"

        snapshot_download(
            repo_id=repo_id,
            local_dir="./submodule/Video-Depth-Anything/checkpoints",
            repo_type="model",
            resume_download=True,
        )

    video_depth_anything.load_state_dict(
        torch.load(checkoutpoint_path, map_location="cpu"), strict=True
    )

    video_depth_anything = video_depth_anything.to(DEVICE).eval()

    if args.input_video.endswith(".mp4"):
        frames, target_fps = read_video_frames(
            args.input_video, args.max_len, args.target_fps, args.max_res
        )
        image_files = None
    elif os.path.isdir(args.input_video):
        # 图像处理调用
        frames, target_fps, image_files = read_image_frames(
            image_folder=args.input_video,
            process_length=args.max_len,
            target_fps=args.target_fps,
            max_res=args.max_res,
        )
    else:
        raise ValueError(
            "Unsupported input format. Please provide a video file or a directory of images."
        )

    if os.path.exists(args.depth_dir):
        sensor_depths, _, depth_files = read_image_frames(
            image_folder=args.depth_dir,
            process_length=args.max_len,
            target_fps=args.target_fps,
            max_res=args.max_res,
            depth_mode=True,
        )
        assert len(sensor_depths) == len(
            frames
        ), f"Mismatch between number of frames ({len(frames)}) and sensor depth ({len(sensor_depths)})"

    depths, fps = video_depth_anything.infer_video_depth(
        frames,
        target_fps,
        input_size=args.input_size,
        device=DEVICE,
        fp32=args.fp32,
    )

    # Depth scale and shift
    # mask if depths depth_frames is zeros
    del video_depth_anything
    torch.cuda.empty_cache()

    # import pdb; pdb.set_trace()
    if args.no_depth_scaled:
        print("No scale and shift computating ...")
    else:
        mask = (sensor_depths > 0) & (depths > 0)
        re_scale_mask = torch.from_numpy(mask).float()
        prediction_depth = torch.from_numpy(depths)  # 原始深度预测
        target = torch.from_numpy(sensor_depths.astype(np.float32)).float()

        print("Scale and shift computating ...")
        for _ in range(2):
            # 计算误差图 (绝对值误差)
            err_map = torch.abs(prediction_depth - target) * re_scale_mask

            # 动态计算误差阈值 (保留前80%的低误差区域)
            sorted_err, _ = torch.sort(err_map.view(-1).cpu())

            valid_pixels = (re_scale_mask == 1).sum().item()  # 有效像素数
            length = sorted_err.shape[0]
            start_index = int(length - valid_pixels)
            thr = sorted_err[start_index:]  # 跳过无效像素

            thr = thr[int(filter_error_percentage * len(thr))]

            # 更新误差掩码
            err_mask = (err_map < thr).float()

            re_scale_mask = err_mask * re_scale_mask

            # 重新计算尺度和偏移
            scale, shift = compute_scale_and_shift_total(
                prediction_depth,  # 原始预测深度
                target,  # 传感器深度真值
                re_scale_mask,  # 更新后的掩码
            )

        if scale == 0 or shift == 0:
            print("Scale or shift is zero, using original depths.")

        # 应用尺度偏移修正
        print(f"Scale: {scale}, Shift: {shift}")
        prediction_depth = scale.view(
            -1, 1, 1
        ) * prediction_depth + shift.view(-1, 1, 1)
        depths_scaled = depths * scale.cpu().numpy() + shift.cpu().numpy()

    video_name = os.path.basename(args.input_video)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if args.input_video.endswith(".mp4"):
        processed_video_path = os.path.join(
            args.output_dir, os.path.splitext(video_name)[0] + "_src.mp4"
        )
        depth_vis_path = os.path.join(
            args.output_dir, os.path.splitext(video_name)[0] + "_vis.mp4"
        )
        concat_video_path = os.path.join(
            args.output_dir, os.path.splitext(video_name)[0] + "_concat.mp4"
        )

    elif os.path.isdir(args.input_video):
        processed_video_path = os.path.join(args.output_dir, "src.mp4")
        depth_vis_path = os.path.join(args.output_dir, "vis.mp4")
        concat_video_path = os.path.join(args.output_dir, "concat.mp4")

    if args.save_png:
        os.makedirs(args.output_dir, exist_ok=True)
        if image_files is not None:
            assert len(image_files) == len(
                depths
            ), "Mismatch between number of images and depths"
            for i, img_path in enumerate(image_files):
                if args.no_depth_scaled:
                    depth = depths[i] * 10
                else:
                    depth = depths_scaled[i]  # scaled 1000
                image_name = os.path.basename(img_path)
                png_dir = os.path.join(
                    args.output_dir, f"{image_name[:-4]}.png"
                )
                depth[depth < 0] = 0
                cv2.imwrite(png_dir, np.round(depth).astype(np.uint16))
        else:
            for i, depth in enumerate(depths):
                png_dir = os.path.join(args.output_dir, f"frame_{i:05d}.png")
                cv2.imwrite(png_dir, np.round(depth).astype(np.uint16))

    if args.save_vis:
        save_video(frames, processed_video_path, fps=fps if fps > 0 else 10)
        save_video(
            depths,
            depth_vis_path,
            fps=fps if fps > 0 else 10,
            is_depths=True,
            grayscale=args.grayscale,
        )
        if os.path.exists(args.depth_dir):
            if args.no_depth_scaled:
                show_depths = depths
            else:
                show_depths = depths_scaled

            save_muti_video(
                frames=frames,
                depths=show_depths,
                sensor_depths=sensor_depths,
                output_video_path=concat_video_path,
                fps=fps if fps > 0 else 10,
                grayscale=args.grayscale,
            )

    print(f"Processed video saved at: {processed_video_path}")
