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
import logging
import pickle
from pathlib import Path

import cv2
import h5py
import numpy as np
from tqdm import tqdm

# Configure logging
logger = logging.getLogger(__name__)


class SimDataProcessor:
    """Processes simulation episode data from either HDF5 files or directories of pickle files."""

    def __init__(self, output_base_dir: str, camera_names: list):
        self.output_base_dir = Path(output_base_dir)
        self.camera_names = camera_names
        # Ensure the base output directory exists
        self.output_base_dir.mkdir(parents=True, exist_ok=True)

    def _read_image_from_bytes(self, img_bytes: bytes) -> np.ndarray:
        """Decodes JPEG bytes into a BGR NumPy array."""
        # The user's provided HDF5 creation script uses null-byte padding.
        nparr = np.frombuffer(img_bytes.rstrip(b"\0"), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img

    def _save_images(
        self,
        output_root: Path,
        camera_name: str,
        frame_idx: int,
        rgb_img: np.ndarray,
        depth_img: np.ndarray,
    ):
        """Saves RGB and Depth images to the correct directory."""
        try:
            # Normalize camera name for file paths if needed
            key_name = (
                "middle_camera"
                if camera_name == "head_camera"
                else camera_name
            )

            # Create specific directories for RGB and Depth
            rgb_dir = output_root / key_name / "rgb"
            depth_dir = output_root / key_name / "depth"
            rgb_dir.mkdir(parents=True, exist_ok=True)
            depth_dir.mkdir(parents=True, exist_ok=True)

            # Save RGB as JPEG
            rgb_path = rgb_dir / f"{frame_idx}.jpg"
            cv2.imwrite(str(rgb_path), rgb_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if depth_img is not None:
                # Save Depth as 16-bit PNG
                depth_img_uint16 = (np.round(depth_img)).astype(np.uint16)
                depth_path = depth_dir / f"{frame_idx}.png"
                cv2.imwrite(
                    str(depth_path),
                    depth_img_uint16,
                    [cv2.IMWRITE_PNG_COMPRESSION, 0],
                )
            else:
                logger.warning(
                    f"Depth image is None for frame {frame_idx}, camera {camera_name}. Skipping saving depth image."
                )
        except Exception as e:
            logger.error(
                f"Error saving images for frame {frame_idx}, camera {camera_name}: {e}"
            )

    def _process_hdf5_episode(
        self, episode_path: Path, episode_name: str, sample_rate: int
    ):
        """Processes a single HDF5 episode file with low memory usage."""
        logger.info(f"Processing HDF5 episode: {episode_name}")
        episode_output_root = self.output_base_dir / episode_name

        try:
            with h5py.File(episode_path, "r") as hf:
                if "observation" not in hf:
                    logger.error(
                        f"HDF5 file {episode_path} is missing 'observation' group. Aborting."
                    )
                    return

                # Determine total frames from the first available camera's data
                total_frames = 0
                for cam_name in self.camera_names:
                    rgb_path = f"observation/{cam_name}/rgb"
                    if rgb_path in hf:
                        total_frames = hf[rgb_path].shape[0]
                        break

                if total_frames == 0:
                    logger.error(
                        f"Could not determine frame count from any specified camera in {episode_path}. Aborting."
                    )
                    return

                frame_indices_to_process = range(0, total_frames, sample_rate)
                processed_count = 0

                for frame_idx in tqdm(
                    frame_indices_to_process,
                    desc=f"Processing frames in {episode_name}",
                    ncols=100,
                ):
                    frame_has_data = False
                    for camera_name in self.camera_names:
                        try:
                            # Access data for the specific frame index directly from the file
                            encoded_rgb = hf[f"observation/{camera_name}/rgb"][
                                frame_idx
                            ]
                            rgb_img = self._read_image_from_bytes(encoded_rgb)

                            try:
                                depth_img = hf[
                                    f"observation/{camera_name}/depth"
                                ][frame_idx]
                            except KeyError:
                                # If depth data is missing, set it to None
                                depth_img = None

                            if rgb_img is None:
                                logger.warning(
                                    f"Failed to decode or load data for frame {frame_idx}, camera {camera_name}."
                                )
                                continue

                            self._save_images(
                                episode_output_root,
                                camera_name,
                                frame_idx,
                                rgb_img,
                                depth_img,
                            )
                            frame_has_data = True

                        except KeyError:
                            # This is expected if a camera/modality is missing, so debug level is fine.
                            logger.debug(
                                f"Data for camera '{camera_name}' not found for frame {frame_idx}."
                            )
                        except Exception as e:
                            logger.error(
                                f"An unexpected error occurred processing frame {frame_idx}, camera {camera_name}: {e}"
                            )
                    if frame_has_data:
                        processed_count += 1

                logger.info(
                    f"Finished processing {processed_count} frames from '{episode_path}'. Data saved to {episode_output_root}."
                )

        except Exception as e:
            logger.error(
                f"Failed to open or process HDF5 file {episode_path}: {e}"
            )

    def _process_pkl_episode(
        self, episode_path: Path, episode_name: str, sample_rate: int
    ):
        """Processes an episode from a directory of pickle files."""
        logger.info(f"Processing Pickle directory episode: {episode_name}")
        episode_output_root = self.output_base_dir / episode_name

        # Discover and sort pickle files
        try:
            pkl_files_info = [
                (int(p.stem), p)
                for p in episode_path.glob("*.pkl")
                if p.stem.isdigit()
            ]
            if not pkl_files_info:
                logger.warning(
                    f"No numerically named .pkl files found in '{episode_path}'. Skipping."
                )
                return
            pkl_files_info.sort()

            # Validate sequence continuity
            for i, (num, _) in enumerate(pkl_files_info):
                if i != num:
                    logger.error(
                        f"Missing pickle file {i}.pkl in {episode_path}. Aborting."
                    )
                    return
        except Exception as e:
            logger.error(
                f"Error discovering or validating pickle files in {episode_path}: {e}"
            )
            return

        files_to_process = pkl_files_info[::sample_rate]
        processed_count = 0

        for frame_idx, file_path in tqdm(
            files_to_process,
            desc=f"Processing frames in {episode_name}",
            ncols=100,
        ):
            try:
                with open(file_path, "rb") as f:
                    frame_data = pickle.load(f)

                if (
                    "observation" not in frame_data
                    or not frame_data["observation"]
                ):
                    logger.warning(
                        f"Skipping {file_path}: Missing or empty 'observation' data."
                    )
                    continue

                frame_has_data = False
                for camera_name in self.camera_names:
                    cam_data = frame_data["observation"].get(camera_name, {})
                    rgb_img = cam_data.get("rgb")
                    depth_img = cam_data.get("depth")

                    if rgb_img is not None and depth_img is not None:
                        self._save_images(
                            episode_output_root,
                            camera_name,
                            frame_idx,
                            rgb_img,
                            depth_img,
                        )
                        frame_has_data = True

                if frame_has_data:
                    processed_count += 1

            except Exception as e:
                logger.error(
                    f"Failed to load or process pickle file {file_path}: {e}"
                )

        logger.info(
            f"Finished processing {processed_count} frames from '{episode_path}'. Data saved to {episode_output_root}."
        )

    def process_episode(self, episode_path: str, sample_rate: int = 1):
        """Processes an episode, dispatching to the correct handler based on input type.

        Args:
            episode_path (str): Path to the episode (either a .hdf5 file or a directory).
            sample_rate (int): Rate at which to sample frames.
        """
        episode_path_obj = Path(episode_path)

        if episode_path_obj.is_file() and episode_path_obj.suffix == ".hdf5":
            self._process_hdf5_episode(
                episode_path_obj, episode_path_obj.stem, sample_rate
            )
        elif episode_path_obj.is_dir():
            self._process_pkl_episode(
                episode_path_obj, episode_path_obj.name, sample_rate
            )
        else:
            logger.error(
                f"Input path '{episode_path}' is not a valid .hdf5 file or directory. Aborting."
            )
            return


# Keep your main function as it is, it works correctly with the refactored class.
def main():
    parser = argparse.ArgumentParser(
        description="Process an episode, supporting either a single HDF5 file or a directory of pickle files, for RGB and Depth data."
    )
    # ... (rest of your main function is unchanged) ...
    parser.add_argument(
        "--episode_path",
        type=str,
        required=True,
        help="Full path to the episode (e.g., /path/to/data/episode0.hdf5 or /path/to/data/episode0/).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output/tmp/robotwin_processed_images",
        help="Base output directory for processed image files",
    )
    parser.add_argument(
        "--camera_names",
        type=str,
        nargs="+",
        default=["head_camera", "left_camera", "right_camera"],
        help="List of camera names to process",
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=1,
        help="Sample rate for processing frames within the episode (e.g., 2 to process every other frame).",
    )
    args = parser.parse_args()

    # Set logging level to INFO for general use. Use DEBUG for more verbose output.
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    processor = SimDataProcessor(
        output_base_dir=args.output_dir,
        camera_names=args.camera_names,
    )

    processor.process_episode(
        episode_path=args.episode_path, sample_rate=args.sample_rate
    )


if __name__ == "__main__":
    main()
