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


import imageio
import numpy as np


def save_images_to_mp4(images, output_path, fps=30):
    frames = [np.array(frame) for frame in images]
    imageio.mimsave(output_path, frames, fps=fps, codec="libx264")
