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

import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin


class GuiderNet(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(self, in_channels=3, mid_channels=4, out_channels=8):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 4, 2, 1),
            nn.SiLU(),
            nn.Conv2d(mid_channels, mid_channels, 4, 2, 1),
            nn.SiLU(),
            nn.Conv2d(mid_channels, out_channels, 4, 2, 1),
        )

    def forward(self, x):
        return self.layers(x)
