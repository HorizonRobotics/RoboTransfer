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
