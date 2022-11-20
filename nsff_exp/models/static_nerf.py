import torch
torch.autograd.set_detect_anomaly(False)
import torch.nn as nn
import torch.nn.functional as F

from models.base_nerf import BaseNeRF

import nvtx

# Static  NeRF model for static portions of the scene
# Adds an additional "blending" weight to blend the static and dynamic portions of a scene
class StaticNeRF(BaseNeRF):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=True):
        super(StaticNeRF, self).__init__(D, W, input_ch, input_ch_views, output_ch, skips, use_viewdirs)

        self.blending_linear = nn.Linear(W, 1)

    @nvtx.annotate("StaticNeRF forward")
    def forward(self, x):
        h, rgb, density = super().forward(x)
        blending = F.sigmoid(self.blending_linear(h))

        return torch.cat([rgb, density, blending], dim=-1)
