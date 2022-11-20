
import torch
torch.autograd.set_detect_anomaly(False)
import torch.nn as nn
import torch.nn.functional as F

from models.base_nerf import BaseNeRF

import nvtx

# Dynamic NeRF model for dynamic portions of the scene
# Generates additional scene flow field vectors and an disocclusion blending factor
class DynamicNeRF(BaseNeRF):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=True):
        super(DynamicNeRF, self).__init__(D, W, input_ch, input_ch_views, output_ch, skips, use_viewdirs)

        self.scene_flow_linear = nn.Linear(W, 6)
        self.disocclusion_linear = nn.Linear(W, 2)

    @nvtx.annotate("DynamicNeRF forward")
    def forward(self, x):
        h, rgb, density = super().forward(x)

        scene_flow = F.tanh(self.scene_flow_linear(h))
        disocclusion_blend = F.sigmoid(self.disocclusion_linear(h))

        return torch.cat([rgb, density, scene_flow, disocclusion_blend], dim=-1)
