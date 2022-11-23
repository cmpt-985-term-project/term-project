#
# NeRF based on NVIDIA's CUTLASS MLP
#
# This should be numerically identical to the original NeRF implementation, whereas
# the FusedMLP implementation will be different, owing to its limitation of a width of
# 128 neurons per layer.

import torch
torch.autograd.set_detect_anomaly(False)
import torch.nn as nn
import torch.nn.functional as F

import tinycudann as tcnn
import json
import nvtx

class CutlassNeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4]):
        super(CutlassNeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips

        # To simplify implementation, we will assume skips = [4]
        if skips != [4]:
            raise "Positional encoding skip connection must be at layer 4 for Cutlass NeRF implementation"

        # Backbone network is split into two 4-layer MLPs
        half_backbone_json = json.loads(f'''
        {{
            "otype": "CutlassMLP",
            "activation": "ReLU",
            "output_activation": "ReLU",
            "n_neurons": {W},
            "n_hidden_layers": 3,
            "feedback_alignment": false
        }}
        ''')

        # "We follow the DeepSDF [32] architecture and include a skip connection that concatenates this input to the fifth layer’s activation"
        self.backbone1 = tcnn.Network(n_input_dims=input_ch, n_output_dims=W, network_config=half_backbone_json)
        self.backbone2 = tcnn.Network(n_input_dims=(W+input_ch), n_output_dims=W, network_config=half_backbone_json)

        # "An additional layer outputs the volume density σ..."
        self.density_linear = nn.Linear(W, 1)

        #  ... and a 256-dimensional feature vector."
        self.feature_linear = nn.Linear(W, W)

        # "This feature vector is concatenated with the positional encoding of the input viewing direction (γ(d)),
        #  and is processed by an additional fully-connected ReLU layer with 128 channels."
        self.views_linear = nn.Linear(input_ch_views + W, W//2)

        # "A final layer (with a sigmoid activation) outputs the emitted RGB radiance at position x,
        #  as viewed by a ray with direction d." NOTE: sigmoid is happening in raw2outputs()
        self.rgb_linear = nn.Linear(W//2, 3)

    @nvtx.annotate("NeRF Forward")
    def forward(self, x):
        input_position, input_viewdir = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)

        h = self.backbone1(input_position)
        h = self.backbone2(torch.cat([input_position, h], dim=-1))

        # "An additional layer outputs the volume density σ..."
        density = self.density_linear(h)

        #  ... and a 256-dimensional feature vector."
        feature = self.feature_linear(h)

        # This feature vector is concatenated with the positional encoding of the input viewing direction (γ(d)),
        # and is processed by an additional fully-connected ReLU layer with 128 channels.
        x = torch.cat([feature, input_viewdir.half()], -1)
        x = F.relu(self.views_linear(x))

        # "A final layer (with a sigmoid activation) outputs the emitted RGB radiance at position x,
        #  as viewed by a ray with direction d."
        rgb = self.rgb_linear(x)

        return h, rgb, density


# Dynamic NeRF model for dynamic portions of the scene
# Generates additional scene flow field vectors and an disocclusion blending factor
class CutlassDynamicNeRF(CutlassNeRF):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4]):
        super(CutlassDynamicNeRF, self).__init__(D, W, input_ch, input_ch_views, output_ch, skips)

        self.scene_flow_linear = nn.Linear(W, 6, dtype=torch.float16)
        self.disocclusion_linear = nn.Linear(W, 2, dtype=torch.float16)

    @nvtx.annotate("Dynamic Cutlass NeRF forward")
    def forward(self, x):
        h, rgb, density = super().forward(x)

        scene_flow = F.tanh(self.scene_flow_linear(h))
        disocclusion_blend = F.sigmoid(self.disocclusion_linear(h))

        return torch.cat([rgb, density, scene_flow, disocclusion_blend], dim=-1)

class CutlassStaticNeRF(CutlassNeRF):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4]):
        super(CutlassStaticNeRF, self).__init__(D, W, input_ch, input_ch_views, output_ch, skips)

        self.blending_linear = nn.Linear(W, 1, dtype=torch.float16)

    @nvtx.annotate("Cutlass Static NeRF forward")
    def forward(self, x):
        h, rgb, density = super().forward(x)
        blending = F.sigmoid(self.blending_linear(h))

        return torch.cat([rgb, density, blending], dim=-1)
