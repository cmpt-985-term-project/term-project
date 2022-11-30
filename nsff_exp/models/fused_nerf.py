#
# NeRF based on NVIDIA's FusedMLP
#
# From Instant NGP, section 5.4:
# "Informed by the analysis in Figure 10, our results were generated with a 
#  1-hidden-layer density MLP and a 2-hidden-layer color MLP, both 64 neurons wide."
#
# For Neural Scene Flow Fields, we will have two NeRFs:
#   - A Dynamic NeRF, which uses 4 position channels (3-space, 1-time)
#   - A Static NeRF, which uses 3 position channels (3-space)
#
# Each NeRF will have a "density" network, which will hash-encode the position channels and produce the following:
#   - For the Dynamic NeRF: the forward and backward scene flow vectors, forward and backward
#     "disocclusion" weights, and a 16-dimensional feature vector with the first dimension as density
#   - For the Static NeRF: a blending weight to blend the static and dynamic RGB and density values,
#     and a 16-dimensional feature vector with the first dimension as density
#
# Both NeRFs will also concatenate the 16-dimensional feature vector with a viewing direction encoded onto a
# spherical harmonic basis of degree 4 (so, also 16-dimensional). This 32-dimensional input will be fed into
# a 4-layer "color" MLP which will produce an RGB color value for the dynamic and static portions of the scene.

import torch
torch.autograd.set_detect_anomaly(False)
import torch.nn as nn

import tinycudann as tcnn
import json
import nvtx

# A "Density" (not view-angle dependent) MLP
class FusedDensityMLP(nn.Module):
    def __init__(self, in_channels, out_channels, degrees=10):
        super(FusedDensityMLP, self).__init__()

        # Network parameters
        self.W = 128

        encoding_config = json.loads(f'{{"otype":"Frequency", "n_frequencies":{degrees}}}')
        self.position_encoder = tcnn.Encoding(n_input_dims=in_channels, encoding_config=encoding_config)

        network_config1 = json.loads(f'''
            {{"otype":"FullyFusedMLP", "activation":"ReLU", "output_activation":"None", "n_neurons":{self.W},
              "n_hidden_layers":2, "feedback_alignment":false}}''')
        self.model_part1 = tcnn.Network(n_input_dims=self.position_encoder.n_output_dims, n_output_dims=self.W, network_config=network_config1)

        network_config2 = json.loads(f'''
            {{"otype":"FullyFusedMLP", "activation":"ReLU", "output_activation":"None", "n_neurons":{self.W},
              "n_hidden_layers":3, "feedback_alignment":false}}''')
        self.model_part2 = tcnn.Network(n_input_dims=self.W + self.position_encoder.n_output_dims, n_output_dims=out_channels, network_config=network_config2)

    def forward(self, x):
        encoded_position = self.position_encoder(x)
        part1 = self.model_part1(encoded_position)
        part2 = self.model_part2(torch.cat([encoded_position, part1], dim=-1))
        return part2

# An "Color" (view-angle dependent) MLP
class FusedColorMLP(nn.Module):
    def __init__(self, degrees=4):
        super(FusedColorMLP, self).__init__()
        self.W = 128

        # For consistency with original paper, we will use the position encoder on the viewing angle,
        # even though a spherical harmonic encoder makes more sense.
        encoding_config = json.loads(f'{{"otype":"Frequency", "n_frequencies":{degrees}}}')
        self.view_encoder = tcnn.Encoding(n_input_dims=3, encoding_config=encoding_config)

        network_config = json.loads(f'''
            {{"otype":"FullyFusedMLP", "activation":"ReLU", "output_activation":"None", "n_neurons":{self.W},
              "n_hidden_layers":1, "feedback_alignment":false}}''')
        self.model = tcnn.Network(n_input_dims=self.view_encoder.n_output_dims + self.W, n_output_dims=3, network_config=network_config)

    def forward(self, x):
        input_view, feature_vector = x.split([3, self.W], dim=-1)
        encoded_view = self.view_encoder(input_view)
        return self.model(torch.cat([encoded_view, feature_vector], dim=-1))


# Dynamic NeRF model for dynamic portions of the scene
class FusedDynamicNeRF(nn.Module):
    def __init__(self):
        super(FusedDynamicNeRF, self).__init__()
        self.W = 128

        # 24 channels = scene flow (2 x 3-dim) + disocclusion weights (2 x 1-dim) + density and feature vector (128-dim)
        self.density_mlp = FusedDensityMLP(in_channels=4, out_channels=self.W + 8)
        self.color_mlp = FusedColorMLP()

    @nvtx.annotate("Fused Dynamic NeRF forward")
    def forward(self, x):
        # 4 input channels, 3 view channels
        input_position, input_view = x.split([4, 3], dim=-1)
        x = self.density_mlp(input_position)

        # 2 x 3-dim scene flow, 2 x 1-dim disocclusion blend, 128-dim feature vector
        scene_flow, disocclusion_blend, feature_vector = torch.split(x, [6, 2, self.W], dim=-1)

        scene_flow = torch.tanh(scene_flow)
        disocclusion_blend = torch.sigmoid(disocclusion_blend)
        density = feature_vector[:, 0:1]

        rgb = self.color_mlp(torch.cat([input_view, feature_vector], dim=-1))

        return torch.cat([rgb, density, scene_flow, disocclusion_blend], dim=-1)

# Static NeRF model for static portions of the scene
class FusedStaticNeRF(nn.Module):
    def __init__(self):
        super(FusedStaticNeRF, self).__init__()
        self.W = 128

        # 17 channels = static/dynamic blending weight (1-dim) + density and feature vector (128-dim)
        self.density_mlp = FusedDensityMLP(in_channels=3, out_channels=self.W+1)
        self.color_mlp = FusedColorMLP()

    @nvtx.annotate("Fused Static NeRF forward")
    def forward(self, x):
        # 3 input channels, 3 view channels
        input_position, input_view = x.split([3, 3], dim=-1)
        x = self.density_mlp(input_position)

        # 1-dim blending weight, 128-dim feature vector
        blending, feature_vector = x.split([1, self.W], dim=-1)

        blending = torch.sigmoid(blending)
        density = feature_vector[:, 0:1]

        rgb = self.color_mlp(torch.cat([input_view, feature_vector], dim=-1))

        return torch.cat([rgb, density, blending], dim=-1)
