import torch
torch.autograd.set_detect_anomaly(False)
import torch.nn as nn
import torch.nn.functional as F

import nvtx

# Positional encoding (section 5.1)
class PositionalEncoder(nn.Module):
    def __init__(self, in_channels, degrees, include_inputs=True):
        super(PositionalEncoder, self).__init__()
        embed_fns = []
        out_channels = 0

        # Optionally include the inputs in the encoding.
        # The supplementary material in the NSFF paper says that this is set to False for the
        # static NeRF, but the code seems to keep it as True.
        if include_inputs:
            embed_fns.append(lambda x: x)
            out_channels += in_channels

        # Encoding is powers of 2 times sin and cos of input
        freq_bands = 2.**torch.linspace(0., degrees-1, steps=degrees)
        for freq in freq_bands:
            for p_fn in [torch.sin, torch.cos]:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_channels += in_channels

        self.embed_fns = embed_fns
        self.out_channels = out_channels

    @nvtx.annotate("Positional encoding")
    def forward(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


# Base NeRF model - returns RGB and Density values, as well as an intermediate output for processing
# by the Scene Flow Field part of the Dynamic NeRF model
class BaseNeRF(nn.Module):
    def __init__(self, position_channels, view_channels, position_encoding_degrees, view_encoding_degrees):
        super(BaseNeRF, self).__init__()

        self.position_channels = position_channels
        self.view_channels = view_channels

        # In the original code, these parameters were passed into constructor, but they never changed.
        # We will hard-code network parameters, based on the specific NeRF implementation.
        D = 8
        W = 256

        # Positional encoding will be used for both position and viewing angle, as in the original NeRF paper
        # Subsequent implementations used spherical harmonics for viewing angle, which seems better
        self.position_encoder = PositionalEncoder(position_channels, position_encoding_degrees)
        self.view_encoder = PositionalEncoder(view_channels, view_encoding_degrees)

        self.encoded_position_channels = self.position_encoder.out_channels
        self.encoded_view_channels = self.view_encoder.out_channels

        # "We follow the DeepSDF [32] architecture and include a skip connection that concatenates this input to the fifth layer’s activation"
        self.skips = [4]
        self.pts_linears = nn.ModuleList(
            [nn.Linear(self.encoded_position_channels, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + self.encoded_position_channels, W) for i in range(D-1)])

        # "An additional layer outputs the volume density σ..."
        self.density_linear = nn.Linear(W, 1)

        #  ... and a 256-dimensional feature vector."
        self.feature_linear = nn.Linear(W, W)

        # "This feature vector is concatenated with the positional encoding of the input viewing direction (γ(d)),
        #  and is processed by an additional fully-connected ReLU layer with 128 channels."
        self.views_linear = nn.Linear(self.encoded_view_channels + W, W//2)

        # "A final layer (with a sigmoid activation) outputs the emitted RGB radiance at position x,
        #  as viewed by a ray with direction d."
        self.rgb_linear = nn.Linear(W//2, 3)

    @nvtx.annotate("NeRF Forward")
    def forward(self, x):
        input_position, input_view = torch.split(x, [self.position_channels, self.view_channels], dim=-1)

        # The positional encoding of the input location (γ(x)) is passed through 8 fully-connected ReLU layers, each with 256 channels.
        # We follow the DeepSDF [32] architecture and include a skip connection that concatenates this input to the fifth layer’s activation.
        encoded_position = self.position_encoder(input_position)
        h = encoded_position
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([encoded_position, h], -1)

        # "An additional layer outputs the volume density σ..."
        density = self.density_linear(h)

        #  ... and a 256-dimensional feature vector."
        feature = self.feature_linear(h)

        # This feature vector is concatenated with the positional encoding of the input viewing direction (γ(d)),
        # and is processed by an additional fully-connected ReLU layer with 128 channels.
        encoded_view = self.view_encoder(input_view)
        x = torch.cat([feature, encoded_view], -1)
        x = F.relu(self.views_linear(x))

        # "A final layer (with a sigmoid activation) outputs the emitted RGB radiance at position x,
        #  as viewed by a ray with direction d."
        rgb = self.rgb_linear(x)

        return h, rgb, density

# Dynamic NeRF model for dynamic portions of the scene
# Generates additional scene flow field vectors and an disocclusion blending factor
class DynamicNeRF(BaseNeRF):
    def __init__(self):
        super(DynamicNeRF, self).__init__(position_channels=4, view_channels=3, position_encoding_degrees=10, view_encoding_degrees=4)
        W = 256

        self.scene_flow_linear = nn.Linear(W, 6)
        self.disocclusion_linear = nn.Linear(W, 2)

    @nvtx.annotate("DynamicNeRF forward")
    def forward(self, x):
        h, rgb, density = super().forward(x)

        scene_flow = F.tanh(self.scene_flow_linear(h))
        disocclusion_blend = F.sigmoid(self.disocclusion_linear(h))

        return torch.cat([rgb, density, scene_flow, disocclusion_blend], dim=-1)

# Static  NeRF model for static portions of the scene
# Adds an additional "blending" weight to blend the static and dynamic portions of a scene
class StaticNeRF(BaseNeRF):
    def __init__(self):
        super(StaticNeRF, self).__init__(position_channels=3, view_channels=3, position_encoding_degrees=10, view_encoding_degrees=4)
        W = 256

        self.blending_linear = nn.Linear(W, 1)

    @nvtx.annotate("StaticNeRF forward")
    def forward(self, x):
        h, rgb, density = super().forward(x)
        blending = F.sigmoid(self.blending_linear(h))

        return torch.cat([rgb, density, blending], dim=-1)
