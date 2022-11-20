import torch
torch.autograd.set_detect_anomaly(False)
import torch.nn as nn
import torch.nn.functional as F

import nvtx

# Base NeRF model - returns RGB and Density values, as well as an intermediate output for processing
# by the Scene Flow Field part of the Dynamic NeRF model
class BaseNeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False):
        super(BaseNeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs

        # "We follow the DeepSDF [32] architecture and include a skip connection that concatenates this input to the fifth layer’s activation"
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)])

        # "An additional layer outputs the volume density σ..."
        self.density_linear = nn.Linear(W, 1)

        if use_viewdirs:
            #  ... and a 256-dimensional feature vector."
            self.feature_linear = nn.Linear(W, W)

            # "This feature vector is concatenated with the positional encoding of the input viewing direction (γ(d)),
            #  and is processed by an additional fully-connected ReLU layer with 128 channels."
            self.views_linear = nn.Linear(input_ch_views + W, W//2)

            # "A final layer (with a sigmoid activation) outputs the emitted RGB radiance at position x,
            #  as viewed by a ray with direction d."
            self.rgb_linear = nn.Linear(W//2, 3)
        else:
            self.rgb_linear = nn.Linear(W, 3)

    @nvtx.annotate("NeRF Forward")
    def forward(self, x):
        input_position, input_viewdir = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)

        # The positional encoding of the input location (γ(x)) is passed through 8 fully-connected ReLU layers, each with 256 channels.
        # We follow the DeepSDF [32] architecture and include a skip connection that concatenates this input to the fifth layer’s activation.
        h = input_position
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_position, h], -1)

        # "An additional layer outputs the volume density σ..."
        density = self.density_linear(h)

        if self.use_viewdirs:
            #  ... and a 256-dimensional feature vector."
            feature = self.feature_linear(h)

            # This feature vector is concatenated with the positional encoding of the input viewing direction (γ(d)),
            # and is processed by an additional fully-connected ReLU layer with 128 channels.
            x = torch.cat([feature, input_viewdir], -1)
            x = F.relu(self.views_linear(x))

            # "A final layer (with a sigmoid activation) outputs the emitted RGB radiance at position x,
            #  as viewed by a ray with direction d."
            rgb = self.rgb_linear(x)
        else:
            rgb = self.rgb_linear(h)

        return h, rgb, density
