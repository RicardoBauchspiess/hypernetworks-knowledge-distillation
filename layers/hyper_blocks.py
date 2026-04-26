import torch
import torch.nn as nn
import torch.nn.functional as F

import layers.hyper_layers as hl
import layers.utils as utils

class HyperTrunk(nn.Module):
    def __init__(self, z_dim, hidden_dim=128, depth=2):
        super().__init__()

        layers = []
        in_dim = z_dim

        layers.append(nn.LayerNorm(in_dim))
        for _ in range(depth):
            layers.append(nn.Linear(in_dim, hidden_dim, bias = False))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.LayerNorm(hidden_dim))
        self.net = nn.Sequential(*layers)


    def forward(self, z):
        h = self.net(z)

        return h

# Wrapper to adapt conv layers to the hypernet format
class ConvWrapper(nn.Module):
    def __init__(self, conv):
        super().__init__()
        self.conv = conv

    def forward(self, x, z=None, iter=None):
        return self.conv(x)
    
# Wrapper for sequential with hyperlayers
class HyperSequential(nn.Module):
    def __init__(self, *layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x, z, iter=None):
        for layer in self.layers:
            x = layer(x, z, iter=iter)
        return x

# Standard residual block using any combination of hyperconv2d variations and regular conv2d
class ResBlock(nn.Module):
    def __init__(
        self,
        in_c,
        out_c,
        k,
        z_dim,
        stride=1,
        groups=1,
        rank=None, # LoRA style low rank for hyper layers
        compression=None, # Alternative calculation of low rank
        modulate=True,
        decompose=True,
        conv_layer=hl.ResidualHyperConv2d, # Use tuple, list or dict to combine different types of layers
        act=F.relu,
        norm_layer=nn.BatchNorm2d,
        norm_groups=8, # Groups if GroupNorm is used
    ):
        super().__init__()

        base_params = {
            "z_dim": z_dim,
            "groups": groups,
            "rank": rank,
            "compression": compression,
            "modulate": modulate,
            "decompose": decompose,
        }

        # Conv parameters
        params1 = {
            **base_params,
            "in_c": in_c,
            "out_c": out_c,
            "stride": stride,
            "k": k,
        }

        params2 = {
            **base_params,
            "in_c": out_c,
            "out_c": out_c,
            "stride": 1,
            "k": k,
        }

        params_shortcut = {
            **base_params,
            "in_c": in_c,
            "out_c": out_c,
            "stride": stride,
            "k": 1, 
        }

        conv1_layer, conv2_layer, shortcut_layer = self._resolve_conv_layers(conv_layer)

        # Main path
        self.conv1 = self._make_conv(params1, conv1_layer)
        self.norm1 = utils.make_norm(out_c, norm_layer, norm_groups)
        self.act = act if callable(act) else act()


        self.conv2 = self._make_conv(params2, conv2_layer)
        self.norm2 = utils.make_norm(out_c, norm_layer, norm_groups)

        # Shortcut
        self.has_shortcut = stride != 1 or in_c != out_c
        if self.has_shortcut:
            self.shortcut = self._make_conv(params_shortcut, shortcut_layer)
        else:
            self.shortcut = nn.Identity()

    # =====================================================
    def _resolve_conv_layers(self, conv_layer):
        # Single layer
        if not isinstance(conv_layer, (list, tuple, dict)):
            return conv_layer, conv_layer, conv_layer

        # Tuple/list: (conv1, conv2, shortcut)
        if isinstance(conv_layer, (list, tuple)):
            conv1 = conv_layer[0]
            conv2 = conv_layer[1] if len(conv_layer) > 1 else conv1
            shortcut = conv_layer[2] if len(conv_layer) > 2 else conv2
            return conv1, conv2, shortcut

        # Dict: explicit override
        if isinstance(conv_layer, dict):
            default = conv_layer.get("default", hl.ResidualHyperConv2d)
            conv1 = conv_layer.get("conv1", default)
            conv2 = conv_layer.get("conv2", conv1)
            shortcut = conv_layer.get("shortcut", conv2)
            return conv1, conv2, shortcut

        raise ValueError(f"Unsupported conv_layer type: {type(conv_layer)}")

    # =====================================================
    def _make_conv(self, params, conv_layer):
        if conv_layer is nn.Conv2d:
            # Generate conv layer and wrap it with hyperconv format
            conv =  nn.Conv2d(
                in_channels=params["in_c"],
                out_channels=params["out_c"],
                kernel_size=params["k"],
                stride=params.get("stride", 1),
                groups=params.get("groups", 1),
                bias=False,
            )
            return ConvWrapper(conv)

        allowed_hyperconvs = {
            hl.HyperConv2d,
            hl.HyperModulatedConv2d,
            hl.ResidualHyperConv2d,
            hl.DecomposedHyperConv2d,
        }

        if conv_layer in allowed_hyperconvs:
            return conv_layer(**params)

        raise ValueError(f"Unsupported conv_layer: {conv_layer}")

    # =====================================================
    def forward(self, x, z, iter=None):
        out = self.conv1(x, z, iter=iter)
        out = self.norm1(out)
        out = self.act(out)

        out = self.conv2(out, z, iter=iter)
        out = self.norm2(out)

        shortcut = x if not self.has_shortcut else self.shortcut(x, z)

        out = out + shortcut

        return out