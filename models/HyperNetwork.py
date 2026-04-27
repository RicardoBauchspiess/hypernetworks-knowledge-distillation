import torch.nn as nn
import torch.nn.functional as F

import layers.hyper_layers as hl
import layers.hyper_blocks as hb
import layers.utils as utils
from configs.model.hypernetwork_config import *


class HyperNet(nn.Module):
    def __init__(self, num_blocks, num_classes=100, z_dim=128, **kwargs):
        super().__init__()

        self.hyper_trunk = hb.HyperTrunk(num_classes, z_dim)

        self.in_c = kwargs.get('in_c', 16)
        norm_layer = kwargs.get('norm_layer', nn.BatchNorm2d)
        norm_groups = kwargs.get('norm_groups', None)

        # Stem (no activation)
        self.conv1 = nn.Conv2d(3, self.in_c, 3, padding=1, bias=False)
        self.norm1 = utils.make_norm(self.in_c, norm_layer, norm_groups)

        # Build stages
        stage_configs = self.get_stage_configs(num_blocks, **kwargs)

        c = self.in_c
        self.layers = nn.ModuleList()

        for i, cfg in enumerate(stage_configs):
            layer = self._make_layer(
                in_c=c,
                num_blocks=num_blocks[i],
                z_dim=z_dim,
                **cfg
            )

            self.layers.append(layer)
            c = cfg["out_c"]

        # Classifier
        self.final_c = c
        self.dropout_rate = kwargs.get("dropout", 0.2)
        self.fc = nn.Linear(self.final_c, num_classes)

    # =====================================================
    def get_stage_configs(self, num_blocks, **kwargs):
        """
        Returns per-stage configuration dicts, merging defaults with user overrides.
        """

        base_c = self.in_c

        default_stage_configs = [
            dict(
                out_c=base_c,
                stride=1,
                conv_layer=nn.Conv2d,
            ),
            dict(
                out_c=base_c * 2,
                stride=2,
                conv_layer=hl.ResidualHyperConv2d,
                modulate=True,
                decompose=True,
            ),
            dict(
                out_c=base_c * 4,
                stride=2,
                conv_layer=hl.ResidualHyperConv2d,
                modulate=True,
                decompose=True,
            ),
        ]

        user_stage_configs = kwargs.get("stage_configs", {})

        stage_configs = []

        for i, default_cfg in enumerate(default_stage_configs):
            if i >= len(num_blocks):
                break

            cfg = default_cfg.copy()

            if i in user_stage_configs:
                cfg.update(user_stage_configs[i])

            stage_configs.append(cfg)

        return stage_configs

    # =====================================================
    def _make_layer(self, in_c, num_blocks, out_c, stride=1, **block_kwargs):
        layers = []

        for i in range(num_blocks):
            s = stride if i == 0 else 1

            layers.append(
                hb.ResBlock(
                    in_c=in_c if i == 0 else out_c,
                    out_c=out_c,
                    stride=s,
                    **block_kwargs
                )
            )

        return hb.HyperSequential(*layers)

    # =====================================================
    def forward(self, x, prior, iter=None, dropout = None):
        
        if dropout is None:
            dropout = self.dropout_rate
        
        # Generate conditioning
        z = self.hyper_trunk(prior)

        # Stem (no activation)
        out = self.conv1(x)
        out = self.norm1(out)

        # Stages
        for layer in self.layers:
            out = layer(out, z, iter)

        # Head
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)

        out = F.dropout(out, p=dropout, training=self.training)
        out = self.fc(out)

        return out
    
# =========================
# HyperResNet20 factory
# =========================
def HyperResNet20(num_classes=100, override = None):
    cfg = DEFAULT_CONFIG.copy()

    # Modify experiment
    if override is not None:
        cfg = merge_configs(cfg, override)

    model = HyperNet(
        num_blocks=[3, 3, 3],
        num_classes=num_classes,
        **cfg
    )
    return model