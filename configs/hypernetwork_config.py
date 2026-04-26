import torch.nn as nn
import layers.hyper_layers as hl

# Alters default config with experimental modifications
def merge_configs(base, override):
    cfg = base.copy()

    for k, v in override.items():
        if k == "stage_configs":
            base_stages = cfg.get("stage_configs", {}).copy()
            new_stages = {}

            for i, stage_cfg in base_stages.items():
                new_stages[i] = stage_cfg.copy()

            for i, stage_override in v.items():
                if i in new_stages:
                    new_stages[i].update(stage_override)
                else:
                    new_stages[i] = stage_override.copy()

            cfg["stage_configs"] = new_stages

        elif isinstance(v, dict) and k in cfg:
            cfg[k] = {**cfg[k], **v}

        else:
            cfg[k] = v

    return cfg

DEFAULT_CONFIG = dict(
    in_c=16,
    z_dim=128,
    dropout=0.2,
    norm_layer=nn.BatchNorm2d,
    stage_configs={
        0: dict(conv_layer=nn.Conv2d),
        1: dict(conv_layer=hl.ResidualHyperConv2d, modulate=True, decompose=True),
        2: dict(conv_layer=hl.ResidualHyperConv2d, modulate=True, decompose=True),
    }
)


FULL_HYPER = dict(
    stage_configs={
        0: dict(conv_layer=hl.HyperConv2d),
        1: dict(conv_layer=hl.HyperConv2d),
        2: dict(conv_layer=hl.HyperConv2d),
    }
)

NO_MODULATION = dict(
    stage_configs={
        1: dict(modulate=False),
        2: dict(modulate=False),
    }
)