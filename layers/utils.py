import torch.nn as nn

def make_norm(num_channels, norm_layer, norm_groups=None):
    if norm_layer is None:
        return nn.Identity()

    if norm_layer is nn.BatchNorm2d:
        return norm_layer(num_channels)

    if norm_layer is nn.InstanceNorm2d:
        return norm_layer(num_channels)

    if norm_layer is nn.GroupNorm:
        if norm_groups is None:
            raise ValueError("norm_groups must be provided for GroupNorm")
        return norm_layer(norm_groups, num_channels)

    # fallback for custom norm layers
    return norm_layer(num_channels)