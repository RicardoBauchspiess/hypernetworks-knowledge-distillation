import torch

def standardize_weight(w, dims, eps=1e-5):
    mean = w.mean(dim=dims, keepdim=True)
    std = w.std(dim=dims, keepdim=True) + eps
    return (w - mean) / std

# quick versions
def standardize_conv(w):
    # w: [B, out_c, in_c, k, k]
    mean = w.mean(dim=(2,3,4), keepdim=True)
    std = w.std(dim=(2,3,4), keepdim=True) + 1e-5
    return (w - mean) / std

def standardize_linear(w):
    # w: [B, out_features, in_features]
    mean = w.mean(dim=2, keepdim=True)
    std = w.std(dim=2, keepdim=True) + 1e-5
    return (w - mean) / std