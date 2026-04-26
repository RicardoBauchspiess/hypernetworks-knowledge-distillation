import torch
import torch.nn.functional as F

def standardize_weight(w, dims, eps=1e-5):
    mean = w.mean(dim=dims, keepdim=True)
    std = w.std(dim=dims, keepdim=True) + eps
    return (w - mean) / std

def hyper_conv2d(x, W, stride=1, padding=None, groups=1):
    B, C_in, H, W_in = x.shape
    _, C_out, C_per_group, k, _ = W.shape

    if padding is None:
        padding = k // 2

    x = x.view(1, B * C_in, H, W_in)
    W = W.view(B * C_out, C_per_group, k, k)

    out = F.conv2d(
        x,
        W,
        stride=stride,
        padding=padding,
        groups=B * groups,
    )

    H_out, W_out = out.shape[-2:]
    return out.view(B, C_out, H_out, W_out)