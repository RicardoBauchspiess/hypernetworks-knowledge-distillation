import torch
import torch.nn as nn
import torch.nn.functional as F

from HyperGatedNetworks import HyperGatedConv2d
from ResNets import ResidualBlock


# =========================
# Modulated HyperConv2d (additive)
# =========================
class HyperModulatedConv2d(nn.Module):
    def __init__(self, in_c, out_c, k, z_dim, stride=1):
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.k = k
        self.stride = stride

        self.base_weight = nn.Parameter(
            torch.randn(out_c, in_c, k, k) * (2.0 / (in_c * k * k)) ** 0.5
        )

        self.delta_weight_head = nn.Linear(z_dim, out_c * in_c * k * k)

    def forward(self, x, z):
        B, _, H, W = x.shape

        delta_w = self.delta_weight_head(z)
        delta_w = delta_w.view(B, self.out_c, self.in_c, self.k, self.k)

        w = self.base_weight.unsqueeze(0) + delta_w

        x = x.view(1, B * self.in_c, H, W)
        w = w.view(B * self.out_c, self.in_c, self.k, self.k)

        out = F.conv2d(x, w, stride=self.stride, padding=self.k // 2, groups=B)

        H_out, W_out = out.shape[-2:]
        out = out.view(B, self.out_c, H_out, W_out)

        return out
    

# =========================
# Modulated HyperLinear (additive)
# =========================
class HyperModulatedLinear(nn.Module):
    def __init__(self, in_features, out_features, z_dim, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias

        self.base_weight = nn.Parameter(
            torch.randn(out_features, in_features) * (2.0 / in_features) ** 0.5
        )

        self.delta_weight_head = nn.Linear(z_dim, out_features * in_features)

        if self.use_bias:
            self.base_bias = nn.Parameter(torch.zeros(out_features))
            self.delta_bias_head = nn.Linear(z_dim, out_features)
        else:
            self.register_parameter('base_bias', None)
            self.delta_bias_head = None

    def forward(self, x, z):
        B = x.size(0)

        delta_w = self.delta_weight_head(z)
        delta_w = delta_w.view(B, self.out_features, self.in_features)

        w = self.base_weight.unsqueeze(0) + delta_w

        x = x.unsqueeze(2)
        out = torch.bmm(w, x).squeeze(2)

        if self.use_bias:
            delta_b = self.delta_bias_head(z)
            out = out + self.base_bias + delta_b

        return out
    
# =========================
# Basic Residual Block (HyperGatedConv)
# =========================
class HyperModulatedResidualBlock(nn.Module):
    def __init__(self, in_c, out_c, z_dim, stride=1, num_groups=8):
        super().__init__()

        self.conv1 = HyperModulatedConv2d(in_c, out_c, 3, z_dim, stride=stride)
        self.norm1 = nn.GroupNorm(num_groups, out_c)

        self.conv2 = HyperModulatedConv2d(out_c, out_c, 3, z_dim, stride=1)
        self.norm2 = nn.GroupNorm(num_groups, out_c)

        self.has_shortcut = stride != 1 or in_c != out_c
        if self.has_shortcut:
            self.shortcut = HyperModulatedConv2d(in_c, out_c, 1, z_dim, stride=stride)

    def forward(self, x, z):
        out = self.conv1(x, z)
        out = self.norm1(out)
        out = F.relu(out)

        out = self.conv2(out, z)
        out = self.norm2(out)

        shortcut = x
        if self.has_shortcut:
            shortcut = self.shortcut(x, z)

        out += shortcut
        out = F.relu(out)

        return out
    

# =========================
# Hybrid ResNet, combining standard conv and hypermodulated conv
# =========================
class HyperModulatedResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100):
        super().__init__()
        self.in_c = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(8, 16)

        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)

        self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, block, out_c, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        for s in strides:
            layers.append(block(self.in_c, out_c, stride=s))
            self.in_c = out_c * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.norm1(out)
        out = F.relu(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out
    


# =========================
# ResNet20 factory
# =========================
def ResNet20(num_classes=100):
    return ResNet(ResidualBlock, [3, 3, 3], num_classes)