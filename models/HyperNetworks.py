import torch
import torch.nn as nn
import torch.nn.functional as F

from ResNets import ResidualBlock


# =========================
# HyperConv2d (per-sample)
# =========================
class HyperConv2d(nn.Module):
    def __init__(self, in_c, out_c, k, z_dim, stride=1):
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.k = k
        self.stride = stride

        self.head = nn.Linear(z_dim, out_c * in_c * k * k)

    def forward(self, x, z):
        B, _, H, W = x.shape

        # generate weights
        w = self.head(z)
        w = w.view(B, self.out_c, self.in_c, self.k, self.k)

        # grouped conv trick
        x = x.view(1, B * self.in_c, H, W)
        w = w.view(B * self.out_c, self.in_c, self.k, self.k)

        out = F.conv2d(x, w, stride=self.stride, padding=self.k // 2, groups=B)

        # restore batch
        H_out, W_out = out.shape[-2:]
        out = out.view(B, self.out_c, H_out, W_out)

        return out
    

# =========================
# Basic Residual Block (HyperConv)
# =========================
class HyperResidualBlock(nn.Module):
    def __init__(self, in_c, out_c, z_dim, stride=1, num_groups=8):
        super().__init__()

        self.conv1 = HyperConv2d(in_c, out_c, 3, z_dim, stride=stride)
        self.norm1 = nn.GroupNorm(num_groups, out_c)

        self.conv2 = HyperConv2d(out_c, out_c, 3, z_dim, stride=1)
        self.norm2 = nn.GroupNorm(num_groups, out_c)

        self.has_shortcut = stride != 1 or in_c != out_c
        if self.has_shortcut:
            self.shortcut = HyperConv2d(in_c, out_c, 1, z_dim, stride=stride)

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
# HyperLinear (hypernetwork, optional bias)
# =========================
class HyperLinear(nn.Module):
    def __init__(self, in_features, out_features, z_dim, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias

        # weight generator
        self.weight_head = nn.Linear(z_dim, out_features * in_features)

        if self.use_bias:
            # bias generator
            self.bias_head = nn.Linear(z_dim, out_features)
        else:
            self.bias_head = None

    def forward(self, x, z):
        """
        x: [B, in_features]
        z: [B, z_dim]
        """
        B = x.size(0)

        # generate weights
        w = self.weight_head(z)
        w = w.view(B, self.out_features, self.in_features)

        # batched matmul
        x = x.unsqueeze(2)
        out = torch.bmm(w, x).squeeze(2)

        if self.use_bias:
            b = self.bias_head(z)
            out = out + b

        return out
    

# =========================
# Hybrid Hyper ResNet, conditioned on class probabilities
# =========================
class HyperResNet(nn.Module):
    def __init__(self, num_blocks, num_classes=100):
        super().__init__()

        self.in_c = 16

        # static stem
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(8, 16)

        # static early layers
        self.layer1 = self._make_static_layer(16, num_blocks[0], stride=1)

        # hyper layers
        self.layer2 = self._make_hyper_layer(32, num_blocks[1], num_classes, stride=2)
        self.layer3 = self._make_hyper_layer(64, num_blocks[2], num_classes, stride=2)

        # hyper classifier
        self.fc = HyperLinear(64, num_classes, num_classes)

    def _make_static_layer(self, out_c, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(ResidualBlock(self.in_c, out_c, s))
            self.in_c = out_c
        return nn.Sequential(*layers)

    def _make_hyper_layer(self, out_c, num_blocks, z_dim, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(HyperResidualBlock(self.in_c, out_c, z_dim, s))
            self.in_c = out_c
        return nn.ModuleList(layers)

    def forward(self, x, z):
        out = self.conv1(x)
        out = self.norm1(out)
        out = F.relu(out)

        out = self.layer1(out)

        for block in self.layer2:
            out = block(out, z)

        for block in self.layer3:
            out = block(out, z)
        return out
    

# =========================
# HyperResNet20 factory
# =========================
def HyperResNet20(num_classes=100):
    return HyperResNet([3, 3, 3], num_classes)