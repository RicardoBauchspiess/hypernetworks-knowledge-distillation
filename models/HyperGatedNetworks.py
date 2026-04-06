import torch
import torch.nn as nn
import torch.nn.functional as F

from ResNets import ResidualBlock


# =========================
# Gated HyperConv2d
# =========================
class HyperGatedConv2d(nn.Module):
    def __init__(self, in_c, out_c, k, z_dim, stride=1, a_min=0.2):
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.k = k
        self.stride = stride
        self.a_min = a_min

        # base weight
        self.base_weight = nn.Parameter(
            torch.randn(out_c, in_c, k, k) * (2.0 / (in_c * k * k))**0.5
        )

        # hypernetwork (delta weights)
        self.delta_head = nn.Linear(z_dim, out_c * in_c * k * k)

        # learnable gate
        self.s = nn.Parameter(torch.tensor(2.0))

    def compute_a(self, epoch=None, max_epochs=None):
        # training mode: require schedule inputs
        if self.training:
            assert epoch is not None and max_epochs is not None, "Provide epoch and max_epochs during training"
            schedule = 1.0 - (epoch / max_epochs)
            schedule = self.a_min + (1 - self.a_min) * schedule
        else:
            # evaluation mode: assume training finished
            schedule = self.a_min

        # learned gate
        gate = torch.sigmoid(self.s)

        return schedule * gate

    def forward(self, x, z, epoch=None, max_epochs=None):
        """
        x: [B, in_c, H, W]
        z: [B, z_dim]
        """
        B, _, H, W = x.shape

        # epoch/max_epochs optional (treated as finished training if None)

        # generate delta weights
        delta_w = self.delta_head(z)
        delta_w = delta_w.view(B, self.out_c, self.in_c, self.k, self.k)

        # compute gate
        a = self.compute_a(epoch, max_epochs)

        # combine weights
        w = a * self.base_weight.unsqueeze(0) + (1 - a) * delta_w

        # grouped conv trick
        x = x.view(1, B * self.in_c, H, W)
        w = w.view(B * self.out_c, self.in_c, self.k, self.k)

        out = F.conv2d(x, w, stride=self.stride, padding=self.k // 2, groups=B)

        H_out, W_out = out.shape[-2:]
        out = out.view(B, self.out_c, H_out, W_out)

        return out


# =========================
# Gated HyperLinear
# =========================
class HyperGatedLinear(nn.Module):
    def __init__(self, in_features, out_features, z_dim, a_min=0.2, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        self.a_min = a_min

        # base weight
        self.base_weight = nn.Parameter(
            torch.randn(out_features, in_features) * (2.0 / in_features) ** 0.5
        )

        # hypernetwork for weight delta
        self.delta_weight_head = nn.Linear(z_dim, out_features * in_features)

        # bias components
        if self.use_bias:
            self.base_bias = nn.Parameter(torch.zeros(out_features))
            self.delta_bias_head = nn.Linear(z_dim, out_features)
        else:
            self.register_parameter('base_bias', None)
            self.delta_bias_head = None

        # shared gate
        self.s = nn.Parameter(torch.tensor(2.0))

    def compute_a(self, epoch=None, max_epochs=None):
        if self.training:
            assert epoch is not None and max_epochs is not None, "Provide epoch and max_epochs during training"
            schedule = 1.0 - (epoch / max_epochs)
            schedule = self.a_min + (1 - self.a_min) * schedule
        else:
            schedule = self.a_min

        gate = torch.sigmoid(self.s)
        return schedule * gate

    def forward(self, x, z, epoch=None, max_epochs=None):
        """
        x: [B, in_features]
        z: [B, z_dim]
        """
        B = x.size(0)

        # compute gate
        a = self.compute_a(epoch, max_epochs)

        # weights
        delta_w = self.delta_weight_head(z)
        delta_w = delta_w.view(B, self.out_features, self.in_features)

        w = a * self.base_weight.unsqueeze(0) + (1 - a) * delta_w

        # batched matmul
        x = x.unsqueeze(2)
        out = torch.bmm(w, x).squeeze(2)

        # bias
        if self.use_bias:
            delta_b = self.delta_bias_head(z)
            b = a * self.base_bias + (1 - a) * delta_b
            out = out + b

        return out
    
# =========================
# Basic Residual Block (HyperGatedConv)
# =========================
class HyperGatedResidualBlock(nn.Module):
    def __init__(self, in_c, out_c, z_dim, stride=1, num_groups=8):
        super().__init__()

        self.conv1 = HyperGatedConv2d(in_c, out_c, 3, z_dim, stride=stride)
        self.norm1 = nn.GroupNorm(num_groups, out_c)

        self.conv2 = HyperGatedConv2d(out_c, out_c, 3, z_dim, stride=1)
        self.norm2 = nn.GroupNorm(num_groups, out_c)

        self.has_shortcut = stride != 1 or in_c != out_c
        if self.has_shortcut:
            self.shortcut = HyperGatedConv2d(in_c, out_c, 1, z_dim, stride=stride)

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