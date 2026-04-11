import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import models.ResNets as ResNet
from models.ResNets import ResNet20
from layers.functional import standardize_weight


# =========================
# Residual HyperConv2d (additive)
# =========================
class HyperConv2d(nn.Module):
    def __init__(self, in_c, out_c, k, z_dim, stride=1):
        super().__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.k = k
        self.stride = stride

        self.fan_in = (in_c * k * k)

        self.base_weight = nn.Parameter(
            torch.randn(out_c, in_c, k, k) * (2.0 / (self.fan_in)) ** 0.5
        )

        self.z_dim = z_dim
        self.delta_weight_head = nn.Linear(z_dim, out_c * in_c * k * k)

        self.gate_head = nn.Linear(z_dim, out_c)

        self.gate_head.bias.data.fill_(-5.0)

    def forward(self, x, z):
        B, _, H, W = x.shape

        # splits base and hyperconv, to avoid memory overhead

        # -------- base conv --------
        out_base = F.conv2d(
            x,
            self.base_weight,
            stride=self.stride,
            padding=self.k // 2
        )  # [B, out_c, H, W]

        # -------- hyper conv --------
        delta_w = self.delta_weight_head(z)
        delta_w = delta_w.view(B, self.out_c, self.in_c, self.k, self.k)

        # scaling
        delta_w =  standardize_weight(delta_w, dims = (2,3,4)) # weight standardization (optinal)
        fan_in = self.in_c * self.k * self.k
        delta_w = delta_w * (2.0 / fan_in) ** 0.5 # kaiming fan_in scaling

        # grouped conv trick (same as before)
        x_grouped = x.view(1, B * self.in_c, H, W)
        delta_w = delta_w.view(B * self.out_c, self.in_c, self.k, self.k)

        out_delta = F.conv2d(
            x_grouped,
            delta_w,
            stride=self.stride,
            padding=self.k // 2,
            groups=B
        )

        H_out, W_out = out_delta.shape[-2:]
        out_delta = out_delta.view(B, self.out_c, H_out, W_out)

        # -------- gating --------
        g = torch.sigmoid(self.gate_head(z))  # [B, out_c]
        g = g.view(B, self.out_c, 1, 1)

        out = (1 - g) * out_base + g * out_delta

        return out
    

# =========================
# Residual HyperLinear (additive)
# =========================
class HyperLinear(nn.Module):
    def __init__(self, in_features, out_features, z_dim, bias=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias

        self.base_weight = nn.Parameter(
            torch.randn(out_features, in_features) * (2.0 / in_features) ** 0.5
        )

        self.fan_in = in_features

        self.delta_weight_head = nn.Linear(z_dim, out_features * in_features)

        
        self.gate_head = nn.Linear(z_dim, out_features)

        self.gate_head.bias.data.fill_(-5.0)
        
        if self.use_bias:
            self.base_bias = nn.Parameter(torch.zeros(out_features))
            self.delta_bias_head = nn.Linear(z_dim, out_features)
        else:
            self.register_parameter('base_bias', None)
            self.delta_bias_head = None

    def forward(self, x, z):
        B = x.size(0)

        # -------- base path --------
        out_base = F.linear(x, self.base_weight)  # [B, out]

        # -------- hyper path --------
        delta_w = self.delta_weight_head(z)
        delta_w = delta_w.view(B, self.out_features, self.in_features)

        # scaling
        delta_w = standardize_weight(delta_w, dims = 2) # weight standardization (optional)
        delta_w = delta_w * (2.0 / self.in_features) ** 0.5 # keiming fan_in scaling

        # batched matmul
        x_exp = x.unsqueeze(2)  # [B, in, 1]
        out_delta = torch.bmm(delta_w, x_exp).squeeze(2)  # [B, out]

        # -------- gating --------
        g = torch.sigmoid(self.gate_head(z))  # [B, out]

        out = (1 - g) * out_base + g * out_delta

        if self.use_bias:
            delta_b = self.delta_bias_head(z)
            out = out + self.base_bias + delta_b

        return out
    
# =========================
# Basic Residual Block (ResidualHyperConv)
# =========================
class ResBlock(nn.Module):
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
    

class HyperTrunk(nn.Module):
    def __init__(self, z_dim, hidden_dim=128, depth=2):
        super().__init__()

        layers = []
        in_dim = z_dim

        self.dropout = nn.Dropout(p=0.5)

        for _ in range(depth):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim

        self.net = nn.Sequential(*layers)

    def forward(self, z):
        h = self.net(z)
        h = self.dropout(h)
        return h    


# =========================
# Hybrid ResNet, combining standard conv and residual hyper conv
# =========================
class GatedHyperResNet(nn.Module):
    def __init__(self, num_blocks, num_classes=100, predictor_path = None):
        super().__init__()

        
        self.predictor = ResNet20(num_classes)

        if predictor_path is not None:
            state_dict = torch.load(predictor_path, weights_only=True)
            self.predictor.load_state_dict(state_dict)

        z_dim = 128
        self.hyper_trunk = HyperTrunk(num_classes, hidden_dim = z_dim)
        
        
        self.in_c = 16

        # static stem
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(8, 16)

        # static early layers
        self.layer1 = self._make_static_layer(16, num_blocks[0], stride=1)

        # hyper layers
        self.layer2 = self._make_hyper_layer(32, num_blocks[1], z_dim, stride=2)
        self.layer3 = self._make_hyper_layer(64, num_blocks[2], z_dim, stride=2)

        # hyper classifier
        self.fc = HyperLinear(64, num_classes, z_dim)

        self.zero = False

    def _make_static_layer(self, out_c, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(ResNet.ResidualBlock(self.in_c, out_c, s))
            self.in_c = out_c
        return nn.Sequential(*layers)

    def _make_hyper_layer(self, out_c, num_blocks, z_dim, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(ResBlock(self.in_c, out_c, z_dim, s))
            self.in_c = out_c
        return nn.ModuleList(layers)
    
    def predict_hyper(self, x):

        self.predictor.eval()
        with torch.no_grad():
            z = self.predictor(x)

            out = self.predictor.get_internal()
        
        h = self.hyper_trunk(z)

        return h, out
    
    def set_to_zero(self, zero = False):
        self.zero = zero

    def forward(self, x):
        
        z, out = self.predict_hyper(x)
        
        if self.zero:
            x = torch.zeros_like(x)
        
        out = self.conv1(x)
        out = self.norm1(out)
        out = F.relu(out)

        out = self.layer1(out)

        for block in self.layer2:
            out = block(out, z)

        for block in self.layer3:
            out = block(out, z)

        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)       # [B, C]
        out = self.fc(out, z)  

        return out
    


# =========================
# ResNet20 factory
# =========================
def GatedHyperResNet20(num_classes=100, predictor_path = None):
    return GatedHyperResNet( [3, 3, 3], num_classes, predictor_path)