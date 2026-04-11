import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import models.ResNets as ResNet
from models.ResNets import ResNet20
from layers.functional import standardize_weight

hyper_weight = 1.0

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
        self.delta_weight_head = nn.Linear(z_dim, out_c * in_c * k * k, bias = False)

    def forward(self, x, z):
        B, _, H, W = x.shape


        delta_w = self.delta_weight_head(z)
        delta_w = delta_w.view(B, self.out_c, self.in_c, self.k, self.k)
        delta_w =  standardize_weight(delta_w, dims = (2,3,4)) # weight standardization (optinal)
        delta_w = delta_w * (2.0 / (self.fan_in)) ** 0.5 # apply keiming normalization

        
        # apply noise only during training
        '''
        if self.training:
            mask = torch.bernoulli(torch.full_like(delta_w, 0.8))
            delta_w = delta_w * mask
            #delta_w = delta_w * (1 + 0.05 * torch.randn_like(delta_w))
        '''
        
        
        w = self.base_weight.unsqueeze(0)
        # add static and generated weights
        w = (w + hyper_weight * delta_w)
        w = w  / math.sqrt(2) # keep initial variance at 1, or include this in the keiming normalization


        x = x.view(1, B * self.in_c, H, W)
        w = w.view(B * self.out_c, self.in_c, self.k, self.k)
    

        out = F.conv2d(x, w, stride=self.stride, padding=self.k // 2, groups=B)


        H_out, W_out = out.shape[-2:]
        out = out.view(B, self.out_c, H_out, W_out)

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

        self.delta_weight_head = nn.Linear(z_dim, out_features * in_features, bias = False)

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
        delta_w = standardize_weight(delta_w, dims = 2) # weight standardization (optional)
        delta_w = delta_w * (2.0 / (self.fan_in)) ** 0.5 # kaiming normalization

        # apply noise only during training
        '''
        if self.training:
            mask = torch.bernoulli(torch.full_like(delta_w, 0.8))
            delta_w = delta_w * mask
            #delta_w = delta_w * (1 + 0.05 * torch.randn_like(delta_w))
        '''

        w = self.base_weight.unsqueeze(0)
        # add static and generated weights
        w = (w + hyper_weight * delta_w)
        w = w  / math.sqrt(2) # keep initial variance at 1, or include this in the keiming normalization


        x = x.unsqueeze(2)
        out = torch.bmm(w, x).squeeze(2)

        if self.use_bias:
            delta_b = self.delta_bias_head(z)
            delta_b = delta_b * (2.0 / (self.fan_in)) ** 0.5 # kaiming normalization
            b = self.base_bias + delta_b
            b = b / math.sqrt(2)
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
        #h = self.dropout(h)
        return h      


# =========================
# Hybrid ResNet, combining standard conv and residual hyper conv
# =========================
class ResidualHyperResNet(nn.Module):
    def __init__(self, num_blocks, num_classes=100, z_dim = 128):
        super().__init__()


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

    def forward(self, x, z):
        
        z = self.hyper_trunk(z)
        
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
# Hybrid ResNet, combining standard conv and residual hyper conv
# =========================
class PredictorHyperNet(nn.Module):
    def __init__(self, num_blocks, num_classes=100, predictor_path = None, z_dim = 128):
        super().__init__()

        self.prior_dim = num_classes

        self.predictor = ResNet20(num_classes)

        if predictor_path is not None:
            state_dict = torch.load(predictor_path, weights_only=True)
            self.predictor.load_state_dict(state_dict)
            self.freeze_predictor = True # don't train the predictor
        else:
            self.freeze_predictor = False

        self.hypernet = ResidualHyperResNet(num_blocks, num_classes, z_dim = z_dim)

        self.zero = False
        self.no_prior = False

    def set_to_zero(self, zero = False):
        self.zero = zero
        if self.zero:
            self.no_prior = False

    def set_no_prior(self, no_prior = False):
        self.no_prior = no_prior
        if self.no_prior:
            self.zero = False     

    def forward(self, x):
        
        # test prior impact
        if self.no_prior:
            prior = torch.zeros(x.size(0), self.prior_dim, device=x.device)
            prior_hyper = prior.detach()
        elif self.freeze_predictor:
            with torch.no_grad():
                prior = self.predictor(x)
            # soften prediction if a pre-trained model is used, to avoid bypass
            T = 1.6
            prior_hyper = F.softmax(prior.detach() / T, dim=1)
            alpha = 0.03
            prior_hyper = (1 - alpha) * prior_hyper + alpha / self.prior_dim
        else:
            prior = self.predictor(x)
            prior_hyper = prior.detach()

        # test prior bypassing (no image x, uses only prior)
        if self.zero:
            x = torch.zeros_like(x)
        
        out = self.hypernet(x, prior_hyper)

        return out, prior
    


# =========================
# ResNet20 factory
# =========================
def ResidualHyperResNet20(num_classes=100, predictor_path = None):
    return PredictorHyperNet( [3, 3, 3], num_classes, predictor_path = predictor_path)