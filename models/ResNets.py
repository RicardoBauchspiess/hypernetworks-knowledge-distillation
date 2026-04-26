import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================
# Standard Basic Residual Block (CIFAR-style)
# =========================
class ResidualBlock(nn.Module):
    expansion = 1

    def __init__(self, in_c, out_c, stride=1, num_groups=8):
        super().__init__()

        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(num_groups, out_c)

        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(num_groups, out_c)

        self.has_shortcut = stride != 1 or in_c != out_c
        if self.has_shortcut:
            self.shortcut = nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        out = self.conv1(x)
        out = self.norm1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        shortcut = x
        if self.has_shortcut:
            shortcut = self.shortcut(x)

        out += shortcut
        out = F.relu(out)

        return out
    


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100):
        super().__init__()
        self.in_c = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(8, 16)

        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)

        self.fc = nn.Linear(64, num_classes)

        self.dropout_rate = 0.2

    def _make_layer(self, block, out_c, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        for s in strides:
            layers.append(block(self.in_c, out_c, stride=s))
            self.in_c = out_c * block.expansion

        return nn.Sequential(*layers)

    def get_internal(self,):
        return self.internal_output
    def forward(self, x, dropout = None):

        if dropout is None:
            dropout = self.dropout_rate

        out = self.conv1(x)
        out = self.norm1(out)
        out = F.relu(out)

        out = self.layer1(out)

        self.internal_output = out.clone().detach()
        out = self.layer2(out)
        out = self.layer3(out)

        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = F.dropout(out, p=dropout, training=self.training)
        out = self.fc(out)

        return out
    


# =========================
# ResNet20 factory
# =========================
def ResNet20(num_classes=100):
    return ResNet(ResidualBlock, [3, 3, 3], num_classes)