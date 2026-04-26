import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.functional import hyper_conv2d

class HyperConv2d(nn.Module):
    """
    Hypernetwork-based 2D convolution.

    Generates convolution weights from a conditioning vector `z`
    and applies them dynamically to the input.

    Supports grouped conv and optional low-rank projection.
    """
    def __init__(
        self,
        in_c,
        out_c,
        k,
        z_dim,
        stride=1,
        groups=1,
        rank=None,
        compression=None,
        **kwargs  # allows future extension without breaking
    ):
        super().__init__()

        assert in_c % groups == 0, "in_c must be divisible by groups"
        assert out_c % groups == 0, "out_c must be divisible by groups"

        self.in_c = in_c
        self.out_c = out_c
        self.k = k
        self.groups = groups
        self.stride = stride

        self.in_per_group = in_c // groups
        self.out_per_group = out_c // groups

        layers = []

        # Rank based on compression
        if rank is None and compression is not None:
            rank = z_dim // compression

        # Low-rank projection (LoRA-style)
        if rank is not None:
            layers.append(nn.Linear(z_dim, rank, bias=False))
            z_dim = rank

        # Final projection to conv weights
        layers.append(
            nn.Linear(
                z_dim,
                groups * self.out_per_group * self.in_per_group * k * k,
                bias=False
            )
        )

        self.layers = nn.Sequential(*layers)

    # =====================================================
    def get_weight(self, z):
        B = z.size(0)

        # Generate weights from conditioning vector
        w = self.layers(z).view(
            B,
            self.groups,
            self.out_per_group,
            self.in_per_group,
            self.k,
            self.k
        )

        # Kaiming fan-in normalization
        w = w * ((2.0) / (self.k * self.k * self.in_per_group)) ** 0.5

        # Reshape to grouped conv format
        return w.view(
            B,
            self.out_c,
            self.in_per_group,
            self.k,
            self.k
        )

    # =====================================================
    def forward(self, x, z):
        # Get conditioned convolution weights
        w = self.get_weight(z)

        # Apply convolution
        out = hyper_conv2d(
            x,
            w,
            stride=self.stride,
            groups=self.groups
        )

        return out
    
class DecomposedHyperConv2d(nn.Module):
    """
    Decomposed Hyper Convolutional Layer
    Generates a full kernel via pointwise (channel mixing)
    and depthwise (spatial structure) components.

    """

    def __init__(
        self,
        in_c,
        out_c,
        k,
        z_dim,
        stride=1,
        groups=1,
        rank=None,
        compression=None,
        **kwargs,  # <-- important for compatibility
    ):
        super().__init__()

        self.in_c = in_c
        self.out_c = out_c
        self.k = k
        self.stride = stride
        self.groups = groups

        # Pointwise: channel mixing
        self.pw = HyperConv2d(
            in_c=in_c,
            out_c=out_c,
            k=1, # pointwise
            z_dim=z_dim,
            groups=groups,
            rank=rank,
            compression=compression,
        )

        # Depthwise: spatial structure
        self.dw = HyperConv2d(
            in_c=out_c,
            out_c=out_c,
            k=k,
            z_dim=z_dim,
            stride=stride,
            groups=out_c,  # depthwise
            rank=rank,
            compression=compression,
        )

    # =====================================================
    def get_weight(self, z):
        """
        Combines pointwise and depthwise into a single kernel
        to be used by other modules

        Equivalent to:
            W[o, i, :, :] = pw[o, i] * dw[o, :, :]
        """
        

        # (B, out_c, in_c, 1, 1)
        pw = self.pw.get_weight(z)

        # (B, out_c, 1, k, k)
        dw = self.dw.get_weight(z)

        # Merge into full kernel
        # (B, out_c, in_c, k, k)
        w = pw * dw

        return w

    # =====================================================
    def forward(self, x, z, iter=None):
        # Sequential application (used in normal forward)
        out = self.pw(x, z, iter=iter)
        out = self.dw(out, z, iter=iter)
        return out
    
class HyperModulatedConv2d(nn.Module):
    """
    Convolution with channel-wise modulation from a conditioning vector z.

    Two modes:
    - forward(): fast path (conv + output modulation)
    - get_weight(): returns sample-wise modulated weights for composition
    """

    def __init__(
        self,
        in_c,
        out_c,
        k,
        z_dim,
        stride=1,
        groups=1,
        rank=None,
        compression=None,
        modulate=True,
        **kwargs
    ):
        super().__init__()

        assert in_c % groups == 0
        assert out_c % groups == 0

        self.in_c = in_c
        self.out_c = out_c
        self.k = k
        self.groups = groups
        self.stride = stride
        self.modulate = modulate

        self.in_per_group = in_c // groups

        # Base convolution weight
        self.weight = nn.Parameter(
            torch.randn(out_c, self.in_per_group, k, k) *
            (2 / (self.in_per_group * k * k)) ** 0.5
        )

        # Modulation gate
        if self.modulate:
            layers = []

            if rank is None and compression is not None:
                rank = max(1, z_dim // compression)

            if rank is not None:
                layers.append(nn.Linear(z_dim, rank, bias=False))
                z_dim = rank

            layers.append(nn.Linear(z_dim, out_c, bias=True))

            self.gate = nn.Sequential(*layers)

    # =====================================================
    def get_gate(self, z):
        """
        Returns channel-wise gate:
            (B, out_c)
        """
        return torch.sigmoid(self.gate(z))

    # =====================================================
    def get_weight(self, z):
        """
        Returns modulated weights:
            (B, out_c, in_c/groups, k, k)
        """
        w = self.weight.unsqueeze(0)  # (1, out_c, in_c/groups, k, k)

        if self.modulate:
            g = self.get_gate(z).view(z.size(0), self.out_c, 1, 1, 1)
            w = w * g

        return w

    # =====================================================
    def forward(self, x, z, iter=None):
        """
        Fast path: standard conv + output modulation
        """
        out = F.conv2d(
            x,
            self.weight,
            stride=self.stride,
            padding=self.k // 2,
            groups=self.groups
        )

        if self.modulate:
            g = self.get_gate(z).view(x.size(0), self.out_c, 1, 1)
            out = out * g

        return out

class ResidualHyperConv2d(nn.Module):
    """
    Residual Hyper Convolution.

    Combines:
        - Base weight (optionally modulated): Wb
        - Conditional hyper weight: Wz

    Final kernel:
        W = (Wb + Wz) / sqrt(2)

    Supports:
        - Full hyperconv (Wz)
        - Decomposed hyperconv (Wz)
        - Modulated base conv (Wb)
    """

    def __init__(
        self,
        in_c,
        out_c,
        k,
        z_dim,
        stride=1,
        groups=1,
        rank=None,
        compression=None,
        modulate=True,
        decompose=True,
        **kwargs
    ):
        super().__init__()

        self.stride = stride
        self.groups = groups

        # Conditional path (Wz)
        if decompose:
            self.Wz = DecomposedHyperConv2d(
                in_c=in_c,
                out_c=out_c,
                k=k,
                z_dim=z_dim,
                stride=stride,
                groups=groups,
                rank=rank,
                compression=compression,
            )
        else:
            self.Wz = HyperConv2d(
                in_c=in_c,
                out_c=out_c,
                k=k,
                z_dim=z_dim,
                stride=stride,
                groups=groups,
                rank=rank,
                compression=compression,
            )

        # Base path (Wb)
        self.Wb = HyperModulatedConv2d(
            in_c=in_c,
            out_c=out_c,
            k=k,
            z_dim=z_dim,
            stride=stride,
            groups=groups,
            modulate=modulate,
            rank=rank,
            compression=compression,
        )

    # =====================================================
    def get_weight(self, z):
        """
        Returns combined kernel:
            (B, out_c, in_c/groups, k, k)
        """

        wb = self.Wb.get_weight(z)  # base (modulated optional)
        wz = self.Wz.get_weight(z)  # conditional

        # Residual combination (variance-preserving)
        w = (wz + wb) / (2 ** 0.5)

        return w

    # =====================================================
    def forward(self, x, z, iter=None):
        """
        Always uses merged weight path.
        """
        w = self.get_weight(z)

        out = hyper_conv2d(
            x,
            w,
            stride=self.stride,
            groups=self.groups
        )

        return out