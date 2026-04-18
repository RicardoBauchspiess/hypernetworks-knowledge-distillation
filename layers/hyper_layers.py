import torch
import torch.nn as nn
import torch.nn.functional as F

import warnings

from layers.functional import hyper_conv2d
# =========================================================
# OPERATORS
# =========================================================

class FullOp(nn.Module):
    def __init__(self, in_c, out_c, k, z_dim, groups=1):
        super().__init__()

        assert in_c % groups == 0
        assert out_c % groups == 0

        self.in_c = in_c
        self.out_c = out_c
        self.k = k
        self.groups = groups

        self.in_per_group = in_c // groups
        self.out_per_group = out_c // groups

        self.linear = nn.Linear(
            z_dim,
            groups * self.out_per_group * self.in_per_group * k * k,
            bias=False
        )

    # =====================================================
    def get_weight(self, z):
        B = z.size(0)

        W = self.linear(z).view(
            B,
            self.groups,
            self.out_per_group,
            self.in_per_group,
            self.k,
            self.k
        )

        # reshape to grouped conv format
        return W.view(
            B,
            self.out_c,
            self.in_per_group,
            self.k,
            self.k
        )

    # =====================================================
    def forward(self, x, z):
        from layers.functional import hyper_conv2d
        W = self.get_weight(z)
        return hyper_conv2d(x, W, groups=self.groups)


class PWOp(nn.Module):
    def __init__(self, in_c, out_c, z_dim, groups=1):
        super().__init__()

        assert in_c % groups == 0
        assert out_c % groups == 0

        self.in_c = in_c
        self.out_c = out_c
        self.groups = groups

        self.in_per_group = in_c // groups
        self.out_per_group = out_c // groups

        # one linear that outputs all groups
        self.linear = nn.Linear(
            z_dim,
            groups * self.out_per_group * self.in_per_group,
            bias=False
        )

    # =====================================================
    def forward(self, x, z):
        B, _, H, W = x.shape

        pw = self.linear(z).view(
            B,
            self.groups,
            self.out_per_group,
            self.in_per_group
        )

        # reshape x into groups
        x = x.view(B, self.groups, self.in_per_group, H, W)

        # einsum per group
        out = torch.einsum(
            "bgchw, bgoc -> bgo hw",
            x,
            pw
        )

        return out.reshape(B, self.out_c, H, W)

    # =====================================================
    def get_weight(self, z):
        B = z.size(0)

        pw = self.linear(z).view(
            B,
            self.groups,
            self.out_per_group,
            self.in_per_group
        )

        w = torch.zeros(
            B,
            self.groups,
            self.out_per_group,
            self.groups,
            self.in_per_group,
            1,
            1,
            device=z.device,
            dtype=z.dtype
        )

        # fill diagonal blocks
        idx = torch.arange(self.groups)
        w[:, idx, :, idx, :, 0, 0] = pw

        return w.view(B, self.out_c, self.in_c, 1, 1)


class DWOp(nn.Module):
    def __init__(self, out_c, k, z_dim):
        super().__init__()
        self.out_c = out_c
        self.k = k
        self.linear = nn.Linear(z_dim, out_c * k * k, bias=False)

    def forward(self, x, z, stride=1, padding=None):
        B, C, H, W = x.shape

        if padding is None:
            padding = self.k // 2

        W_dw = self.linear(z).view(B, self.out_c, 1, self.k, self.k)

        out =  hyper_conv2d(
                    x,
                    W_dw,
                    stride=stride,
                    padding=padding,
                    groups=self.out_c,  
                )
        return out

    def get_weight(self, z):
        B = z.size(0)
        return self.linear(z).view(B, self.out_c, 1, self.k, self.k)


class BaseOp(nn.Module):
    def __init__(self, in_c, out_c, k, z_dim, use_gate, groups=1):
        super().__init__()

        assert in_c % groups == 0
        assert out_c % groups == 0

        self.in_c = in_c
        self.out_c = out_c
        self.k = k
        self.groups = groups
        self.use_gate = use_gate

        self.in_per_group = in_c // groups

        self.weight = nn.Parameter(
            torch.randn(out_c, self.in_per_group, k, k) *
            (2 / (self.in_per_group * k * k))**0.5
        )

        self.gate = nn.Linear(z_dim, out_c, bias=False) if use_gate else None

    def get_weight(self, z):
        B = z.size(0)

        W = self.weight.unsqueeze(0)

        if self.use_gate:
            gate = self.gate(z).view(B, self.out_c, 1, 1, 1)
            W = W * gate
        else:
            W = W.expand(B, -1, -1, -1, -1)

        return W


# =========================================================
# MAIN MODULE
# =========================================================

class HyperConv2d(nn.Module):
    def __init__(
        self,
        in_c,
        out_c,
        k,
        z_dim,
        groups = 1,
        stride=1, 
        padding=None,
        use_base=True,
        use_gate=True,
        use_pw=True,
        use_dw=True,
        use_full = False,
    ):
        super().__init__()

        self.use_base = use_base
        self.use_pw = use_pw
        self.use_dw = use_dw and k > 1
        self.use_full = use_full
        self.stride = stride
        self.padding = padding if padding is not None else k // 2

        

        # ---- depthwise constraint ----
        is_depthwise = (groups == in_c)

        if is_depthwise:
            if use_pw or use_dw:
                if not use_full:
                    warnings.warn(
                        "Depthwise mode (groups == in_c) does not support pw/dw. "
                        "Switching to use_full=True."
                    )

                # force full
                use_full = True
                self.use_pw = False
                self.use_dw = False

        # operators
        if use_full:
            # regular conv hyper weights
            self.full = FullOp(in_c, out_c, k, z_dim, groups)
            self.pw = None
            self.dw = None
        else:
            # hyper conv decomposition
            self.full = None
            self.pw = PWOp(in_c, out_c, z_dim, groups) if use_pw else None
            self.dw = DWOp(out_c, k, z_dim) if self.use_dw else None

        # regular conv static (learnable) weights
        self.base = BaseOp(
        in_c, out_c, k, z_dim, use_gate, groups) if use_base else None


    # =====================================================
    # SEQUENTIAL
    # =====================================================
    def _forward_sequential(self, x, z):
        out = None

        # ---- base ----
        if self.base is not None:
            W_base = self.base.get_weight(z)
            out = hyper_conv2d(
                x,
                W_base,
                stride=self.stride,
                padding=self.padding,
                groups=self.groups,
            )

        # ---- delta ----
        if self.full is not None:
            x_delta = self.full(x, z)
            out = x_delta if out is None else out + x_delta

        elif self.pw is not None:
            x_delta = self.pw(x, z)

            if self.dw is not None:
                x_delta = self.dw(
                    x_delta,
                    z,
                    stride=self.stride,
                    padding=self.padding,
                )

            out = x_delta if out is None else out + x_delta

        elif self.dw is not None:
            if out is None:
                raise ValueError("dw requires base or pw")

            x_delta = self.dw(
                out,
                z,
                stride=self.stride,
                padding=self.padding,
            )

            # matches merged: base + base*dw
            out = out + x_delta

        return out

    # =====================================================
    # MERGED
    # =====================================================
    def _forward_merged(self, x, z):
        w = None

        # ---- base ----
        if self.base is not None:
            w = self.base.get_weight(z)

        # ---- delta ----
        if self.full is not None:
            delta = self.full.get_weight(z)
        elif self.pw is not None:
            pw_w = self.pw.get_weight(z)
            delta = pw_w
            if self.dw is not None:
                dw_w = self.dw.get_weight(z)
                delta = delta * dw_w

        elif self.dw is not None:
            if self.base is None:
                raise ValueError("dw requires base or pw")
            dw_w = self.dw.get_weight(z)
            base_w = w
            delta = base_w * dw_w
            w = None# base_w was already used (no need to add w to delta w)
        else:
            delta = None

        if delta is not None:
            w = delta if w is None else w + delta

        if w is None:
            raise ValueError("No active paths")

        out =  hyper_conv2d(
                x,
                w,
                stride=self.stride,
                padding=self.padding,
                groups=self.groups,
            )
        return out

    # =====================================================
    def forward(self, x, z, merge=True):
        return self._forward_merged(x, z) if merge else self._forward_sequential(x, z)