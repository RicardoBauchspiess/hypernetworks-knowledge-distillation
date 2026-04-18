import torch
import torch.nn as nn
import torch.nn.functional as F



import torch
import torch.nn as nn
import torch.nn.functional as F

class HyperMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

    def proj(self, x, w):
        """
        x: (B, N, D)
        w: (D, D) or (B, D, D)
        """
        if w.dim() == 2:
            # shared weight
            return x @ w
        elif w.dim() == 3:
            # per-sample weight
            return torch.bmm(x, w)
        else:
            raise ValueError(f"Invalid weight shape: {w.shape}")

    def split_heads(self, x):
        B, N, D = x.shape
        return x.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        # (B, H, N, head_dim)

    def merge_heads(self, x):
        B, H, N, Hd = x.shape
        return x.transpose(1, 2).contiguous().view(B, N, H * Hd)

    def forward(self, x, w_q, w_k, w_v, w_o):
        """
        x:   (B, N, D)
        w_*: (D, D) or (B, D, D)
        """

        B, N, D = x.shape

        # ---- projections (auto-handle shapes) ----
        q = self.proj(x, w_q)
        k = self.proj(x, w_k)
        v = self.proj(x, w_v)

        # ---- split heads ----
        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)

        # ---- attention ----
        attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = torch.softmax(attn, dim=-1)

        out = attn @ v

        # ---- merge heads ----
        out = self.merge_heads(out)

        # ---- output projection ----
        out = self.proj(out, w_o)

        return out
    
class HyperLinearWeight(nn.Module):
    def __init__(self, in_dim, out_dim, z_dim, rank=32, scale=0.5):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.rank = rank
        self.scale = scale

        # base weight
        self.W = nn.Parameter(torch.randn(in_dim, out_dim) * 0.02)

        # low-rank hypernet
        self.A = nn.Linear(z_dim, in_dim * rank)
        self.B = nn.Linear(z_dim, rank * out_dim)

    def forward(self, z):
        """
        z: (B, z_dim)

        returns:
        W: (D, D) or (B, D, D)
        """

        B = z.size(0)

        A = self.A(z).view(B, self.in_dim, self.rank)
        Bm = self.B(z).view(B, self.rank, self.out_dim)

        delta_w = torch.bmm(A, Bm)   # (B, D, D)
        delta_w = torch.tanh(delta_w)

        W = self.W.unsqueeze(0) + self.scale * delta_w

        return W

class HyperAttentionWrapper(nn.Module):
    def __init__(self, embed_dim, num_heads,
                 q_gen=None, k_gen=None, v_gen=None, o_gen=None):
        super().__init__()

        self.embed_dim = embed_dim

        # attention core (your flexible one)
        self.attn = HyperMultiheadAttention(embed_dim, num_heads)

        # ---- generators (can be None) ----
        self.q_gen = q_gen
        self.k_gen = k_gen
        self.v_gen = v_gen
        self.o_gen = o_gen

        # ---- fallback static weights ----
        self.W_q = nn.Parameter(torch.randn(embed_dim, embed_dim) * 0.02)
        self.W_k = nn.Parameter(torch.randn(embed_dim, embed_dim) * 0.02)
        self.W_v = nn.Parameter(torch.randn(embed_dim, embed_dim) * 0.02)
        self.W_o = nn.Parameter(torch.randn(embed_dim, embed_dim) * 0.02)

    def get_weight(self, gen, W, z):
        """
        gen: generator module or None
        W:   base weight (D, D)
        z:   (B, z_dim)
        """
        if gen is not None:
            return gen(z)         # (B, D, D)
        else:
            return W              # (D, D)

    def forward(self, x, z=None):
        """
        x: (B, N, D)
        z: (B, z_dim) or None
        """

        # ---- select weights ----
        W_q = self.get_weight(self.q_gen, self.W_q, z)
        W_k = self.get_weight(self.k_gen, self.W_k, z)
        W_v = self.get_weight(self.v_gen, self.W_v, z)
        W_o = self.get_weight(self.o_gen, self.W_o, z)

        # ---- attention ----
        out = self.attn(x, W_q, W_k, W_v, W_o)

        return out

# -------------------------
# Patch Embedding
# -------------------------
class PatchEmbed(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_ch=3, dim=192):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_ch, dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # (B, dim, H/ps, W/ps)
        x = x.flatten(2).transpose(1, 2)  # (B, N, dim)
        return x


# -------------------------
# Transformer Block
# -------------------------
class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(self, x):
        # Attention
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h)
        x = x + attn_out

        # MLP
        h = self.norm2(x)
        x = x + self.mlp(h)

        return x


# -------------------------
# Small ViT
# -------------------------
class SmallViT(nn.Module):
    def __init__(self, num_classes=100, dim=192, depth=6, heads=3):
        super().__init__()

        self.patch_embed = PatchEmbed(patch_size=4, dim=dim)
        num_patches = (32 // 4) ** 2

        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, dim))

        self.blocks = nn.Sequential(*[
            Block(dim, heads) for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        B = x.size(0)

        x = self.patch_embed(x)

        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed

        x = self.blocks(x)
        x = self.norm(x)

        cls = x[:, 0]
        return self.head(cls)