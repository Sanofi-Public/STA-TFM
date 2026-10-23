#  Fusion Pose Transformer
import torch
import torch.nn as nn
import torch.nn.functional as F

# DropPath (stochastic depth) 
class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        mask.floor_()
        return x * (mask / keep_prob)


# Simple MLP block
class Mlp(nn.Module):
    def __init__(self, dim, hidden_ratio=4.0, drop=0.0, act_layer=nn.SiLU):
        super().__init__()
        hidden = int(dim * hidden_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        x = self.fc1(x); x = self.act(x); x = self.drop(x)
        x = self.fc2(x); x = self.drop(x)
        return x


#  Mask-aware Attention 
class Attention(nn.Module):
    def __init__(self, dim, nhead, drop=0.0):
        super().__init__()
        assert dim % nhead == 0
        self.nhead = nhead
        self.head_dim = dim // nhead
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.nhead, C // self.nhead).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, h, N, Hd]
        attn = (q @ k.transpose(-2, -1)) * self.scale

        if mask is not None:
            m = mask.unsqueeze(1).unsqueeze(2)
            attn = attn.masked_fill(~m.bool(), float("-inf"))

        #  stable softmax 
        attn = torch.nan_to_num(attn, nan=0.0, neginf=-1e4, posinf=1e4)
        max_attn = attn.max(dim=-1, keepdim=True).values
        attn = torch.exp(attn - max_attn)
        denom = attn.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        attn = attn / denom
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


#Transformer Block
class Block(nn.Module):
    def __init__(self, dim, nhead, mlp_ratio=4.0, drop=0.0, drop_path=0.0, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, nhead, drop=drop)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(dim, mlp_ratio, drop=drop, act_layer=nn.SiLU)
        self.gamma1 = nn.Parameter(torch.ones(dim) * 1e-2)
        self.gamma2 = nn.Parameter(torch.ones(dim) * 1e-2)
        self.drop_path1 = DropPath(drop_path)
        self.drop_path2 = DropPath(drop_path)

    def forward(self, x, mask=None):
        x = x + self.drop_path1(self.gamma1 * self.attn(self.norm1(x), mask=mask))
        x = x + self.drop_path2(self.gamma2 * self.mlp(self.norm2(x)))
        return x


# Fusion Pose Transformer 
class MPLStyleViewFuser(nn.Module):
    """
    Fuses per-view features with a Transformer over view tokens (MPL-style),
    includes DropPath, view-joint embeddings, stable attention, and confidence gating.
    """
    def __init__(self,
                 num_joints: int,
                 feat_dim: int,
                 num_heads: int = 8,
                 depth: int = 12,
                 mlp_ratio: float = 3.0,
                 drop: float = 0.0,
                 num_views: int = 4,
                 use_linear_pool: bool = False,
                 cam_geom_dim: int = 0):
        super().__init__()
        self.J = num_joints
        self.C = feat_dim
        self.V = num_views

        self.embed_dim = self.J * self.C
        assert self.embed_dim % num_heads == 0

        # DropPath schedule (smaller for stability)
        dpr = torch.linspace(0, 0.05, steps=depth).tolist()
        self.blocks = nn.ModuleList([
            Block(self.embed_dim, num_heads, mlp_ratio, drop=drop, drop_path=dpr[i])
            for i in range(depth)
        ])
        self.view_norm = nn.LayerNorm(self.embed_dim)

        # View + joint embeddings
        self.view_embed = nn.Parameter(torch.randn(self.V, self.embed_dim) * 0.02)
        self.view_joint_embed = nn.Parameter(torch.randn(self.V, self.J, self.C) * 0.02)

        # Optional camera geometry encoder
        self.cam_geom_dim = cam_geom_dim
        if cam_geom_dim > 0:
            self.cam_geom_proj = nn.Sequential(
                nn.Linear(cam_geom_dim, self.embed_dim),
                nn.SiLU(),
                nn.Linear(self.embed_dim, self.embed_dim)
            )

        # Confidence-gated weighting
        self.view_gate = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, self.embed_dim // 4),
            nn.SiLU(),
            nn.Linear(self.embed_dim // 4, 1),
            nn.Sigmoid()
        )

        # Pooling
        self.use_linear_pool = use_linear_pool
        if self.use_linear_pool:
            self.pool_linear = nn.Linear(self.V * self.embed_dim, self.embed_dim)
        else:
            self.pool_conv = nn.Conv1d(in_channels=self.V, out_channels=1, kernel_size=1)

    def forward(self, f, view_mask=None, cam_geom=None, view_conf=None):
        """
        f: [B, T, J, V, C]
        view_mask: [B, T, V] or [B, T, J, V]
        view_conf: [B, T, V] optional average 2D confidence
        """
        if f.dim() != 5:
            raise ValueError("Expected f shape [B, T, J, V, C]")
        B, T, J, V, C = f.shape
        assert J == self.J and C == self.C and V <= self.V, "Shape/config mismatch"

        # Fix joint embedding orientation 
        vj = self.view_joint_embed[:V].permute(1, 0, 2)  # [J,V,C]
        f = f + vj.unsqueeze(0).unsqueeze(0)             # [1,1,J,V,C]

        # Flatten joints into view tokens
        x = f.permute(0, 1, 3, 2, 4).reshape(B * T, V, J * C)  # [B*T, V, E]

        # Build mask
        vm = None
        if view_mask is not None:
            if view_mask.dim() == 4:
                view_mask = view_mask.any(dim=2)
            vm = view_mask.reshape(B * T, V).to(torch.bool)

        # Add embeddings
        x = x + self.view_embed[:V].unsqueeze(0)
        if self.cam_geom_dim > 0 and cam_geom is not None:
            cg = cam_geom.reshape(B * T, V, -1)
            x = x + self.cam_geom_proj(cg)

        # Transformer over views
        for blk in self.blocks:
            x = blk(x, mask=vm)

        x = self.view_norm(x)

        # Stable confidence gating
        gates = self.view_gate(x)  # [B*T,V,1]
        gates = gates.clamp(min=1e-3, max=1.0)
        if view_conf is not None:
            vc = view_conf.reshape(B * T, V, 1).clamp(0, 1)
            gates = gates * vc
        if vm is not None:
            gates = gates * vm.unsqueeze(-1).float()

        # Normalize gating weights
        denom = gates.sum(dim=1, keepdim=True).clamp_min(1e-6)
        gates = gates / denom

        x = x * gates

        # Residual stabilizer
        resid = x.mean(dim=1, keepdim=True)
        x = x + 0.1 * resid

        # Pool across views
        if self.use_linear_pool:
            x_flat = x.reshape(B * T, V * self.embed_dim)
            out = self.pool_linear(x_flat)
        else:
            out = self.pool_conv(x).squeeze(1)

        out = out.reshape(B, T, J, C)
        return out
