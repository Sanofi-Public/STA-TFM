import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.FPT import MPLStyleViewFuser

class MotionBERT_MV(nn.Module):
    """
    Multiview 2D→3D using MotionBERT(DSTformer) per view and MPL-like Fusion Pose Transformer.
    """
    def __init__(self,
                 backbone,
                 num_joints=17,
                 in_dim_2d=3,
                 feat_dim=256,
                 bottleneck=4,
                 dstformer_depth=5,
                 dstformer_heads=8,
                 dstformer_mlp=4.0,
                 maxlen=243,
                 att_fuse=True,
                 fusion_depth=12,
                 fusion_heads=16,
                 fusion_mlp=4.0,
                 max_views=4,
                 use_time_in_fusion=False):
        super().__init__()

        # One DSTformer shared across views (weights reused)
        self.backbone = backbone
        # Bottle Neck dim
        bottleneck_dim = feat_dim // bottleneck
        self.proj = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, bottleneck_dim),
            nn.GELU()
        )
        # MPL Fusion over views
        self.fusion = MPLStyleViewFuser(
            num_joints=num_joints,
            feat_dim=bottleneck_dim,                
            num_heads=fusion_heads,           
            depth=fusion_depth,               
            mlp_ratio=fusion_mlp,             
            drop=0.0,
            num_views=max_views,
            # linear_weighted_mean=False       
        )

        # 3D regressor
        self.head_3d = nn.Sequential(
            nn.LayerNorm(bottleneck_dim),
            nn.Linear(bottleneck_dim, bottleneck_dim),
            nn.GELU(),
            nn.Linear(bottleneck_dim, 3)
        )

    @torch.no_grad()
    def _encode_one_view(self, x_view):
        """
        x_view: [B, T, J, in_dim_2d]
        returns: [B, T, J, C]
        """
        return self.backbone(x_view)  # keep DSTformer’s forward contract

    def forward(self, x2d, view_mask=None):
        """
        x2d: [B, T, V, J, in_dim_2d]
        view_mask: [B, T, V] or [B, T, J, V]
        """
        B, T, V, J, D = x2d.shape
        feats = []

        # run DSTformer backbone per view
        for v in range(V):
            f_v = self._encode_one_view(x2d[:, :, v, :, :])   # [B,T,J,feat_dim]
            f_v = self.proj(f_v) # removing bottleneck
            feats.append(f_v)

        f = torch.stack(feats, dim=2)  # [B,T,V,J,C]
        f = f.permute(0, 1, 3, 2, 4).contiguous()  # [B,T,J,V,C]

        if view_mask is not None:
            if view_mask.dim() == 3:
                view_mask = view_mask.unsqueeze(2).expand(B, T, J, V)

        # derive per-view confidence (mean over joints)
        view_conf = None
        if x2d.shape[-1] >= 3:
            conf = x2d[..., 2]  # [B,T,V,J]
            if view_mask is not None:
                conf = conf * view_mask.transpose(2, 3).float()  # align dims
            view_conf = conf.mean(dim=-1)  # [B,T,V]

        fused = self.fusion(f, view_mask=view_mask, view_conf=view_conf)
        xyz = self.head_3d(fused)
        return xyz
