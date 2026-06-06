import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from mmengine.model.weight_init import constant_init, kaiming_init


class MotionGuidedModulation_HeatmapV10(BaseModule):
    """Motion-Guided Spatial Refinement Module (MG-SRM).

    Uses direction-aware multi-order temporal differences to suppress
    static background noise in pose heatmaps via pyramid motion saliency.
    """

    def __init__(self,
                 in_channels,
                 num_heads=4,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        # Channel Alignment
        self.in_channels = in_channels
        self.pad_channels = 0
        if in_channels % 4 != 0:
            self.pad_channels = 4 - (in_channels % 4)
        self.working_channels = in_channels + self.pad_channels

        self.c_motion = self.working_channels // 2
        self.c_branch = self.working_channels // 4

        # 1. Mapping
        self.mapping = nn.Conv3d(self.working_channels, self.working_channels, 1)

        # 2. Direction-Aware Motion Reduce
        # Input: 4 * c_motion (d1_pos, d1_neg, d2_pos, d2_neg)
        self.motion_reduce = nn.Conv3d(self.c_motion * 4, self.c_motion, 1)

        # 3. Pyramid Motion Encoder
        # Path A: Fine Scale
        self.motion_net_fine = nn.Sequential(
            nn.Conv3d(self.c_motion, self.c_motion // 2,
                      kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=False),
            nn.Conv3d(self.c_motion // 2, self.c_motion,
                      kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(self.c_motion),
            nn.GELU()
        )

        # Path B: Coarse Scale (Decomposed Large Kernel)
        self.motion_net_coarse = nn.Sequential(
            nn.Conv3d(self.c_motion, self.c_motion // 2,
                      kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=False),
            nn.Conv3d(self.c_motion // 2, self.c_motion // 2,
                      kernel_size=(1, 1, 5), padding=(0, 0, 2),
                      groups=self.c_motion // 2, bias=False),
            nn.Conv3d(self.c_motion // 2, self.c_motion,
                      kernel_size=(1, 5, 1), padding=(0, 2, 0), bias=False),
            nn.BatchNorm3d(self.c_motion),
            nn.GELU()
        )

        # 4. Saliency Generator (single spatial mask)
        self.saliency_fusion = nn.Sequential(
            nn.Conv3d(self.c_motion * 2, 1, 1),
            nn.Sigmoid()
        )

        # 5. Feature Branches
        self.spatial_conv = nn.Sequential(
            nn.Conv3d(self.c_branch, self.c_branch,
                      kernel_size=(1, 1, 5), padding=(0, 0, 2),
                      groups=self.c_branch, bias=False),
            nn.Conv3d(self.c_branch, self.c_branch,
                      kernel_size=(1, 5, 1), padding=(0, 2, 0),
                      groups=1, bias=False),
            nn.BatchNorm3d(self.c_branch),
            nn.GELU()
        )

        self.temporal_k3 = nn.Sequential(
            nn.Conv3d(self.c_branch, self.c_branch // 2,
                      kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=False),
            nn.BatchNorm3d(self.c_branch // 2),
            nn.GELU()
        )
        self.temporal_k5 = nn.Sequential(
            nn.Conv3d(self.c_branch, self.c_branch // 2,
                      kernel_size=(5, 1, 1), padding=(2, 0, 0), bias=False),
            nn.BatchNorm3d(self.c_branch // 2),
            nn.GELU()
        )

        # 6. Modulation Heads
        self.MSM = nn.Sequential(
            nn.Conv3d(self.c_motion, self.c_branch * 2, 1), nn.Tanh())
        self.MTM = nn.Sequential(
            nn.Conv3d(self.c_motion, self.c_branch * 2, 1), nn.Tanh())

        # 7. Aggregation
        self.gate_conv = nn.Conv3d(self.working_channels, 3, 1)
        self.proj = nn.Conv3d(self.working_channels, self.in_channels, 1)

        self.alpha = nn.Parameter(torch.zeros(1, self.in_channels, 1, 1, 1))

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                kaiming_init(m)
            elif isinstance(m, nn.BatchNorm3d):
                constant_init(m, 1)
        constant_init(self.proj, 0)

    def get_direction_aware_motion(self, x):
        """Compute direction-aware multi-order temporal differences.

        Decomposes each diff into positive/negative components to preserve
        directional information. Returns 4 * c_motion channels:
        [d1_pos, d1_neg, d2_pos, d2_neg]
        """
        N, C, T, H, W = x.shape

        if T < 3:
            return torch.zeros(N, C * 4, T, H, W, device=x.device,
                               dtype=x.dtype)

        # Order 1: first-order temporal difference (velocity)
        d1 = x[:, :, 1:] - x[:, :, :-1]
        d1 = F.pad(d1, (0, 0, 0, 0, 0, 1))

        # Order 2: second-order temporal difference
        d2 = x[:, :, 2:] - x[:, :, :-2]
        d2 = F.pad(d2, (0, 0, 0, 0, 0, 2))

        d1_pos = F.relu(d1)
        d1_neg = F.relu(-d1)
        d2_pos = F.relu(d2)
        d2_neg = F.relu(-d2)

        return torch.cat([d1_pos, d1_neg, d2_pos, d2_neg], dim=1)

    def forward(self, x):
        if x.size(2) < 4:
            return x

        identity = x

        # Padding
        if self.pad_channels > 0:
            x_in = F.pad(x, (0, 0, 0, 0, 0, 0, 0, self.pad_channels))
        else:
            x_in = x

        # 1. Mapping & Split
        x_mapped = self.mapping(x_in)
        x_motion_in, x_st = torch.split(
            x_mapped, [self.c_motion, self.c_motion], dim=1)
        x_spat_in, x_temp_in = torch.chunk(x_st, 2, dim=1)

        # 2. Direction-Aware Motion Encoding
        raw_motion = self.get_direction_aware_motion(x_motion_in)
        motion_base = self.motion_reduce(raw_motion)

        # Pyramid: fine + coarse
        motion_fine = self.motion_net_fine(motion_base)
        motion_coarse = self.motion_net_coarse(motion_base)
        motion_feat = motion_fine + motion_coarse

        # 3. Spatial Saliency
        mask_input = torch.cat([motion_fine, motion_coarse], dim=1)
        saliency_mask = self.saliency_fusion(mask_input)

        x_spat_clean = x_spat_in * (1 + saliency_mask)
        x_temp_clean = x_temp_in * (1 + saliency_mask)

        # 4. Branches
        x_gc = self.spatial_conv(x_spat_clean)
        x_tc = torch.cat([
            self.temporal_k3(x_temp_clean),
            self.temporal_k5(x_temp_clean)
        ], dim=1)

        # 5. Modulation
        z_s = self.MSM(motion_feat)
        gamma_s, beta_s = torch.chunk(z_s, 2, dim=1)
        x_gcm = x_gc * (1 + gamma_s) + beta_s

        z_t = self.MTM(motion_feat)
        gamma_t, beta_t = torch.chunk(z_t, 2, dim=1)
        x_tcm = x_tc * (1 + gamma_t) + beta_t

        # 6. Aggregation
        x_agg_in = torch.cat([x_tcm, x_gcm, motion_feat], dim=1)
        gates = torch.sigmoid(self.gate_conv(x_agg_in))

        out_feat = torch.cat([
            x_tc * gates[:, 0:1, ...],
            x_gc * gates[:, 1:2, ...],
            motion_feat * gates[:, 2:3, ...]
        ], dim=1)

        out = self.proj(out_feat)

        return identity + out * self.alpha
