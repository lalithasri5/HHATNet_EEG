import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiBandCNN(nn.Module):
    def __init__(self, n_bands=5, out_ch=16):
        super().__init__()
        self.conv15 = nn.Conv2d(n_bands, out_ch, kernel_size=(1, 15), padding=(0, 7))
        self.conv31 = nn.Conv2d(n_bands, out_ch, kernel_size=(1, 31), padding=(0, 15))
        self.conv63 = nn.Conv2d(n_bands, out_ch, kernel_size=(1, 63), padding=(0, 31))
        self.bn = nn.BatchNorm2d(out_ch * 3)

    def forward(self, x):
        x = torch.cat([self.conv15(x), self.conv31(x), self.conv63(x)], dim=1)
        return F.elu(self.bn(x))


class SpatialCNN(nn.Module):
    def __init__(self, in_ch, n_channels):
        super().__init__()
        self.spatial = nn.Conv2d(in_ch, in_ch, kernel_size=(n_channels, 1), groups=in_ch)
        self.bn = nn.BatchNorm2d(in_ch)

    def forward(self, x):
        return F.elu(self.bn(self.spatial(x)))


class GroupedSEAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(channels, channels // 4, 1),
            nn.ReLU(),
            nn.Conv2d(channels // 4, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.fc(x)


def apply_rope(x):
    # x: batch, heads, time, dim
    b, h, t, d = x.shape
    half = d // 2

    freqs = torch.arange(half, device=x.device).float()
    freqs = 1.0 / (10000 ** (freqs / half))

    pos = torch.arange(t, device=x.device).float()
    angles = pos[:, None] * freqs[None, :]

    sin = angles.sin()[None, None, :, :]
    cos = angles.cos()[None, None, :, :]

    x1 = x[..., :half]
    x2 = x[..., half:half * 2]

    rope = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

    if d % 2 == 1:
        rope = torch.cat([rope, x[..., -1:]], dim=-1)

    return rope


class GQAWithRoPE(nn.Module):
    def __init__(self, dim=64, heads=4, kv_groups=2, dropout=0.2):
        super().__init__()

        assert dim % heads == 0

        self.dim = dim
        self.heads = heads
        self.kv_groups = kv_groups
        self.head_dim = dim // heads
        self.heads_per_group = heads // kv_groups

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, kv_groups * self.head_dim)
        self.v_proj = nn.Linear(dim, kv_groups * self.head_dim)

        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        b, t, d = x.shape

        q = self.q_proj(x).view(b, t, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.kv_groups, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.kv_groups, self.head_dim).transpose(1, 2)

        q = apply_rope(q)
        k = apply_rope(k)

        k = k.repeat_interleave(self.heads_per_group, dim=1)
        v = v.repeat_interleave(self.heads_per_group, dim=1)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(b, t, d)

        return self.out_proj(out)


class TMSABranch(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.local3 = nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.local5 = nn.Conv1d(dim, dim, kernel_size=5, padding=2, groups=dim)
        self.local7 = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)

        self.global_attn = nn.MultiheadAttention(dim, num_heads=4, batch_first=True)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(dim * 2, dim)
        )

    def forward(self, x):
        xt = x.transpose(1, 2)
        local = (self.local3(xt) + self.local5(xt) + self.local7(xt)).transpose(1, 2)
        global_out, _ = self.global_attn(x, x, x)

        x = self.norm1(x + local + global_out)
        x = self.norm2(x + self.ffn(x))

        return x


class GQABranch(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.gqa = GQAWithRoPE(dim=dim, heads=4, kv_groups=2)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(dim * 2, dim)
        )

    def forward(self, x):
        attn_out = self.gqa(x)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x


class FeatureFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )

        self.fusion = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(0.3)
        )

    def forward(self, tmsa_out, gqa_out):
        combined = torch.cat([tmsa_out, gqa_out], dim=-1)
        gate = self.gate(combined)

        fused = gate * tmsa_out + (1 - gate) * gqa_out
        fused = fused + self.fusion(combined)

        return fused


class CompactTCN(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm1d(dim),
            nn.ELU(),

            nn.Conv1d(dim, dim, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm1d(dim),
            nn.ELU(),

            nn.Conv1d(dim, dim, kernel_size=3, padding=4, dilation=4),
            nn.BatchNorm1d(dim),
            nn.ELU()
        )

    def forward(self, x):
        y = self.block(x.transpose(1, 2)).transpose(1, 2)
        return x + y


class HHATNet(nn.Module):
    def __init__(self, n_channels=22, n_classes=4, n_bands=5, dim=64):
        super().__init__()

        self.multiband_cnn = MultiBandCNN(n_bands=n_bands, out_ch=16)

        self.spatial = SpatialCNN(
            in_ch=48,
            n_channels=n_channels
        )

        self.grouped_se = GroupedSEAttention(channels=48)

        self.pool = nn.AvgPool2d(kernel_size=(1, 8))
        self.dropout = nn.Dropout(0.3)

        self.project = nn.Linear(48, dim)

        self.tmsa_branch = TMSABranch(dim)
        self.gqa_branch = GQABranch(dim)

        self.fusion = FeatureFusion(dim)
        self.tcn = CompactTCN(dim)

        self.embedding = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(0.3)
        )

        self.classifier = nn.Linear(dim, n_classes)

    def forward(self, x):
        # input: batch, 5, 22, time

        x = self.multiband_cnn(x)
        x = self.spatial(x)
        x = self.grouped_se(x)

        x = self.pool(x)
        x = self.dropout(x)

        x = x.squeeze(2)
        x = x.transpose(1, 2)

        x = self.project(x)

        tmsa_out = self.tmsa_branch(x)
        gqa_out = self.gqa_branch(x)

        x = self.fusion(tmsa_out, gqa_out)
        x = self.tcn(x)

        x = x.mean(dim=1)
        x = self.embedding(x)

        return self.classifier(x)