"""
RecdSER backbone: Swin-FFA-Net V2 network definition (~101M params).

This file contains ONLY the network definition of the Swin-FFA-Net backbone
used by the RecdSER project. No training code, no checkpoint loader, no
downstream head. Head is defined in ``head.py`` and the composed classifier
in ``classifier.py``.

Architecture summary:
  Input : raw waveform -> 80-dim Mel spectrogram (B, 80, T) -> (B, T, 80)

  Branch A: 2-Stage hierarchical Swin Transformer branch
    Patch Embedding -> Stage1 Swin Blocks (dim=embed_dim, W=8) -> Patch Merging
    -> Stage2 Swin Blocks (dim=embed_dim*2, W=16) -> F_temp (B, T//4, 2*embed_dim)

  Branch B: ConvNeXt-style frequency branch -> F_freq (B, T//4, 2*embed_dim)

  Cross-Branch Interaction: bidirectional cross-attention -> F_temp', F_freq'

  Fusion: concat -> 1x1 Linear -> AvgPool+MaxPool -> (B, 4*embed_dim)

  Projection Head (optional): Linear -> GELU -> Linear -> L2 normalize

Output (forward): (B, proj_dim) L2-normalized utterance embedding, or
(B, 4*embed_dim) when the projection head is disabled
(see ``extract_backbone_embedding``).
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def safe_l2_normalize(x, dim=-1, eps=1e-6):
    """FP32 L2 normalize, avoiding eps underflow under FP16 (0/0 -> NaN)."""
    x = torch.nan_to_num(x.float(), nan=0.0, posinf=1e4, neginf=-1e4)
    x = x.clamp(-1e4, 1e4)
    return F.normalize(x, p=2, dim=dim, eps=eps)


# ======================== Mel Spectrogram Front-end ========================

class MelSpectrogramFrontend(nn.Module):
    """Convert raw waveform into an 80-dim Mel spectrogram."""

    def __init__(self, sample_rate=16000, n_fft=400, hop_length=160, n_mels=80):
        super().__init__()
        self.mel_spec = _build_torchaudio_mel(sample_rate, n_fft, hop_length, n_mels)
        self.hop_length = hop_length

    def forward(self, waveform, padding_mask=None):
        with torch.cuda.amp.autocast(enabled=False):
            mel = self.mel_spec(waveform.float())
            mel = torch.log(mel.clamp(min=1e-5))
        mel = mel.transpose(1, 2)

        mel_mask = None
        if padding_mask is not None:
            lengths = (~padding_mask).sum(dim=1)
            mel_lengths = lengths // self.hop_length
            T_mel = mel.size(1)
            mel_mask = torch.arange(T_mel, device=mel.device).unsqueeze(0) >= mel_lengths.unsqueeze(1)

        return mel, mel_mask


def _build_torchaudio_mel(sample_rate, n_fft, hop_length, n_mels):
    import torchaudio
    return torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length,
        n_mels=n_mels, power=2.0,
    )


# ======================== Drop Path ========================

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth)."""

    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = torch.floor(random_tensor + keep_prob)
        return x / keep_prob * random_tensor


# ======================== Branch A: 2-Stage Hierarchical Swin Transformer ========================

class WindowAttention(nn.Module):
    """Window self-attention (1D temporal version)."""

    def __init__(self, dim, num_heads=8, window_size=8, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.scale = (dim // num_heads) ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(2 * window_size - 1, num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        coords = torch.arange(window_size)
        relative_coords = coords.unsqueeze(0) - coords.unsqueeze(1)
        relative_coords += window_size - 1
        self.register_buffer("relative_position_index", relative_coords)

    def forward(self, x):
        B_W, W, C = x.shape
        qkv = self.qkv(x).reshape(B_W, W, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        rel_pos_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size, self.window_size, -1
        )
        rel_pos_bias = rel_pos_bias.permute(2, 0, 1)
        attn = attn + rel_pos_bias.unsqueeze(0)

        attn = attn.clamp(-50.0, 50.0)
        attn = F.softmax(attn.float(), dim=-1).type_as(q)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_W, W, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    """Temporal Swin Transformer Block (1D version)."""

    def __init__(self, dim, num_heads=8, window_size=8, shift_size=0,
                 mlp_ratio=4.0, drop=0.0, attn_drop=0.0, drop_path=0.1):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim, num_heads=num_heads, window_size=window_size,
            attn_drop=attn_drop, proj_drop=drop,
        )
        self.drop_path = nn.Identity() if drop_path <= 0. else DropPath(drop_path)
        self.norm2 = nn.LayerNorm(dim)

        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(drop),
        )

    def forward(self, x):
        B, T, C = x.shape

        shortcut = x
        x = self.norm1(x)

        if self.shift_size > 0:
            x = torch.roll(x, shifts=-self.shift_size, dims=1)

        pad_r = (self.window_size - T % self.window_size) % self.window_size
        if pad_r > 0:
            x = F.pad(x, (0, 0, 0, pad_r))

        T_padded = x.size(1)
        num_windows = T_padded // self.window_size
        x = x.view(B, num_windows, self.window_size, C)
        x = x.reshape(B * num_windows, self.window_size, C)

        x = self.attn(x)

        x = x.view(B, num_windows, self.window_size, C)
        x = x.reshape(B, T_padded, C)

        if pad_r > 0:
            x = x[:, :T, :]

        if self.shift_size > 0:
            x = torch.roll(x, shifts=self.shift_size, dims=1)

        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PatchMerging1D(nn.Module):
    """
    1D Patch Merging: merge adjacent two frames -> downsample + up-dimension.
    (B, T, dim) -> (B, T//2, dim*2)
    """

    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(2 * dim)
        self.reduction = nn.Linear(2 * dim, 2 * dim, bias=False)

    def forward(self, x):
        B, T, C = x.shape

        if T % 2 != 0:
            x = F.pad(x, (0, 0, 0, 1))
            T = T + 1

        x0 = x[:, 0::2, :]
        x1 = x[:, 1::2, :]
        x = torch.cat([x0, x1], dim=-1)

        x = self.norm(x)
        x = self.reduction(x)
        return x


class TemporalSwinBranch(nn.Module):
    """
    Branch A: 2-Stage hierarchical Swin Transformer branch.
    Output: (B, T//4, embed_dim*2)
    """

    def __init__(self, in_dim=80, embed_dim=192, num_heads=8, window_size=8,
                 mlp_ratio=4.0, drop=0.1, attn_drop=0.1, drop_path=0.1,
                 stage1_blocks=3, stage2_blocks=3):
        super().__init__()
        self.embed_dim = embed_dim

        self.patch_embed = nn.Sequential(
            nn.Conv1d(in_dim, embed_dim, kernel_size=3, stride=2, padding=1),
            nn.LayerNorm(embed_dim),
        )

        self.stage1 = nn.ModuleList()
        for i in range(stage1_blocks):
            ws = window_size
            shift_size = 0 if (i % 2 == 0) else ws // 2
            self.stage1.append(SwinTransformerBlock(
                dim=embed_dim, num_heads=num_heads, window_size=ws,
                shift_size=shift_size, mlp_ratio=mlp_ratio,
                drop=drop, attn_drop=attn_drop, drop_path=drop_path,
            ))

        self.patch_merging = PatchMerging1D(embed_dim)

        stage2_dim = embed_dim * 2
        self.stage2 = nn.ModuleList()
        for i in range(stage2_blocks):
            ws = window_size * 2
            shift_size = 0 if (i % 2 == 0) else ws // 2
            self.stage2.append(SwinTransformerBlock(
                dim=stage2_dim, num_heads=num_heads, window_size=ws,
                shift_size=shift_size, mlp_ratio=mlp_ratio,
                drop=drop, attn_drop=attn_drop, drop_path=drop_path,
            ))

        self.out_norm = nn.LayerNorm(stage2_dim)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.patch_embed[0](x)
        x = x.transpose(1, 2)
        x = self.patch_embed[1](x)

        for block in self.stage1:
            x = block(x)

        x = self.patch_merging(x)

        for block in self.stage2:
            x = block(x)

        x = self.out_norm(x)
        return x


# ======================== Branch B: ConvNeXt-style Frequency Branch ========================

class ConvNeXtBlock1D(nn.Module):
    """
    ConvNeXt-style 1D Block:
    DWConv(k=7) -> LayerNorm -> Linear(4x expand) -> GELU -> Linear(compress) -> SE -> residual
    """

    def __init__(self, dim, out_dim=None, kernel_size=7, expansion=4, se_reduction=4, drop=0.1, drop_path=0.1):
        super().__init__()
        out_dim = out_dim or dim
        padding = kernel_size // 2

        self.dwconv = nn.Conv1d(dim, dim, kernel_size, padding=padding, groups=dim)
        self.norm = nn.LayerNorm(dim)

        hidden_dim = dim * expansion
        self.pwconv1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(hidden_dim, out_dim)

        self.se = nn.Sequential(nn.AdaptiveAvgPool1d(1))
        self.se_fc = nn.Sequential(
            nn.Linear(out_dim, out_dim // se_reduction),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim // se_reduction, out_dim),
            nn.Sigmoid(),
        )

        self.residual_proj = nn.Linear(dim, out_dim) if dim != out_dim else nn.Identity()

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.dropout = nn.Dropout(drop)

    def forward(self, x):
        residual = self.residual_proj(x)

        h = x.transpose(1, 2)
        h = self.dwconv(h)
        h = h.transpose(1, 2)

        h = self.norm(h)
        h = self.pwconv1(h)
        h = self.act(h)
        h = self.pwconv2(h)
        h = self.dropout(h)

        se_input = h.transpose(1, 2)
        se_weight = F.adaptive_avg_pool1d(se_input, 1).squeeze(-1)
        se_weight = self.se_fc(se_weight).unsqueeze(1)
        h = h * se_weight

        x = residual + self.drop_path(h)
        return x


class FrequencyBranch(nn.Module):
    """
    Branch B: ConvNeXt-style frequency feature extraction branch.
    Output: F_freq (B, T//4, out_dim)
    """

    def __init__(self, in_dim=80, compress_dim=128, out_dim=384, num_blocks=4,
                 kernel_size=7, se_reduction=4, drop=0.1, drop_path=0.1):
        super().__init__()
        self.compress = nn.Sequential(
            nn.Linear(in_dim, compress_dim),
            nn.LayerNorm(compress_dim),
            nn.GELU(),
        )

        dims = self._compute_dims(compress_dim, out_dim, num_blocks)
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            self.blocks.append(ConvNeXtBlock1D(
                dim=dims[i],
                out_dim=dims[i + 1],
                kernel_size=kernel_size,
                expansion=4,
                se_reduction=se_reduction,
                drop=drop,
                drop_path=drop_path,
            ))

        self.out_norm = nn.LayerNorm(out_dim)

    def _compute_dims(self, in_dim, out_dim, num_blocks):
        dims = []
        for i in range(num_blocks + 1):
            d = in_dim + (out_dim - in_dim) * i // num_blocks
            dims.append(d)
        return dims

    def forward(self, x, target_len=None):
        x = self.compress(x)

        for block in self.blocks:
            x = block(x)

        x = self.out_norm(x)

        if target_len is not None:
            x = x.transpose(1, 2)
            x = F.adaptive_avg_pool1d(x, target_len)
            x = x.transpose(1, 2)
        else:
            T = x.size(1)
            x = x.transpose(1, 2)
            x = F.adaptive_avg_pool1d(x, T // 4)
            x = x.transpose(1, 2)

        return x


# ======================== Cross-Branch Interaction ========================

class CrossBranchAttention(nn.Module):
    """Bidirectional cross-attention between the two branches."""

    def __init__(self, dim=384, num_heads=8, drop=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.norm_t = nn.LayerNorm(dim)
        self.norm_f_for_t = nn.LayerNorm(dim)
        self.q_t = nn.Linear(dim, dim)
        self.kv_f = nn.Linear(dim, dim * 2)
        self.proj_t = nn.Linear(dim, dim)

        self.norm_f = nn.LayerNorm(dim)
        self.norm_t_for_f = nn.LayerNorm(dim)
        self.q_f = nn.Linear(dim, dim)
        self.kv_t = nn.Linear(dim, dim * 2)
        self.proj_f = nn.Linear(dim, dim)

        self.attn_drop = nn.Dropout(drop)
        self.drop_path = DropPath(0.1)

        self.norm_t2 = nn.LayerNorm(dim)
        self.ffn_t = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(dim * 4, dim),
            nn.Dropout(drop),
        )

        self.norm_f2 = nn.LayerNorm(dim)
        self.ffn_f = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(dim * 4, dim),
            nn.Dropout(drop),
        )

    def _cross_attn(self, q_input, kv_input, q_proj, kv_proj, out_proj):
        B, T_q, C = q_input.shape
        T_kv = kv_input.size(1)

        q = q_proj(q_input).reshape(B, T_q, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = kv_proj(kv_input).reshape(B, T_kv, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.clamp(-50.0, 50.0)
        attn = F.softmax(attn.float(), dim=-1).type_as(q)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, T_q, C)
        out = out_proj(out)
        return out

    def forward(self, f_temp, f_freq):
        t_normed = self.norm_t(f_temp)
        f_normed_for_t = self.norm_f_for_t(f_freq)
        cross_t = self._cross_attn(t_normed, f_normed_for_t, self.q_t, self.kv_f, self.proj_t)
        f_temp = f_temp + self.drop_path(cross_t)
        f_temp = f_temp + self.drop_path(self.ffn_t(self.norm_t2(f_temp)))

        f_normed = self.norm_f(f_freq)
        t_normed_for_f = self.norm_t_for_f(f_temp)
        cross_f = self._cross_attn(f_normed, t_normed_for_f, self.q_f, self.kv_t, self.proj_f)
        f_freq = f_freq + self.drop_path(cross_f)
        f_freq = f_freq + self.drop_path(self.ffn_f(self.norm_f2(f_freq)))

        return f_temp, f_freq


# ======================== Fusion Block ========================

class FusionBlock(nn.Module):
    """Dual-branch fusion. Output: (B, 2*hidden_dim) fused feature."""

    def __init__(self, in_dim=768, hidden_dim=384):
        super().__init__()
        self.conv1x1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

    def forward(self, f_temp, f_freq):
        f_cat = torch.cat([f_temp, f_freq], dim=-1)
        f_fuse = self.conv1x1(f_cat)

        f_fuse_t = f_fuse.transpose(1, 2)
        avg_pool = F.adaptive_avg_pool1d(f_fuse_t, 1).squeeze(-1)
        max_pool = F.adaptive_max_pool1d(f_fuse_t, 1).squeeze(-1)

        out = torch.cat([avg_pool, max_pool], dim=-1)
        return out


# ======================== Full Backbone ========================

class SwinFFANet(nn.Module):
    """
    Swin-FFA-Net V2: dual-branch Speech Emotion Recognition backbone (~101M params).

    Produces an L2-normalized utterance-level embedding.

    Args:
        sample_rate / n_mels / n_fft / hop_length: Mel front-end params.
        embed_dim: Swin Stage1 dim (Stage2 = embed_dim*2).
        num_heads / window_size / mlp_ratio: attention params.
        stage1_blocks / stage2_blocks: Swin block counts.
        freq_compress_dim / num_freq_blocks / freq_kernel_size: ConvNeXt branch.
        drop / attn_drop / drop_path: regularizations.
        proj_dim: projection head output dim (0 = disabled).
    """

    def __init__(
        self,
        sample_rate=16000,
        n_mels=80,
        n_fft=400,
        hop_length=160,
        embed_dim=192,
        num_heads=8,
        window_size=8,
        mlp_ratio=4.0,
        stage1_blocks=3,
        stage2_blocks=3,
        freq_compress_dim=128,
        num_freq_blocks=4,
        freq_kernel_size=7,
        drop=0.1,
        attn_drop=0.1,
        drop_path=0.1,
        proj_dim=256,
    ):
        super().__init__()

        out_dim = embed_dim * 2

        self.mel_frontend = MelSpectrogramFrontend(
            sample_rate=sample_rate, n_fft=n_fft,
            hop_length=hop_length, n_mels=n_mels,
        )

        self.temporal_branch = TemporalSwinBranch(
            in_dim=n_mels, embed_dim=embed_dim, num_heads=num_heads,
            window_size=window_size, mlp_ratio=mlp_ratio,
            drop=drop, attn_drop=attn_drop, drop_path=drop_path,
            stage1_blocks=stage1_blocks, stage2_blocks=stage2_blocks,
        )

        self.frequency_branch = FrequencyBranch(
            in_dim=n_mels, compress_dim=freq_compress_dim, out_dim=out_dim,
            num_blocks=num_freq_blocks, kernel_size=freq_kernel_size,
            drop=drop, drop_path=drop_path,
        )

        self.cross_branch = CrossBranchAttention(
            dim=out_dim, num_heads=num_heads, drop=drop,
        )

        self.fusion = FusionBlock(in_dim=out_dim * 2, hidden_dim=out_dim)

        fusion_out_dim = out_dim * 2
        self.proj_head = None
        if proj_dim > 0:
            self.proj_head = nn.Sequential(
                nn.Linear(fusion_out_dim, fusion_out_dim),
                nn.GELU(),
                nn.Dropout(drop),
                nn.Linear(fusion_out_dim, proj_dim),
            )

        self.embed_dim_out = fusion_out_dim
        self.proj_dim = proj_dim

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            if m.elementwise_affine:
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, source, padding_mask=None):
        """Return (B, D) L2-normalized embedding."""
        mel, mel_mask = self.mel_frontend(source, padding_mask)

        f_temp = self.temporal_branch(mel)
        T_quarter = f_temp.size(1)
        f_freq = self.frequency_branch(mel, target_len=T_quarter)

        f_temp, f_freq = self.cross_branch(f_temp, f_freq)

        embedding = self.fusion(f_temp, f_freq)

        if self.proj_head is not None:
            embedding = self.proj_head(embedding)

        embedding = safe_l2_normalize(embedding, dim=-1, eps=1e-6)
        return embedding

    def extract_backbone_embedding(self, source, padding_mask=None):
        """Extract backbone embedding (without projection head): (B, 4*embed_dim) L2-normalized."""
        mel, mel_mask = self.mel_frontend(source, padding_mask)
        f_temp = self.temporal_branch(mel)
        T_quarter = f_temp.size(1)
        f_freq = self.frequency_branch(mel, target_len=T_quarter)
        f_temp, f_freq = self.cross_branch(f_temp, f_freq)
        embedding = self.fusion(f_temp, f_freq)
        return safe_l2_normalize(embedding, dim=-1, eps=1e-6)

    def extract_cross_attention_frames(self, source, padding_mask=None):
        """
        Per-frame features after bidirectional cross-attention (concat of two branches).

        Returns:
            frames: (B, T//4, 2*out_dim)
            frame_mask: (B, T//4) True = padding; None if no padding_mask
        """
        mel, mel_mask = self.mel_frontend(source, padding_mask)
        f_temp = self.temporal_branch(mel)
        T_quarter = f_temp.size(1)
        f_freq = self.frequency_branch(mel, target_len=T_quarter)

        f_temp, f_freq = self.cross_branch(f_temp, f_freq)

        frames = torch.cat([f_temp, f_freq], dim=-1)

        frame_mask = None
        if mel_mask is not None:
            mel_lengths = (~mel_mask).sum(dim=1)
            frame_lengths = torch.div(
                torch.div(mel_lengths, 2, rounding_mode="floor") + 1,
                2, rounding_mode="floor",
            ).clamp(min=1, max=T_quarter)
            arange = torch.arange(T_quarter, device=frames.device).unsqueeze(0)
            frame_mask = arange >= frame_lengths.unsqueeze(1)

        return frames, frame_mask


def build_swin_ffa_net(cfg: dict) -> SwinFFANet:
    """Instantiate a SwinFFANet from a flat dict config (see ``modules/config.yaml``)."""
    def _g(key, default):
        return cfg.get(key, default)

    return SwinFFANet(
        sample_rate=_g("sample_rate", 16000),
        n_mels=_g("n_mels", 80),
        n_fft=_g("n_fft", 1024),
        hop_length=_g("hop_length", 256),
        embed_dim=_g("embed_dim", 384),
        num_heads=_g("num_heads", 12),
        window_size=_g("window_size", 8),
        mlp_ratio=_g("mlp_ratio", 4.0),
        stage1_blocks=_g("stage1_blocks", 6),
        stage2_blocks=_g("stage2_blocks", 8),
        freq_compress_dim=_g("freq_compress_dim", 256),
        num_freq_blocks=_g("num_freq_blocks", 6),
        freq_kernel_size=_g("freq_kernel_size", 7),
        drop=_g("drop", 0.1),
        attn_drop=_g("attn_drop", 0.1),
        drop_path=_g("drop_path", 0.2),
        proj_dim=_g("proj_dim", 768),
    )
