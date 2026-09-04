"""Windowed sink attention (WSA) utilities, vendored from smulelabs/windowed-roformer.

Implements the FlexAttention-based windowed attention with global sink tokens used
by "Efficient Vocal Source Separation Through Windowed Sink Attention" models
(checkpoints carrying a `sink_tokens` parameter, e.g. mbr_wsa). Requires
torch >= 2.5. Kept as a drop-in module so the main mel_band_roformer.py only
imports it on the WSA code path.
"""
from functools import lru_cache
from typing import Optional

import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import (
    flex_attention,
    create_block_mask,
    _mask_mod_signature,
)


@lru_cache(maxsize=128)
def create_block_mask_cached(
    mask_mod: _mask_mod_signature,
    B: Optional[int],
    H: Optional[int],
    Q_LEN: int,
    KV_LEN: int,
    device: torch.device,
):
    block_mask = create_block_mask(
        mask_mod,
        B=B,
        H=H,
        Q_LEN=Q_LEN,
        KV_LEN=KV_LEN,
        device=device,
    )
    return block_mask


def get_compiled_flex_attention(compile: bool = True, mode: str = "default"):
    if compile:
        return torch.compile(flex_attention, dynamic=False, mode=mode)
    return flex_attention


def generate_sliding_window_with_sinks(
    window_size: int,
    num_sink_tokens: int,
) -> _mask_mod_signature:
    half_window = window_size // 2

    def sliding_window_with_global_sinks(b, h, q_idx, kv_idx):
        is_query_sink = q_idx < num_sink_tokens
        is_kv_sink = kv_idx < num_sink_tokens
        is_in_window = torch.abs(q_idx - kv_idx) <= half_window
        # Query sinks can attend to everything
        # Regular queries can attend to: sinks OR tokens within sliding window
        return is_query_sink | is_kv_sink | is_in_window

    sliding_window_with_global_sinks.__name__ = (
        f"sliding_window_w{window_size}_sinks{num_sink_tokens}"
    )
    return sliding_window_with_global_sinks


class FlexAttention(nn.Module):
    def __init__(
        self,
        mask_mod: _mask_mod_signature,
        dropout: float = 0.0,
        scale: Optional[float] = None,
        compile: bool = False,
        compile_mode: str = "max-autotune",
    ):
        super().__init__()
        self.mask_mod = mask_mod
        self.scale = scale
        self.dropout = dropout
        self.attn_dropout = nn.Dropout(dropout) if dropout > 0 else None
        # Get compiled or uncompiled flex_attention
        self.flex_attn_fn = get_compiled_flex_attention(compile, mode=compile_mode)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_heads, seq_len, head_dim = q.shape
        device = q.device
        # Create block mask (cached based on mask_mod and dimensions)
        # Note: We use None for B and H to allow broadcasting across batches/heads
        block_mask = create_block_mask_cached(
            self.mask_mod,
            B=None,
            H=None,
            Q_LEN=seq_len,
            KV_LEN=seq_len,
            device=device.type,
        )
        # Apply scale if specified
        if self.scale is not None:
            # Adjust query by scale factor
            default_scale = head_dim ** -0.5
            q = q * (self.scale / default_scale)
        # Apply flex attention with block mask
        out = self.flex_attn_fn(
            q, k, v,
            block_mask=block_mask,
            scale=self.scale,
        )
        # Apply dropout if training
        if self.training and self.attn_dropout is not None:
            out = self.attn_dropout(out)
        return out