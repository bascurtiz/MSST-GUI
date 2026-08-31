from functools import partial

import torch
from torch import nn, tensor, Tensor
from torch.nn import Module, ModuleList
import torch.nn.functional as F

from torch.utils.checkpoint import checkpoint

from beartype.typing import Tuple, Optional, Callable
from beartype import beartype

from einops import rearrange, pack, unpack

from rotary_embedding_torch import RotaryEmbedding

from models.bs_roformer.attend import Attend
from models.bs_roformer.bs_roformer import (
    RMSNorm,
    FeedForward,
    BandSplit,
    MaskEstimator,
    MLP,
    DEFAULT_FREQS_PER_BANDS,
)


def exists(val):
    return val is not None


def default(v, d):
    return v if exists(v) else d


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class LearnedRotaryEmbedding(Module):
    """
    Learned absolute position tables (cos/sin), shared across layers. Each
    transformer keeps a reference to the *same* shared Parameter objects, so
    the saved state_dict contains both the top-level name (cos_emb_time /
    sin_emb_time / cos_emb_freq / sin_emb_freq) and each per-attention path,
    exactly like the SW checkpoints.
    """

    def __init__(self, cos_emb, sin_emb):
        super().__init__()
        self.cos_emb = nn.Parameter(cos_emb)
        self.sin_emb = nn.Parameter(sin_emb)

    def apply(self, q, k):
        n = q.shape[-2]
        cos = self.cos_emb[:n].unsqueeze(0).unsqueeze(0)  # [1, 1, n, d]
        sin = self.sin_emb[:n].unsqueeze(0).unsqueeze(0)
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        return q, k


class AttentionSW(Module):
    def __init__(
            self,
            dim,
            heads=8,
            dim_head=64,
            dropout=0.,
            rotary_embed=None,
            learned_rotary=False,
            shared_qkv_bias=None,
            shared_out_bias=None,
            flash=True,
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        dim_inner = heads * dim_head

        self.rotary_embed = rotary_embed
        self.learned_rotary = learned_rotary

        self.attend = Attend(flash=flash, dropout=dropout)

        self.norm = RMSNorm(dim)
        self.to_qkv = nn.Linear(dim, dim_inner * 3, bias=True)
        if shared_qkv_bias is not None:
            self.to_qkv.bias = shared_qkv_bias

        self.to_gates = nn.Linear(dim, heads)

        self.to_out = nn.Sequential(
            nn.Linear(dim_inner, dim, bias=True),
            nn.Dropout(dropout)
        )
        if shared_out_bias is not None:
            self.to_out[0].bias = shared_out_bias

    def forward(self, x):
        x = self.norm(x)

        q, k, v = rearrange(self.to_qkv(x), 'b n (qkv h d) -> qkv b h n d', qkv=3, h=self.heads)

        if exists(self.rotary_embed):
            if self.learned_rotary:
                q, k = self.rotary_embed.apply(q, k)
            else:
                q = self.rotary_embed.rotate_queries_or_keys(q)
                k = self.rotary_embed.rotate_queries_or_keys(k)

        out = self.attend(q, k, v)

        gates = self.to_gates(x)
        out = out * rearrange(gates, 'b n h -> b h n 1').sigmoid()

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class TransformerSW(Module):
    def __init__(
            self,
            *,
            dim,
            depth,
            dim_head=64,
            heads=8,
            attn_dropout=0.,
            ff_dropout=0.,
            ff_mult=4,
            norm_output=True,
            rotary_embed=None,
            learned_rotary=False,
            shared_qkv_bias=None,
            shared_out_bias=None,
            flash_attn=True,
    ):
        super().__init__()
        self.layers = ModuleList([])

        for _ in range(depth):
            attn = AttentionSW(
                dim=dim,
                heads=heads,
                dim_head=dim_head,
                dropout=attn_dropout,
                rotary_embed=rotary_embed,
                learned_rotary=learned_rotary,
                shared_qkv_bias=shared_qkv_bias,
                shared_out_bias=shared_out_bias,
                flash=flash_attn
            )
            self.layers.append(ModuleList([
                attn,
                FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)
            ]))

        self.norm = RMSNorm(dim) if norm_output else nn.Identity()

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)


class BSRoformerSW(Module):

    @beartype
    def __init__(
            self,
            dim,
            *,
            depth,
            stereo=False,
            num_stems=1,
            time_transformer_depth=2,
            freq_transformer_depth=2,
            linear_transformer_depth=0,
            freqs_per_bands: Tuple[int, ...] = DEFAULT_FREQS_PER_BANDS,
            dim_head=64,
            heads=8,
            attn_dropout=0.,
            ff_dropout=0.,
            flash_attn=True,
            dim_freqs_in=1025,
            stft_n_fft=2048,
            stft_hop_length=512,
            stft_win_length=2048,
            stft_normalized=False,
            stft_window_fn: Optional[Callable] = None,
            zero_dc=True,
            mask_estimator_depth=2,
            multi_stft_resolution_loss_weight=1.,
            multi_stft_resolutions_window_sizes: Tuple[int, ...] = (4096, 2048, 1024, 512, 256),
            multi_stft_hop_size=147,
            multi_stft_normalized=False,
            multi_stft_window_fn: Callable = torch.hann_window,
            mlp_expansion_factor=4,
            use_torch_checkpoint=False,
            skip_connection=False,
            sw_seq_len_time=1151,
            use_shared_bias=True,
            position_mode: str = 'learned',
    ):
        """
        Shared-bias BS-Roformer variant used by the SW (super-wideband) and
        Logic 6-stem models.

        position_mode='learned' (SW `sw: true`):
            - learned absolute cos/sin position tables shared across layers
              (cos_emb_time/sin_emb_time/cos_emb_freq/sin_emb_freq) plus
              per-attention references.
            - shared biases named `shared_qkv_bias` / `shared_out_bias`.

        position_mode='rope' (Logic `bs_logic_6stem`):
            - per-attention RotaryEmbedding (buffer `freqs`), like the base
              BSRoformer but with a separate instance per attention.
            - shared biases named `linear_62_bias_0` / `linear_64_bias_0`.
        """
        super().__init__()

        self.stereo = stereo
        self.audio_channels = 2 if stereo else 1
        self.num_stems = num_stems
        self.use_torch_checkpoint = use_torch_checkpoint
        self.skip_connection = skip_connection

        dim_inner = heads * dim_head

        # shared biases across every attention layer — the parameter names must
        # match the checkpoint exactly (differ between SW and Logic).
        if position_mode == 'rope':
            self.linear_62_bias_0 = nn.Parameter(torch.zeros(dim_inner * 3))
            self.linear_64_bias_0 = nn.Parameter(torch.zeros(dim))
            shared_qkv_bias = self.linear_62_bias_0
            shared_out_bias = self.linear_64_bias_0
        else:
            self.shared_qkv_bias = nn.Parameter(torch.zeros(dim_inner * 3))
            self.shared_out_bias = nn.Parameter(torch.zeros(dim))
            shared_qkv_bias = self.shared_qkv_bias
            shared_out_bias = self.shared_out_bias

        self.layers = ModuleList([])

        transformer_kwargs = dict(
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            flash_attn=flash_attn,
            norm_output=False,
            shared_qkv_bias=shared_qkv_bias,
            shared_out_bias=shared_out_bias,
        )

        if position_mode == 'rope':
            # per-attention standard rotary embedding (separate freqs per layer)
            for _ in range(depth):
                if linear_transformer_depth > 0:
                    raise NotImplementedError(
                        "BSRoformerSW (rope) does not support linear_transformer_depth > 0"
                    )
                time_rotary_embed = RotaryEmbedding(dim=dim_head)
                freq_rotary_embed = RotaryEmbedding(dim=dim_head)
                self.layers.append(nn.ModuleList([
                    TransformerSW(
                        depth=time_transformer_depth,
                        rotary_embed=time_rotary_embed,
                        learned_rotary=False,
                        **transformer_kwargs
                    ),
                    TransformerSW(
                        depth=freq_transformer_depth,
                        rotary_embed=freq_rotary_embed,
                        learned_rotary=False,
                        **transformer_kwargs
                    ),
                ]))
        else:
            # learned absolute position tables, shared across all layers
            num_bands = len(freqs_per_bands)
            self.cos_emb_time = nn.Parameter(torch.zeros(sw_seq_len_time, dim_head))
            self.sin_emb_time = nn.Parameter(torch.zeros(sw_seq_len_time, dim_head))
            self.cos_emb_freq = nn.Parameter(torch.zeros(num_bands, dim_head))
            self.sin_emb_freq = nn.Parameter(torch.zeros(num_bands, dim_head))

            time_rotary_embed = LearnedRotaryEmbedding(self.cos_emb_time, self.sin_emb_time)
            freq_rotary_embed = LearnedRotaryEmbedding(self.cos_emb_freq, self.sin_emb_freq)

            for _ in range(depth):
                if linear_transformer_depth > 0:
                    raise NotImplementedError(
                        "BSRoformerSW does not support linear_transformer_depth > 0"
                    )
                self.layers.append(nn.ModuleList([
                    TransformerSW(
                        depth=time_transformer_depth,
                        rotary_embed=time_rotary_embed,
                        learned_rotary=True,
                        **transformer_kwargs
                    ),
                    TransformerSW(
                        depth=freq_transformer_depth,
                        rotary_embed=freq_rotary_embed,
                        learned_rotary=True,
                        **transformer_kwargs
                    ),
                ]))

        self.final_norm = RMSNorm(dim)

        self.stft_kwargs = dict(
            n_fft=stft_n_fft,
            hop_length=stft_hop_length,
            win_length=stft_win_length,
            normalized=stft_normalized
        )

        self.stft_window_fn = partial(default(stft_window_fn, torch.hann_window), stft_win_length)

        freqs = torch.stft(
            torch.randn(1, 4096), **self.stft_kwargs,
            window=torch.ones(stft_win_length), return_complex=True
        ).shape[1]

        assert len(freqs_per_bands) > 1
        assert sum(
            freqs_per_bands) == freqs, \
            f'the number of freqs in the bands must equal {freqs} based on the STFT settings, ' \
            f'but got {sum(freqs_per_bands)}'

        freqs_per_bands_with_complex = tuple(2 * f * self.audio_channels for f in freqs_per_bands)

        self.band_split = BandSplit(
            dim=dim,
            dim_inputs=freqs_per_bands_with_complex
        )

        self.mask_estimators = nn.ModuleList([])

        for _ in range(num_stems):
            mask_estimator = MaskEstimator(
                dim=dim,
                dim_inputs=freqs_per_bands_with_complex,
                depth=mask_estimator_depth,
                mlp_expansion_factor=mlp_expansion_factor,
            )
            self.mask_estimators.append(mask_estimator)

        self.zero_dc = zero_dc

        self.multi_stft_resolution_loss_weight = multi_stft_resolution_loss_weight
        self.multi_stft_resolutions_window_sizes = multi_stft_resolutions_window_sizes
        self.multi_stft_n_fft = stft_n_fft
        self.multi_stft_window_fn = multi_stft_window_fn

        self.multi_stft_kwargs = dict(
            hop_length=multi_stft_hop_size,
            normalized=multi_stft_normalized
        )

    def forward(
            self,
            raw_audio,
            target=None,
            active_stem_ids=None,
            return_loss_breakdown=False
    ):
        """
        b - batch
        f - freq
        t - time
        s - audio channel (1 for mono, 2 for stereo)
        n - number of 'stems'
        c - complex (2)
        d - feature dimension
        """

        device = raw_audio.device
        x_is_mps = True if device.type == "mps" else False

        if raw_audio.ndim == 2:
            raw_audio = rearrange(raw_audio, 'b t -> b 1 t')

        channels = raw_audio.shape[1]
        assert (not self.stereo and channels == 1) or (
                    self.stereo and channels == 2), \
            ('stereo needs to be set to True if passing in audio signal that is stereo (channel dimension of 2).'
             ' also need to be False if mono (channel dimension of 1)')

        # to stft
        raw_audio, batch_audio_channel_packed_shape = pack_one(raw_audio, '* t')

        stft_window = self.stft_window_fn(device=device)

        try:
            stft_repr = torch.stft(
                raw_audio,
                **self.stft_kwargs,
                window=stft_window,
                return_complex=True
            )
        except Exception:
            stft_repr = torch.stft(
                raw_audio.cpu() if x_is_mps else raw_audio,
                **self.stft_kwargs,
                window=stft_window.cpu() if x_is_mps else stft_window,
                return_complex=True
            ).to(device)
        stft_repr = torch.view_as_real(stft_repr)

        stft_repr = unpack_one(stft_repr, batch_audio_channel_packed_shape, '* f t c')

        # merge stereo / mono into the frequency, with frequency leading dimension, for band splitting
        stft_repr = rearrange(stft_repr, 'b s f t c -> b (f s) t c')

        x = rearrange(stft_repr, 'b f t c -> b t (f c)')

        if self.use_torch_checkpoint:
            x = checkpoint(self.band_split, x, use_reentrant=False)
        else:
            x = self.band_split(x)

        # axial / hierarchical attention
        store = [None] * len(self.layers)
        for i, transformer_block in enumerate(self.layers):
            time_transformer, freq_transformer = transformer_block

            if self.skip_connection:
                for j in range(i):
                    x = x + store[j]

            x = rearrange(x, 'b t f d -> b f t d')
            x, ps = pack([x], '* t d')

            if self.use_torch_checkpoint:
                x = checkpoint(time_transformer, x, use_reentrant=False)
            else:
                x = time_transformer(x)

            x, = unpack(x, ps, '* t d')
            x = rearrange(x, 'b f t d -> b t f d')
            x, ps = pack([x], '* f d')

            if self.use_torch_checkpoint:
                x = checkpoint(freq_transformer, x, use_reentrant=False)
            else:
                x = freq_transformer(x)

            x, = unpack(x, ps, '* f d')

            if self.skip_connection:
                store[i] = x

        x = self.final_norm(x)

        if active_stem_ids is None:
            heads = self.mask_estimators
            stem_ids = list(range(len(self.mask_estimators)))
        else:
            heads = [self.mask_estimators[i] for i in active_stem_ids]
            stem_ids = active_stem_ids

        num_stems = len(heads)

        if self.use_torch_checkpoint:
            mask = torch.stack([checkpoint(fn, x, use_reentrant=False) for fn in heads], dim=1)
        else:
            mask = torch.stack([fn(x) for fn in heads], dim=1)
        mask = rearrange(mask, 'b n t (f c) -> b n f t c', c=2)

        # modulate frequency representation
        stft_repr = rearrange(stft_repr, 'b f t c -> b 1 f t c')

        # complex number multiplication
        stft_repr = torch.view_as_complex(stft_repr)
        mask = torch.view_as_complex(mask)

        stft_repr = stft_repr * mask

        # istft
        stft_repr = rearrange(stft_repr, 'b n (f s) t -> (b n s) f t', s=self.audio_channels)

        if self.zero_dc:
            stft_repr = stft_repr.index_fill(1, tensor(0, device=device), 0.)

        try:
            recon_audio = torch.istft(
                stft_repr, **self.stft_kwargs, window=stft_window,
                return_complex=False, length=raw_audio.shape[-1]
            )
        except Exception:
            recon_audio = torch.istft(
                stft_repr.cpu() if x_is_mps else stft_repr,
                **self.stft_kwargs,
                window=stft_window.cpu() if x_is_mps else stft_window,
                return_complex=False,
                length=raw_audio.shape[-1]
            ).to(device)

        recon_audio = rearrange(
            recon_audio,
            '(b n s) t -> b n s t',
            s=self.audio_channels,
            n=num_stems
        )

        if not exists(target):
            return recon_audio

        if target.ndim == 2:
            target = rearrange(target, '... t -> ... 1 t')

        target = target[..., :recon_audio.shape[-1]]

        target_sel = target[:, stem_ids]
        loss = F.l1_loss(recon_audio, target_sel)

        multi_stft_resolution_loss = 0.

        for window_size in self.multi_stft_resolutions_window_sizes:
            res_stft_kwargs = dict(
                n_fft=max(window_size, self.multi_stft_n_fft),
                win_length=window_size,
                return_complex=True,
                window=self.multi_stft_window_fn(window_size, device=device),
                **self.multi_stft_kwargs,
            )

            recon_Y = torch.stft(rearrange(recon_audio, 'b n s t -> (b n s) t'), **res_stft_kwargs)
            target_Y = torch.stft(rearrange(target_sel, 'b n s t -> (b n s) t'), **res_stft_kwargs)

            multi_stft_resolution_loss = multi_stft_resolution_loss + F.l1_loss(recon_Y, target_Y)

        weighted_multi_resolution_loss = multi_stft_resolution_loss * self.multi_stft_resolution_loss_weight

        total_loss = loss + weighted_multi_resolution_loss

        if not return_loss_breakdown:
            return total_loss

        return total_loss, (loss, multi_stft_resolution_loss)


def pack_one(t, pattern):
    return pack([t], pattern)


def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]