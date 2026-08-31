"""
Medley-Vox (conv-tasnet + STFT) support.

Medley-Vox models (https://github.com/jeonchangbin49/MedleyVox) are Conv-TasNet
style encoder/masker/decoder models built entirely on the `asteroid` /
`asteroid_filterbanks` libraries (already present in the runtime). Their
checkpoints store both an `online_model` and an `ema_model` (DataParallel
wrapped), plus `initted`/`step` scalars.

The wrapper below mirrors the checkpoint layout exactly so a strict
``load_state_dict`` succeeds, and forwards with the EMA weights (the ones
the config selects for inference).

Two variants are supported, selected by ``config.model.mixture_consistency``:
- "mixture_consistency"  -> base enc/mask/dec with mixture-consistent output
- "sfsrnet"               -> same, plus a super-resolution head (SFSRNet /
                             SFSRNet_ConvNext) on the STFT magnitudes

The models are mono (trained at 24 kHz), so stereo input is downmixed for the
forward pass and the mono stems are broadcast back to the input channels.
"""

import torch
import torch.nn as nn

from asteroid_filterbanks import make_enc_dec
from asteroid_filterbanks.transforms import mag, magphase, from_magphase
from asteroid.masknn import TDConvNet, norms
from asteroid.models.base_models import (
    BaseEncoderMaskerDecoder,
    _unsqueeze_to_3d,
    _shape_reconstructed,
)
from asteroid.utils.torch_utils import pad_x_to_y, jitable_shape


def normalize_mag_spec(S, min_level_db=-100.0):
    return torch.clamp((S - min_level_db) / -min_level_db, min=0.0, max=1.0)


def denormalize_mag_spec(S, min_level_db=-100.0):
    return torch.clamp(S, min=0.0, max=1.0) * -min_level_db + min_level_db


class BaseEncoderMaskerDecoder_mixture_consistency(BaseEncoderMaskerDecoder):
    """Enc/mask/dec model with mixture-consistent output (Wisdom et al.)."""

    def forward(self, wav):
        shape = jitable_shape(wav)
        wav = _unsqueeze_to_3d(wav)

        tf_rep = self.forward_encoder(wav)
        est_masks = self.forward_masker(tf_rep)
        masked_tf_rep = self.apply_masks(tf_rep, est_masks)
        decoded = self.forward_decoder(masked_tf_rep)

        reconstructed = _shape_reconstructed(pad_x_to_y(decoded, wav), shape)

        reconstructed = reconstructed + 1 / reconstructed.shape[1] * (
            wav - reconstructed.sum(dim=1, keepdim=True)
        )

        return reconstructed


class SFSRNet(nn.Module):
    """SFSRNet from Rixon et al., AAAI 2022 (as used by MedleyVox)."""

    def __init__(self, n_src=2, norm_type="gLN"):
        super().__init__()
        input_channels = 1 + n_src * 2
        conv_norm = norms.get(norm_type)

        self.conv_1 = nn.Conv2d(input_channels, 128, 5, padding=5 // 2)
        self.ln_1 = conv_norm(128)
        self.conv_2 = nn.Conv2d(128, 256, 9, padding=9 // 2)
        self.ln_2 = conv_norm(256)
        self.conv_3 = nn.Conv2d(256, 128, 11, padding=11 // 2)
        self.ln_3 = conv_norm(128)
        self.conv_4 = nn.Conv2d(128, n_src, 11, padding=11 // 2)
        self.ln_4 = conv_norm(n_src)

        self.relu = nn.ReLU()

    def forward(self, mix_mag, est_mag, heuristic_out):
        inp = torch.cat([mix_mag, est_mag, heuristic_out], dim=1)
        out = self.ln_2(
            self.relu(self.conv_2(self.ln_1(self.relu(self.conv_1(inp)))))
        )
        out = self.ln_4(
            self.relu(self.conv_4(self.ln_3(self.relu(self.conv_3(out)))))
        )
        return out


class SFSRNet_ConvNext(nn.Module):
    """ConvNext-style SFSRNet used by the MedleyVox ISRNet models."""

    def __init__(self, n_src=2, norm_type="gLN"):
        super().__init__()
        input_channels = 1 + n_src * 2
        conv_norm = norms.get(norm_type)

        self.conv_0 = nn.Conv2d(input_channels, 96, 5, padding=5 // 2)

        self.conv_1 = nn.Conv2d(96, 96, 7, padding=7 // 2, groups=96)
        self.ln_1 = conv_norm(96)
        self.conv_2 = nn.Conv2d(96, 384, 1)
        self.conv_3 = nn.Conv2d(384, 96, 1)

        self.conv_4 = nn.Conv2d(96, 96, 1, groups=96)
        self.ln_4 = conv_norm(96)
        self.conv_5 = nn.Conv2d(96, 384, 1)
        self.conv_6 = nn.Conv2d(384, 96, 1)

        self.conv_out = nn.Conv2d(96, n_src, 1)
        self.ln_out = conv_norm(n_src)

        self.gelu = nn.GELU()
        self.relu = nn.ReLU()

    def forward(self, mix_mag, est_mag, heuristic_out):
        inp = torch.cat([mix_mag, est_mag, heuristic_out], dim=1)
        out = self.gelu(self.conv_0(inp))
        out_1 = self.conv_3(self.gelu(self.conv_2(self.ln_1(self.conv_1(out)))))
        out_2 = self.conv_6(self.gelu(self.conv_5(self.ln_4(self.conv_4(out + out_1)))))
        out_3 = self.relu(self.ln_out(self.conv_out(out_1 + out_2)))
        return out_3


class BaseEncoderMaskerDecoder_mixture_consistency_super_resolution(
    BaseEncoderMaskerDecoder
):
    """Enc/mask/dec + super-resolution model (as used by MedleyVox ISRNet)."""

    def __init__(
        self,
        encoder,
        masker,
        decoder,
        sr_net,
        window_size=2048,
        above_freq=3000.0,
        sample_rate=24000,
        encoder_activation=None,
        db_normalize=False,
        sr_input_res=False,
        sr_out_mix_consistency=False,
    ):
        super().__init__(encoder, masker, decoder, encoder_activation)
        self.sr_net = sr_net
        total_n_bins = int(1 + window_size / 2)
        stft_bins_freqs = (
            torch.arange(0, total_n_bins).float() * sample_rate / window_size
        )
        self.over_freq_index = int((stft_bins_freqs >= above_freq).nonzero()[0].item())
        self.db_normalize = db_normalize
        self.sr_input_res = sr_input_res
        self.sr_out_mix_consistency = sr_out_mix_consistency

    def forward_pre(self, wav):
        shape = jitable_shape(wav)
        wav = _unsqueeze_to_3d(wav)

        tf_rep = self.forward_encoder(wav)
        est_masks = self.forward_masker(tf_rep)
        masked_tf_rep = self.apply_masks(tf_rep, est_masks)
        decoded = self.forward_decoder(masked_tf_rep)

        reconstructed = _shape_reconstructed(pad_x_to_y(decoded, wav), shape)
        return reconstructed

    def forward_sr(self, wav, reconstructed):
        shape = jitable_shape(wav)
        wav = _unsqueeze_to_3d(wav)

        tf_rep = self.forward_encoder(wav)
        out_est_stft = self.forward_encoder(reconstructed)

        tf_rep = tf_rep.unsqueeze(1)
        mix_mag = mag(tf_rep)
        est_mag, est_phase = magphase(out_est_stft)

        if self.db_normalize:
            import torchaudio

            mix_mag = normalize_mag_spec(
                torchaudio.functional.amplitude_to_DB(
                    mix_mag, multiplier=20.0, amin=1e-5, db_multiplier=1.0
                )
            )
            est_mag = normalize_mag_spec(
                torchaudio.functional.amplitude_to_DB(
                    est_mag, multiplier=20.0, amin=1e-5, db_multiplier=1.0
                )
            )

        heuristic_out = self.heuristic(mix_mag, est_mag)
        sr_out = self.sr_net(mix_mag, est_mag, heuristic_out)

        if self.sr_input_res:
            sr_out = sr_out + est_mag

        if self.db_normalize:
            import torchaudio

            sr_out = torchaudio.functional.DB_to_amplitude(
                denormalize_mag_spec(sr_out), ref=1.0, power=0.5
            )

        sr_out_stft = from_magphase(sr_out, est_phase)
        sr_out_decoded = self.forward_decoder(sr_out_stft)
        sr_out_recon = _shape_reconstructed(pad_x_to_y(sr_out_decoded, wav), shape)

        if self.sr_out_mix_consistency:
            sr_out_recon = sr_out_recon + 1 / sr_out_recon.shape[1] * (
                wav - sr_out_recon.sum(dim=1, keepdim=True)
            )

        return sr_out_recon

    def heuristic(self, mix_mag, est_mag):
        mix_sum_freq = mix_mag[..., : self.over_freq_index, :].sum(
            dim=-2, keepdim=True
        )
        est_sum_freq = est_mag[..., : self.over_freq_index, :].sum(
            dim=-2, keepdim=True
        )
        ratio = mix_sum_freq / (est_sum_freq + 1e-5)
        mix_high_freqs = mix_mag[..., self.over_freq_index:, :]
        heuristic_out = mix_high_freqs * ratio
        heuristic_out = torch.cat(
            [
                torch.zeros(
                    [
                        mix_high_freqs.shape[0],
                        heuristic_out.shape[1],
                        self.over_freq_index,
                        mix_high_freqs.shape[3],
                    ],
                    dtype=mix_high_freqs.dtype,
                    device=mix_high_freqs.device,
                ),
                heuristic_out,
            ],
            dim=-2,
        )
        return heuristic_out

    def forward(self, wav):
        reconstructed = self.forward_pre(wav)
        sr_out_recon = self.forward_sr(wav, reconstructed)
        return sr_out_recon


def _build_model(config):
    """Build the raw enc/mask/dec (or +SR) model from a Medley-Vox config."""
    m = config.model
    encoder, decoder = make_enc_dec(
        "torch_stft",
        n_filters=m.nfft,
        kernel_size=m.nfft,
        stride=m.nhop,
        sample_rate=m.sample_rate,
    )
    masker = TDConvNet(
        in_chan=encoder.n_feats_out,
        n_src=m.n_src,
        out_chan=None,
        n_blocks=m.n_blocks,
        n_repeats=m.n_repeats,
        bn_chan=m.bn_chan,
        hid_chan=m.hid_chan,
        skip_chan=m.skip_chan,
        mask_act=m.mask_act,
    )

    mixture_consistency = getattr(m, "mixture_consistency", "mixture_consistency")
    if mixture_consistency == "sfsrnet":
        srnet = getattr(m, "srnet", "orig")
        if srnet == "convnext":
            sr_net = SFSRNet_ConvNext(n_src=m.n_src, norm_type="gLN")
        else:
            sr_net = SFSRNet(n_src=m.n_src, norm_type="gLN")
        model = BaseEncoderMaskerDecoder_mixture_consistency_super_resolution(
            encoder,
            masker,
            decoder,
            sr_net,
            window_size=m.nfft,
            above_freq=getattr(m, "above_freq", 3000.0),
            sample_rate=m.sample_rate,
            encoder_activation=getattr(m, "encoder_activation", None),
            db_normalize=bool(getattr(m, "db_normalize", False)),
            sr_input_res=bool(getattr(m, "sr_input_res", False)),
            sr_out_mix_consistency=bool(getattr(m, "sr_out_mix_consistency", False)),
        )
    elif mixture_consistency == "mixture_consistency":
        model = BaseEncoderMaskerDecoder_mixture_consistency(
            encoder,
            masker,
            decoder,
            encoder_activation=getattr(m, "encoder_activation", None),
        )
    else:
        model = BaseEncoderMaskerDecoder(
            encoder,
            masker,
            decoder,
            encoder_activation=getattr(m, "encoder_activation", None),
        )
    return model


class _DataParallelWrap(nn.Module):
    """Mirrors the `model.module` nesting found in the checkpoints."""

    def __init__(self, model):
        super().__init__()
        self.module = model


class MedleyVoxModel(nn.Module):
    """Checkpoint-compatible wrapper around a Medley-Vox model.

    The checkpoint stores both ``online_model.module.*`` and
    ``ema_model.module.*`` state dicts plus ``initted``/``step`` scalars; this
    class mirrors that layout so a strict load succeeds. Inference uses the
    EMA weights (selected by ``config.model.use_ema_model``).
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.online_model = _DataParallelWrap(_build_model(config))
        self.ema_model = _DataParallelWrap(_build_model(config))
        # Non-persistent: these training bookkeeping scalars vary in shape
        # across checkpoints ([] vs [1]) and never affect inference, so they
        # are excluded from the strict state-dict contract and handled by the
        # load_state_dict override below.
        self.register_buffer("initted", torch.tensor([1.0]), persistent=False)
        self.register_buffer("step", torch.tensor([0]), persistent=False)

        use_ema = bool(getattr(config.model, "use_ema_model", True))
        self._use_ema = use_ema

    def _core(self):
        return self.ema_model.module if self._use_ema else self.online_model.module

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """Load a checkpoint state dict, tolerating `initted`/`step` shape drift.

        Different Medley-Vox checkpoints save these training bookkeeping
        scalars as either ``torch.Size([])`` or ``torch.Size([1])``; they don't
        affect inference. Strip them from the strict load and copy their value
        into the buffers with shape/type adaption so both forms load."""
        sd = dict(state_dict)
        for key in ("initted", "step"):
            if key in sd:
                val = sd.pop(key)
                buf = getattr(self, key)
                try:
                    buf.copy_(val.reshape(buf.shape).to(buf.dtype))
                except Exception:
                    pass
        return super().load_state_dict(sd, strict=strict, assign=assign)

    def forward(self, mix):
        """mix: (batch, channels, time) -> (batch, n_src, channels, time)."""
        num_channels = mix.shape[1]
        if num_channels > 1:
            mono = mix.mean(dim=1, keepdim=True)
        else:
            mono = mix
        # The STFT overlap-add is not fp16-safe; always run this model fp32.
        with torch.cuda.amp.autocast(enabled=False):
            out = self._core()(mono)  # (batch, n_src, time)
        return out.unsqueeze(2).expand(-1, -1, num_channels, -1)


def build_medley_vox(config):
    """Build the Medley-Vox wrapper and fill in inference defaults.

    The yaml configs ship without a chunk size; the training ``seq_dur`` (in
    seconds at the model sample rate) is the natural chunk length.
    """
    model = MedleyVoxModel(config)

    sample_rate = int(config.audio.sample_rate)
    seq_dur = float(getattr(config.model, "seq_dur", 3.0))
    chunk_size = int(seq_dur * sample_rate)

    if not hasattr(config.inference, "chunk_size") and not hasattr(
        config.audio, "chunk_size"
    ):
        config.audio.chunk_size = chunk_size
    if not hasattr(config.inference, "num_overlap"):
        config.inference.num_overlap = 2
    if not hasattr(config.inference, "batch_size"):
        config.inference.batch_size = 1
    # STFT overlap-add is not fp16-safe; disable autocast for this model.
    config.training.use_amp = False

    return model
