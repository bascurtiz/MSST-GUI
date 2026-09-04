"""Regression: the three mel-band model failure classes from the 72-model run.

1. mbr_expl_jazzpear (and any mel-band checkpoint whose mask-estimator MLP
   depth disagrees with its YAML) — the engine must sniff the depth baked
   into the checkpoint and build the matching architecture instead of dying
   with "missing key / size mismatch".
2. mbr_denoise_children_phaedrus33 — a mono model (stereo: false) fed a
   stereo source crashed in the model's channel assert; demix now downmixes
   stereo input for mono models.
3. mbr_4stemxl1_aname — the chunk accumulator finalization blew up with
   numpy._core._exceptions._ArrayMemoryError right at the end; the new
   _finalize_sources divides in place and cleans non-finites in slices so
   the temporary masks never exceed a few MB.
4. mbr_wsa — a windowed-sink-attention (WSA) checkpoint carrying a learned
   ``sink_tokens`` embedding; the bundled MelBandRoformer must build that
   parameter and run FlexAttention windowed time attention, or the strict
   load_state_dict dies with unexpected key "sink_tokens".
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from utils.model_utils import _finalize_sources  # noqa: E402
from utils.settings import _sniff_melband_mask_estimator_depth  # noqa: E402

FAILURES = []
CHECKS = 0


def check(cond, msg):
    global CHECKS
    CHECKS += 1
    if not cond:
        FAILURES.append(msg)


def _head_keys(depth, bands=2, wrapped=False):
    """Fake mel-band mask-estimator keys with `depth` MLP layers per band."""
    sd = {}
    for band in range(bands):
        for layer in range(depth + 1):
            sd[f"mask_estimators.0.to_freqs.{band}.0.{2 * layer}.weight"] = torch.zeros(4)
            sd[f"mask_estimators.0.to_freqs.{band}.0.{2 * layer}.bias"] = torch.zeros(4)
    # A couple of non-head keys that must be ignored.
    sd["band_split.to_features.0.1.weight"] = torch.zeros(2)
    return {"model_state_dict": sd} if wrapped else sd


def main():
    # --- depth sniffing ----------------------------------------------------
    with tempfile.TemporaryDirectory() as tmp:
        def save(name, sd):
            path = os.path.join(tmp, name)
            torch.save(sd, path)
            return path

        p1 = save("depth1.ckpt", _head_keys(1))
        p2 = save("depth2.ckpt", _head_keys(2))
        p3 = save("depth1_wrapped.ckpt", _head_keys(1, wrapped=True))
        pnone = save("junk.ckpt", {"not": "a model"})
        pempty = save("empty.ckpt", {})

        check(_sniff_melband_mask_estimator_depth(p1) == 1,
              "depth-1 head must sniff as mask_estimator_depth 1")
        check(_sniff_melband_mask_estimator_depth(p2) == 2,
              "depth-2 head must sniff as mask_estimator_depth 2")
        check(_sniff_melband_mask_estimator_depth(p3) == 1,
              "wrapped model_state_dict must still sniff")
        check(_sniff_melband_mask_estimator_depth(pnone) is None,
              "non-mel-band checkpoint -> None")
        check(_sniff_melband_mask_estimator_depth(pempty) is None,
              "empty checkpoint -> None")
        check(_sniff_melband_mask_estimator_depth(
            os.path.join(tmp, "missing.ckpt")) is None,
            "missing file -> None")

    # --- finalization: in-place mean + bounded nan cleanup + border trim ----
    rng = np.random.default_rng(0)
    # (num_instruments, channels, samples) accumulator pair
    result = torch.from_numpy(rng.random((2, 2, 200), dtype=np.float32))
    counter = torch.ones((2, 2, 200), dtype=torch.float32)
    counter[..., 100:110] = 0.0  # 0/0 region -> NaN after division
    expected = (result / counter).numpy()[..., 25:-25]  # pre-division reference
    out = _finalize_sources(result, counter, border=25)
    check(out.shape == (2, 2, 150), f"border must be trimmed (got {out.shape})")
    check(np.isfinite(out).all(), "no NaN/Inf may survive finalization")
    check(np.allclose(out, np.nan_to_num(expected, nan=0.0)),
          "finalization must equal result/counter (trimmed, nan->0)")
    # untouched data path: no border
    result2 = torch.ones((1, 1, 32), dtype=torch.float32)
    counter2 = torch.full((1, 1, 32), 2.0, dtype=torch.float32)
    out2 = _finalize_sources(result2, counter2, border=0)
    check(out2.shape == (1, 1, 32) and float(out2[0, 0, 0]) == 0.5,
          "border=0 keeps full length and divides correctly")

    # --- WSA (windowed sink attention): mbr_wsa-family checkpoints ---------
    # carry a learned ``sink_tokens`` embedding plus windowed FlexAttention
    # time attention. The bundled MelBandRoformer must build that parameter
    # (so a strict load_state_dict finds no unexpected missing keys) and must
    # strip the sinks from the time axis before mask estimation. The plain
    # model must stay byte-equivalent in behaviour (no sink key, same output).
    from models.bs_roformer.mel_band_roformer import MelBandRoformer  # noqa: E402

    kw = dict(dim=16, depth=1, num_stems=1, time_transformer_depth=1,
              freq_transformer_depth=1, num_bands=4,
              attn_dropout=0.0, ff_dropout=0.0)
    with torch.no_grad():
        wsa = MelBandRoformer(stereo=True, num_sink_tokens=2, window_size=8, **kw)
        plain = MelBandRoformer(stereo=True, **kw)
        x = torch.randn(1, 2, 16384)
        yw, yp = wsa(x), plain(x)

    check("sink_tokens" in wsa.state_dict(),
          "WSA model must expose sink_tokens in its state dict "
          "(the mbr_wsa checkpoint stores it)")
    check(tuple(wsa.state_dict()["sink_tokens"].shape) == (2, 4, 16),
          "sink_tokens shape must be (num_sink_tokens, num_bands, dim)")
    check("sink_tokens" not in plain.state_dict(),
          "non-WSA model must NOT expose sink_tokens in its state dict")
    check(tuple(yw.shape) == tuple(x.shape) and bool(torch.isfinite(yw).all()),
          "WSA forward must return a same-shape finite output")
    check(tuple(yp.shape) == tuple(x.shape) and bool(torch.isfinite(yp).all()),
          "standard forward unaffected by the WSA plumbing")

    # The engine's kwarg filter must forward the WSA params to the model
    # constructor, otherwise the architecture wouldn't match the checkpoint
    # (the exact failure mbr_wsa hit: unexpected key \"sink_tokens\").
    from utils.settings import _fit_model_kwargs  # noqa: E402

    fitted = _fit_model_kwargs(
        MelBandRoformer, {"num_sink_tokens": 8, "window_size": 4})
    check(fitted.get("num_sink_tokens") == 8 and fitted.get("window_size") == 4,
          "_fit_model_kwargs must forward num_sink_tokens/window_size "
          "to the model constructor")

    # --- demix mono guard: a stereo input reaching a stereo:false model ----
    # must be downmixed to one channel before the model ever sees it.
    from utils.model_utils import demix  # noqa: E402

    class MonoFake(torch.nn.Module):
        stereo = False

        def forward(self, x):
            # Real mel-band models assert on channels here; record what arrived.
            self.saw_channels = x.shape[1]
            return torch.zeros_like(x)

    model = MonoFake()
    cfg = type("Cfg", (), {})()
    import ml_collections
    cfg = ml_collections.ConfigDict()
    cfg.inference = {"chunk_size": 64, "num_overlap": 2, "batch_size": 1,
                     "normalize": False}
    cfg.audio = {"chunk_size": 64}
    cfg.training = {"instruments": ["speech", "noise"],
                    "target_instrument": "speech"}
    mix = np.random.default_rng(1).random((2, 128)).astype(np.float32) * 0.5
    demix(cfg, model, mix, "cpu", "mel_band_roformer")
    check(model.saw_channels == 1,
          "stereo input to a stereo:false model must be downmixed to mono")

    if FAILURES:
        print(f"{len(FAILURES)}/{CHECKS} checks FAILED:")
        for f in FAILURES:
            print(" -", f)
        return 1
    print(f"ALL {CHECKS} CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
