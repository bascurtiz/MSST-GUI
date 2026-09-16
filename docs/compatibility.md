> **MSST-GUI note:** This file is copied from [ZFTurbo/Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training). The GUI vendors the classic scripts (inference.py, train.py, valid.py) rather than pip install msst / msst.Separator. Use this document to understand upstream CLI and Python APIs; job subprocesses still call the vendored scripts.


# Model and audio compatibility

## Python

MSST supports CPython 3.10, 3.11, 3.12, and 3.13. The package metadata enforces
`Python >=3.10,<3.14`; CI exercises the oldest and newest supported versions.
Compatibility also depends on whether the selected model extra publishes
wheels for the target Python, operating system, and GPU stack.

## PyTorch and CUDA

MSST supports `torch>=2.0.1,<2.12`. Training and model extras that use
TorchAudio apply the same version range; TorchAudio's own metadata selects the
matching Torch release.

CUDA binaries are not bundled by MSST. On a GPU system, install the wheel from
the appropriate [PyTorch package index](https://pytorch.org/get-started/locally/)
before installing MSST so pip keeps the hardware-compatible build. Verify that
the selected wheel includes kernels for the target GPU architecture; the newest
CUDA wheel does not necessarily support every older architecture.

Triton, Torch-TensorRT, and TorchAO are not MSST runtime dependencies.

## Model extras

Install only the dependencies needed by the selected `model_type`:

```bash
pip install "msst[mel_band_roformer]"
pip install "msst[train,mel_band_roformer]"
```

Python package extras normalize hyphens and underscores, so both
`mel_band_roformer` and `mel-band-roformer` are accepted by pip.

Install dependencies for all generally available model families with:

```bash
pip install "msst[all-models]"
```

`all-models` excludes dependencies restricted to a narrow platform. These stay
in the extra for their architecture, which prevents one unavailable model from
blocking installation of the others. In particular, BSMamba2 requires
`msst[bs_mamba2]` on a supported system. Pip treats one installation command as
a single transaction and cannot otherwise skip an arbitrary dependency that
fails to install.

| `model_type` | pip extra | Notes |
| --- | --- | --- |
| `apollo` | `apollo` |  |
| `bandit` | `bandit` | Full BandIt runtime stack. |
| `bandit_v2` | `bandit_v2` |  |
| `bs_conformer` | `bs_conformer` |  |
| `bs_mamba2` | `bs_mamba2` | `mamba-ssm` is supported here only on Linux x86_64 with a compatible CUDA build. |
| `bs_roformer` | `bs_roformer` | Includes optional PoPE support. |
| `bs_roformer_experimental` | `bs_roformer_experimental` |  |
| `conformer` | `conformer` |  |
| `experimental_mdx23c_stht` | none | Base dependencies are sufficient. |
| `htdemucs` | `htdemucs` |  |
| `mdx23c` | none | Base dependencies are sufficient. |
| `mel_band_conformer` | `mel_band_conformer` |  |
| `mel_band_roformer` | `mel_band_roformer` | Includes optional PoPE support. |
| `mel_band_roformer_experimental` | `mel_band_roformer_experimental` |  |
| `moises_light` | `moises_light` |  |
| `scnet` | none | Base dependencies are sufficient. |
| `scnet_masked` | none | Base dependencies are sufficient. |
| `scnet_tran` | `scnet_tran` |  |
| `scnet_unofficial` | `scnet_unofficial` | Does not require `mamba-ssm`. |
| `segm_models` | `segm_models` | Pretrained encoder weights are used only for a new training run, not inference or checkpoint resume. |
| `swin_upernet` | `swin_upernet` | Inference construction is offline and does not download Hugging Face weights. |
| `torchseg` | `torchseg` | Pretrained encoder weights are used only for a new training run. |

`OptionalDependencyError` reports the matching install command. BSMamba2 gives
a platform-specific message on unsupported systems instead of suggesting an
installation that cannot succeed.

## Input audio

MSST first filters folder entries by extension, then asks librosa/SoundFile to
decode and resample them. The following formats are decoded natively by the
libsndfile build bundled with current SoundFile wheels and were covered by the
package validation matrix:

| Family | Extensions |
| --- | --- |
| Wave | `.wav`, `.wave`, `.rf64`, `.w64` |
| Lossless | `.flac`, `.aif`, `.aifc`, `.aiff` |
| Ogg | `.ogg`, `.oga`, `.opus` |
| MPEG audio | `.mp3` |
| Other libsndfile containers | `.au`, `.snd`, `.caf`, `.voc` |

Actual codec availability still depends on the installed SoundFile/libsndfile
build. A recognized but unreadable file raises `AudioFileError` containing the
path and underlying decoder error.

These extensions are accepted conditionally: `.3gp`, `.aac`, `.ac3`, `.amr`,
`.m4a`, `.m4b`, `.mka`, `.mp4`, `.webm`, and `.wma`. They need a compatible
system decoder, normally FFmpeg. They are not portable guarantees: librosa's
current audioread fallback is deprecated and is scheduled for removal in
librosa 1.0. Prefer WAV or FLAC for datasets, evaluation, and long-lived
pipelines. See the official [librosa I/O formats guide](https://librosa.org/doc/0.11.0/ioformats.html)
and [`librosa.load` documentation](https://librosa.org/doc/latest/generated/librosa.load.html).

Lossy codecs can add encoder delay or padding. In validation, M4A/AAC decoded
to a slightly different sample count, so it must not be used where sample-exact
alignment matters.

Unknown extensions are ignored during folder processing and rejected by
`Separator.separate_file`. A corrupt file with a recognized extension raises a
clear per-file error. With `skip_errors=True`, folder processing logs it and
continues.

## Output audio

The supported output containers are intentionally narrow:

| Output | Default subtype | Allowed public subtypes |
| --- | --- | --- |
| WAV | `FLOAT` | `FLOAT`, `PCM_24`, `PCM_16` |
| FLAC | `PCM_24` | `PCM_24`, `PCM_16` |

`flac_file=True, pcm_type="FLOAT"` is rejected before model execution; it is
never silently changed to another subtype.

The default output template is `{relative_path}/{instr}`. It preserves input
subfolders, prevents albums containing identically named tracks from
overwriting one another, and rejects any remaining collision. When a directory
contains the same basename in multiple codecs, add `{extension}` to the
template, for example `{relative_path}_{extension}/{instr}`.

## Device behavior

- `device_ids` accepts one integer, a sequence of integers, or the legacy
  string form such as `"0,1"`.
- Empty, negative, duplicate, non-integer, and out-of-range CUDA IDs are
  rejected with the requested IDs and detected GPU count in the message.
- `force_cpu=True` takes precedence over CUDA availability.
- If CUDA is not available and CPU was not explicitly requested, inference
  falls back to CPU. Model failures are not silently converted to CPU retries.
