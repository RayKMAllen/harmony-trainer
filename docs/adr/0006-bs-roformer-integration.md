# ADR 0006: BS-RoFormer Integration as Alternative Lead/Backing Separator

## Status

Accepted

## Date

2025-12-03

## Context

MedleyVox (ADR-0005) is the current lead/backing vocal separator. To improve separation quality and provide user choice, we are adding BS-RoFormer (Band-Split RoFormer) as a parallel alternative.

BS-RoFormer is a state-of-the-art music source separation model. The weights and config are available at `anvuew/BS-RoFormer` on Hugging Face:
- `bs_roformer_anvuew_sdr_12.45.ckpt` (204 MB) - trained weights
- `config.yaml` (1.97 kB) - model hyperparameters

The model architecture code is maintained in `ZFTurbo/Music-Source-Separation-Training` (MIT licensed).

## Decision

### A. Vendor BS-RoFormer model architecture

- Add `src/harmony_trainer/models/bs_roformer.py` containing the BS-RoFormer network architecture.
- Source: `ZFTurbo/Music-Source-Separation-Training/models/bs_roformer/` (~500 lines).
- Include MIT license header in vendored files per upstream license.
- Inference-only: no training code, CLI, or dataset utilities.
- Dependencies `einops` and `rotary-embedding-torch` added to `pyproject.toml` as required.

### B. Hardcode model configuration in Python

- The `config.yaml` hyperparameters from `anvuew/BS-RoFormer` will be hardcoded as class defaults in `BSRoformerNet`.
- No runtime YAML parsing required.
- If a new checkpoint with different config is used, update the Python defaults.

### C. Loader subsystem for BS-RoFormer weights

Create `src/harmony_trainer/model_loaders/bs_roformer.py` with:

| Property | Value |
|----------|-------|
| Hugging Face repo | `anvuew/BS-RoFormer` |
| Checkpoint file | `bs_roformer_anvuew_sdr_12.45.ckpt` |
| Revision | `main` |
| Environment variable | `BS_ROFORMER_WEIGHTS_PATH` |
| Local project path | `models/bs_roformer/bs_roformer_anvuew_sdr_12.45.ckpt` |

Functions:
- `get_bs_roformer_checkpoint_path()` - fallback hierarchy per ADR-0003
- `load_bs_roformer_model()` - returns `torch.nn.Module` in eval mode on CPU

### D. BSRoFormerWrapper in separation module

Create `src/harmony_trainer/separation/bs_roformer.py` with:

- `BSRoFormerWrapper` class matching `MedleyVoxWrapper` interface
- `separate(vocals_path, output_dir) -> tuple[Path, Path]` returning `(lead.wav, backing.wav)`
- Auto-resample input to model's expected sample rate
- Standardized output names: `lead.wav`, `backing.wav` (matching MedleyVox)
- Errors wrapped in `SeparationError` per ADR-0004

### E. Model selection strategy

- Both `MedleyVoxWrapper` and `BSRoFormerWrapper` are available as parallel options.
- User explicitly instantiates the wrapper they want.
- No automatic fallback or registry-based selection at this time.
- Future ADR may add a unified interface or selection mechanism.

### F. Audio format handling

- Wrapper auto-resamples input to model's expected sample rate.
- Converts stereo to mono if required by model.
- Output saved as WAV at the model's native sample rate.

## Consequences

### Positive

- Provides alternative separation backend for quality comparison.
- Consistent UX with MedleyVox via matching wrapper interface.
- Vendored architecture ensures stability and reproducibility.
- MIT license allows vendoring with attribution.

### Negative

- Adds ~500 lines of vendored code to maintain.
- Two new dependencies (`einops`, `rotary-embedding-torch`).
- Larger Docker images with additional model weights.

## Alternatives Considered

### Use ZFTurbo repo as pip dependency
Rejected: Not on PyPI, heavy repo with training code, unstable API.

### Use audio-separator library
Rejected: Adds abstraction layer, external maintenance dependency.

### Download config.yaml at runtime
Rejected: Adds complexity; hardcoding is simpler and matches MedleyVox pattern.
