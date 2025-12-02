# ADR 0005: MedleyVox Model Architecture Integration and Inference

## Status

Accepted

## Date

2025-12-02

## Context

MedleyVox is one of the experimental lead/backing separation backends referenced in the
high-level architecture and in ADR-0003 (model weight management). The current implementation provides:

- A loader subsystem in `src/harmony_trainer/model_loaders/medleyvox.py` that resolves
  a checkpoint file (`vocals 100.pth`) via environment variable, local path, or
  Hugging Face Hub (HF repo `Cyru5/MedleyVox`).
- A wrapper `MedleyVoxWrapper` in `src/harmony_trainer/separation/medleyvox.py` that
  prepares the audio input but raises `SeparationError` instead of running inference.

This means we can *locate and load* a checkpoint object, but we do not currently have:

- A vendored MedleyVox network architecture class inside this project.
- A `torch.nn.Module` returned by `load_medleyvox_model()` that can be executed.
- A defined contract for how MedleyVox maps inputs to lead/backing stems.

ADR-0003 states that loader modules must provide both a checkpoint locator and a
`load_<model>_model()` function that constructs the architecture and loads the state
dict. ADR-0004 defines `SeparationError` as the common failure channel for separation
backends. This ADR fills the gap for MedleyVox by specifying how the model is
vendored, instantiated, and invoked.

## Decision

### A. Vendor MedleyVox model architecture under `harmony_trainer.models`

- Add a new module `src/harmony_trainer/models/medleyvox.py` that contains the
  MedleyVox network architecture copied/ported from the public MedleyVox GitHub repo.
- The module will expose a single canonical entrypoint class (e.g. `MedleyVoxNet`)
  whose constructor and `forward()` match the configuration used to train the
  `vocals 100.pth` checkpoint.
- The module is inference-only: no training loops, CLIs, or dataset code are
  vendored, only what is required to construct and run the model.
- Any upstream changes are not automatically tracked; if we update to a new
  architecture or checkpoint, this ADR will be superseded by a later ADR.

### B. Change `load_medleyvox_model()` to return `torch.nn.Module`

- `load_medleyvox_model()` in
  `src/harmony_trainer/model_loaders/medleyvox.py` will be updated to:
  - Resolve `checkpoint_path` via `get_medleyvox_checkpoint_path()` (per ADR-0003).
  - Load the file with `torch.load(..., map_location="cpu", weights_only=False)`.
  - Normalize checkpoint formats into a single `state_dict`:
    - If the loaded object has a `state_dict` key, use that.
    - Else if it has a `model` key containing a plain state dict, use that.
    - Else if it is already a mapping of parameter tensors, treat it as the
      state dict directly.
    - Otherwise raise a `RuntimeError` that will be wrapped in `SeparationError`.
  - Instantiate the MedleyVox network from `harmony_trainer.models.medleyvox` using
    the hyperparameters that correspond to the checkpoint.
  - Load the state dict with `model.load_state_dict(..., strict=True)` (or
    `strict=False` with logged mismatches if required by the actual checkpoint).
  - Return the fully constructed `torch.nn.Module` on CPU in evaluation mode.
- The public signature and name of `load_medleyvox_model()` remain the same; only
  the return type and behavior are strengthened to match ADR-0003.

### C. Define MedleyVox separation semantics in `MedleyVoxWrapper`

- `MedleyVoxWrapper` in `src/harmony_trainer/separation/medleyvox.py` will change
  its cached `_model` type from `dict[str, Any]` to `torch.nn.Module`.
- The `model` property will:
  - Call `load_medleyvox_model(self.checkpoint_path)`.
  - Cache the returned module.
  - Wrap `CheckpointNotFoundError` and `RuntimeError` in `SeparationError` as
    already defined by ADR-0004.
- `_do_separate()` will:
  - Load the input audio via `harmony_trainer.audio_io.load_audio()`.
  - Resample to the MedleyVox sample rate (assumed 44.1 kHz unless the upstream
    repo specifies otherwise) using `torchaudio.transforms.Resample`.
  - Convert stereo to mono or mono to the expected channel layout based on the
    upstream MedleyVox inference script.
  - Add a batch dimension and move the tensor to GPU if available, falling back
    to CPU otherwise.
  - Run a forward pass under `torch.no_grad()`.
  - Interpret the model outputs as follows:
    - If the model produces two vocal stems, map them to (lead, backing).
    - If the model produces only a single vocal stem, treat that as `lead` and
      derive `backing` as the residual `mixture - lead`.
  - Convert mono predictions back to stereo for saving, matching the project
    convention for `lead.wav` and `backing.wav`.
  - Save outputs via `_save_outputs()` and return the resulting paths.
- Any errors in device moves, forward passes, or output interpretation will be
  wrapped and surfaced as `SeparationError` to the CLI and higher layers.

### D. Testing and contracts

- Add unit tests around `get_medleyvox_checkpoint_path()` and
  `load_medleyvox_model()` to confirm:
  - Fallback hierarchy behavior remains as in ADR-0003.
  - Invalid or missing checkpoints raise loader-level errors that become
    `SeparationError` in the wrapper.
- Add at least one integration-style test that monkeypatches
  `load_medleyvox_model()` to return a tiny fake `nn.Module` with the same
  input/output contract and verifies that:
  - `separate_vocals_medleyvox()` produces `lead.wav` and `backing.wav` files.
  - The sample rate and tensor shapes are as expected.
- End-to-end tests that rely on real MedleyVox weights are optional and may be
  added in a separate evaluation harness (M3) to avoid heavy test dependencies.

## Consequences

### Positive

- Brings MedleyVox in line with ADR-0003 by ensuring the loader returns a
  constructed `torch.nn.Module` instead of a raw checkpoint dict.
- Enables actual lead/backing separation using MedleyVox rather than a stub.
- Keeps separation error handling consistent with ADR-0004 via `SeparationError`.
- Centralizes MedleyVox architecture under `harmony_trainer.models`, mirroring the
  SVoice integration pattern.

### Negative

- Vendoring MedleyVox architecture increases the maintenance surface; updates in
  the upstream repo will not be picked up automatically.
- Real MedleyVox inference is heavier than the current stub and may require
  GPU resources for acceptable performance.
- Test coverage for the real model will likely rely on synthetic or small
  fixtures to keep CI runtime reasonable.

### Alternatives Considered

- **Keep MedleyVox as a stub only**:
  - Rejected because the project’s core goal is to evaluate backing extraction
    quality, which requires at least one working lead/backing splitter beyond
    SVoice.
- **Depend on the MedleyVox repo as an external Python package or git submodule**:
  - Rejected to avoid bringing in training code, large or unstable dependencies,
    and to keep deployment simple and self-contained.
- **Use a different pretrained model instead of MedleyVox**:
  - Out of scope for this ADR; could be revisited in a future ADR if MedleyVox
    proves impractical in evaluation.
