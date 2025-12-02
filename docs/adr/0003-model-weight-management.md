# ADR 0003: Management of Third-Party ML Model Weights (Including MedleyVox Checkpoint Integration)

## Status
Accepted

## Date
2024-12-02

## Context

This project depends on external ML model weights (e.g., `.pth` files from Hugging Face) for audio separation tasks. These weights must be usable in:

- **Local development** on a PC
- **Production** environments (e.g., Docker, serverless)

Large binary files cannot be committed to git. One of the required third-party models is **MedleyVox**, which provides lead/backing vocal separation capabilities.

## Decision

### A. Do Not Commit Weights to Git

- Add `models/` and `*.pth` to `.gitignore`
- All weights are treated as external artifacts

### B. Dedicated Loader Subsystem in `src/harmony_trainer/models/`

Define a canonical loader architecture with a module for each third-party model (e.g., `medleyvox.py`).

Each loader module must define:

1. **Canonical source** (HF repo ID)
2. **Canonical filename**
3. **Canonical revision** (pinned to a known commit or branch)
4. **`get_<model>_checkpoint_path()`** function with fallback hierarchy:
   - Environment variable override (`<MODEL>_WEIGHTS_PATH`)
   - Optional local `models/<model>/...` directory
   - `hf_hub_download()` as final fallback
5. **`load_<model>_model()`** function constructing the architecture and loading the state dict

### C. Specific Implementation for MedleyVox

| Property | Value |
|----------|-------|
| Hugging Face repo | `Cyru5/MedleyVox` |
| Checkpoint file | `vocals 100.pth` |
| Revision | `main` (default branch) |
| Environment variable | `MEDLEYVOX_WEIGHTS_PATH` |
| Local project path | `models/medleyvox/vocals_100.pth` |

The loader uses:

```python
from huggingface_hub import hf_hub_download

hf_hub_download(
    repo_id="Cyru5/MedleyVox",
    filename="vocals 100.pth",
    revision="main",
)
```

**Note:** This checkpoint is from a third-party Hugging Face repository (`Cyru5/MedleyVox`), not from the official MedleyVox authors.

MedleyVox is loaded through the generic loader subsystem via `src/harmony_trainer/models/medleyvox.py`.

### D. Local PC Development Workflow

Developers may either:

1. **Allow `hf_hub_download()` to automatically populate the Hugging Face cache** — the default behavior requiring no manual steps
2. **Manually place the weight file** in `models/medleyvox/vocals_100.pth` (ignored by git)

Code operates identically in both cases via the fallback hierarchy.

### E. Production Deployment Strategy

**Preferred (Option 1): Bake weights into the Docker image**

- During Docker build, download MedleyVox checkpoint into `/opt/models/medleyvox/vocals_100.pth`
- Set `MEDLEYVOX_WEIGHTS_PATH=/opt/models/medleyvox/vocals_100.pth`
- Ensures deterministic and network-free startup

**Alternative (Option 2): Download on first startup**

- Uses `hf_hub_download()` with pinned revision
- Only acceptable when build pipeline prohibits embedding weights in image

**This project chooses Option 1 (bake into Docker image) as the official default.**

## Consequences

### Positive

- **Reproducibility** via pinned HF revision
- **No binary artifacts in git**
- **Clean separation** between code and model artifacts
- **Consistent loader logic** across dev and prod
- **Flexibility** to override file paths via environment variables
- **Predictable Docker behavior** with deterministic startup

### Negative

- Docker images will be larger due to embedded weights
- Initial setup requires either network access or manual file placement

## Out of Scope

- Training pipelines
- Retraining MedleyVox
- Long-term model storage choices (S3, GCS, etc.)
- Architecture of MedleyVox model internals beyond loading requirements

## Alternatives Considered

### Git LFS for weights
Rejected: Adds complexity, storage costs, and doesn't solve the fundamental issue of large binaries in version control.

### Download weights at runtime only
Rejected as default: Creates non-deterministic startup behavior and network dependency in production.

### Vendor weights in a separate repository
Rejected: Adds coordination overhead without significant benefits over the chosen approach.
