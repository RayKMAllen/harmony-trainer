# ADR 0008: BS-RoFormer Architecture Alignment with ZFTurbo

## Status

Accepted

## Date

2025-12-03

## Context

Our BS-RoFormer implementation (ADR-0006) was vendored from ZFTurbo/Music-Source-Separation-Training. The anvuew/BS-RoFormer checkpoint was trained using ZFTurbo's codebase. A detailed line-by-line comparison revealed critical discrepancies that will cause incorrect inference.

**Detailed comparison with ZFTurbo source:**

| Component | Our Implementation | ZFTurbo | Status |
|-----------|-------------------|---------|--------|
| RMSNorm | `F.normalize()` | `F.normalize()` | ✅ Match |
| FeedForward | GELU | GELU | ✅ Match |
| Attention | `scaled_dot_product_attention` | Same | ✅ Match |
| Transformer | ModuleList + residuals | Same | ✅ Match |
| BandSplit | ModuleList of [RMSNorm, Linear] | Same | ✅ Match |
| MLP | Extra `nn.Sequential` wrapper | Direct `self.net = nn.Sequential(...)` | ❌ **BREAKS KEYS** |
| MaskEstimator output | `complex_out = audio_channels * 2 * 2` | `freq * 2 * audio_channels` | ❌ **WRONG OUTPUT** |
| Mask application | Complex multiplication `(a+bi)(c+di)` | Element-wise real/imag multiply | ❌ **WRONG MATH** |
| Stem handling | Hardcoded `stems=2` in rearrange | Flexible `num_stems` | ⚠️ Hardcoded |
| Default dim | 256 (code) vs 384 (docstring) | Config-dependent | ⚠️ Mismatch |

## Critical Issues Found

### 1. MLP State Dict Key Mismatch

```python
# ZFTurbo:
class MLP(nn.Module):
    def __init__(self, dim, dim_out, dim_hidden=None):
        self.net = nn.Sequential(nn.Linear(...), nn.Tanh(), nn.Linear(...))
# Keys: to_freqs.0.net.0.weight, to_freqs.0.net.2.weight

# Ours:
class MLP(nn.Sequential):
    def __init__(self, dim_in, dim_out, dim_hidden=None):
        super().__init__(nn.Sequential(...))  # EXTRA WRAPPER
# Keys: to_freqs.0.0.0.weight, to_freqs.0.0.2.weight
```

**Impact**: `load_state_dict()` will fail or silently load wrong weights.

### 2. MaskEstimator Output Dimension

```python
# ZFTurbo:
MLP(dim, freq * 2 * audio_channels, ...)  # 2 for real/imag only

# Ours:
MLP(dim, dim_out * complex_out, ...)
# Where complex_out = audio_channels * 2 * 2  # Extra *2 for stems!
```

**Impact**: Output tensor has wrong shape, masks won't apply correctly.

### 3. Mask Application Math

```python
# ZFTurbo (element-wise):
# Applies mask_real to stft_real, mask_imag to stft_imag separately
# NOT complex multiplication

# Ours (complex multiplication):
out_real = stft_real * mask_real - stft_imag * mask_imag
out_imag = stft_real * mask_imag + stft_imag * mask_real
```

**Impact**: Completely different mathematical operation, wrong separation output.

## Decision

### A. Fix MLP class structure

Remove the extra `nn.Sequential` wrapper to match ZFTurbo's key structure:

```python
class MLP(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden=None):
        super().__init__()
        hidden = dim_hidden or dim_in
        self.net = nn.Sequential(
            nn.Linear(dim_in, hidden),
            nn.Tanh(),
            nn.Linear(hidden, dim_out),
        )

    def forward(self, x):
        return self.net(x)
```

### B. Fix MaskEstimator output dimension

Change `complex_out` to NOT include stem packing:

```python
# Before:
complex_out=complex_as_channel * 2  # Wrong: packs 2 stems

# After:
complex_out=self.audio_channels * 2  # Correct: just real/imag per channel
```

### C. Fix mask application

Replace complex multiplication with element-wise application:

```python
# Before (complex multiplication):
out_real = stft_real * mask_real - stft_imag * mask_imag
out_imag = stft_real * mask_imag + stft_imag * mask_real

# After (element-wise):
out_real = stft_real * mask_real
out_imag = stft_imag * mask_imag
out_stft = torch.complex(out_real, out_imag)
```

### D. Fix forward pass stem handling

Update to properly handle `num_stems` mask estimators instead of hardcoding:

```python
# Each mask_estimator outputs (batch, freq, time, audio_channels * 2)
# Stack to (batch, num_stems, freq, time, audio_channels * 2)
# Reshape to (batch, num_stems, audio_channels, freq, time, 2)
```

### E. Verify and fix dim default

Check anvuew config.yaml and align DEFAULT_DIM with actual checkpoint value.

### F. Inference-only scope

Per user requirement, alignment focuses on inference compatibility only:
- No training code (MultiResolutionSTFTLoss not needed)
- No dataset utilities
- No CLI from upstream

## Consequences

### Positive

- Ensures weight loading works correctly with anvuew checkpoint
- Prevents silent inference quality degradation
- Documents architectural decisions for future maintenance

### Negative

- Requires significant refactoring of bs_roformer.py
- Existing smoke tests may need updates

## Alternatives Considered

### Keep current implementation
Rejected: Critical bugs in MLP keys, MaskEstimator output, and mask application math will cause incorrect inference.

### Align with lucidrains instead
Rejected: Checkpoint was trained with ZFTurbo codebase, not lucidrains.

## Tasks

See project.todo.json:
- T2-15: Fix MLP class structure
- T2-16: Fix MaskEstimator output dimension
- T2-17: Fix mask application math
- T2-18: Fix default dim and verify config
- T2-19: Add integration test with real checkpoint
