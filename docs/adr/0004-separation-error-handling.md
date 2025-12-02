# ADR 0004: Separation Error Handling

## Status
Accepted

## Date
2025-12-02

## Context

The project uses multiple separation backends:
- Demucs: vocal / instrumental
- SVoice: lead / backing vocals
- MedleyVox: lead / backing vocals

Early wrappers for SVoice and MedleyVox:
- Returned `(Path | None, Path | None)`
- Silently swallowed loader and inference errors
- Forced callers to special-case each backend

Demucs already raises on failure. We want one clear contract for all separation backends that:
- Uses a typed error instead of sentinel return values
- Behaves sensibly in both prototyping and production

## Decision

### A. Single domain exception

- Use `SeparationError` as the only public error type for separation failures.
- It lives in `harmony_trainer.separation` and is re-exported via the package `__all__`.
- Backend- or loader-specific exceptions are wrapped in `SeparationError` with a short message.

### B. No `None` in separation results

Public separation entry points always return concrete paths on success and never use `None` to signal failure:
- `DemucsWrapper.separate` → `tuple[Path, Path]`, raises `SeparationError` on failure.
- `SVoiceWrapper.separate` → `tuple[Path, Path]`, raises `SeparationError` on failure.
- `MedleyVoxWrapper.separate` → `tuple[Path, Path]`, raises `SeparationError` on failure.
- `separate_audio`, `separate_vocals`, and `separate_vocals_medleyvox` just delegate and propagate `SeparationError`.

Existing `(Path | None, Path | None)` return types are deprecated and should be removed as code is updated.

### C. Where `SeparationError` is handled

`SeparationError` is only handled at clear boundaries:
- CLI commands and any future web or API layers.
- Batch or evaluation drivers.
- High-level orchestration that can choose fallbacks or partial success.

Within the separation package and its helpers, `SeparationError` is allowed to bubble unchanged.

### D. Prototyping semantics

Default behavior in development is fail-fast:
- Uncaught `SeparationError` terminates the command with a non-zero exit code.
- CLI prints a concise message and optional traceback in verbose or debug mode.
- No automatic fallback between separation backends unless a specific experiment flow wants it.

### E. Production semantics

Default behavior in production is best-effort with clear reporting:
- Per song:
  - On `SeparationError`, log song id, backend, and message.
  - Optionally try configured fallback backends.
  - If all backends fail, mark the song as "no usable stems"; downstream stages that do not require stems may still run.
- Batch jobs:
  - Continue processing remaining songs even if some fail.
  - Emit a summary of separation failures by song id and backend at the end.
- CLI strictness:
  - A `--strict` (or equivalent) flag treats any `SeparationError` as fatal and returns a non-zero exit code.
  - Without strict mode, commands may succeed overall while reporting per-song warnings.
- Logging:
  - Use structured fields (song id, backend, error type, message).
  - Full stack traces are reserved for debug or verbose modes.

### F. Scope

This ADR applies to all separation backends in `harmony_trainer.separation`, at minimum:
- Demucs wrappers.
- SVoice wrappers.
- MedleyVox wrappers.

It does not mandate a specific logging framework or exact CLI flag names; it defines only the behavior around `SeparationError`.

## Consequences

### Positive

- Single, explicit failure channel for all separation backends.
- Consistent behavior between Demucs, SVoice, and MedleyVox.
- Easier debugging in development via fail-fast defaults.
- Production code can combine fallbacks and partial success with clear reporting.
- Callers no longer inspect `None` values to detect errors.

### Negative

- Existing call sites that expect `(Path | None, Path | None)` must be updated.
- Some failures that were previously silent will now surface as explicit errors and require handling.

### Alternatives considered

- Keep the lenient API that returns `(None, None)`:
  - Rejected: hides root causes and scatters error handling across callers.
- Expose separate exception types per backend:
  - Rejected as a default: typical callers only need one domain error; backend detail can still be attached inside `SeparationError` if needed.
