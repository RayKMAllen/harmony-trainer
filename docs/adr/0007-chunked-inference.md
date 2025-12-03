# 7. Chunked Inference for BS-RoFormer

Date: 2025-12-03

## Status

Accepted

## Context

The current `BSRoFormerWrapper` processes the entire audio waveform in a single `model()` call. While simple, this approach has limitations:
1.  **No Progress Feedback**: The operation is blocking and opaque, leaving the user unsure if the process is hung or working, especially for long songs.
2.  **Memory Constraints**: Processing a full 5-minute song at 44.1kHz stereo requires significant GPU memory (VRAM). On consumer hardware, this can lead to Out-Of-Memory (OOM) errors.

## Decision

We will refactor the `BSRoFormerWrapper.separate` method to use **chunked inference with overlap-add reconstruction**.

1.  **Chunking Strategy**: The input audio will be split into fixed-size segments (default: 30 seconds). This duration is chosen to balance memory usage with model context requirements; Transformers benefit from longer contexts, and excessively short chunks would increase the relative overhead of the overlap regions.
2.  **Overlap**: Segments will overlap (e.g., by 2-4 seconds) to prevent boundary artifacts.
3.  **Cross-fading**: Overlapping regions will be blended using a linear cross-fade or windowing function to ensure smooth transitions.
4.  **Progress Reporting**: The wrapper will calculate the total number of chunks and log/print progress after each chunk is processed.

## Consequences

### Positive
- **Better UX**: Users get immediate feedback on progress (e.g., "Processing chunk 1/10... 10%").
- **Lower Memory Footprint**: Peak memory usage becomes a function of chunk size, not total song length, enabling processing of arbitrarily long files on modest hardware.
- **Robustness**: Reduces the risk of crashes on long inputs.

### Negative
- **Complexity**: The implementation of correct overlap-add logic is more complex than a simple forward pass.
- **Performance Overhead**: Slight processing overhead for windowing, stitching, and repeated processing of overlapping regions.

## Alternatives Considered

- **Streaming Inference**: Modifying the model architecture to support true streaming state. This is too invasive for a vendored model.
- **Simple Cutting**: cutting without overlap. This introduces audible "clicks" or discontinuities at boundaries, which is unacceptable for audio separation.
