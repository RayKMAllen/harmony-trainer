# ADR 0002: Automatic Metadata Extraction

## Status
Accepted

## Date
2024-12-01

## Context
When adding songs to the registry via the CLI, users are currently required to manually provide the title and artist as arguments. Since most audio files (MP3, FLAC, etc.) already contain this information in their ID3 tags or other metadata formats, this manual step creates unnecessary friction.

## Decision
We will automate the extraction of song metadata (title and artist) from the source audio file.

### Implementation Details
- **Library**: We will use `tinytag` for reading metadata. It is a lightweight, pure-Python library with no external dependencies (unlike `mutagen` or wrappers around C libraries), making it easy to install and maintain across platforms.
- **CLI Changes**:
    - The `--title` and `--artist` arguments in the `add` command will become optional.
    - If provided, they take precedence over the file metadata.
    - If not provided, the system attempts to read them from the file.
    - If metadata is missing and no argument is provided, the system falls back to using the filename (without extension) as the title.
- **Dependencies**: Add `tinytag` to project dependencies.

## Consequences
- **User Experience**: Simplifies the process of adding songs.
- **Dependencies**: Adds a small dependency (`tinytag`).
- **Resilience**: The system must handle cases where metadata is corrupt or missing gracefully (fallback to filename).

## Alternatives Considered
- **mutagen**: More comprehensive but heavier and more complex API.
- **librosa/torchaudio**: These are already in the stack but are primarily for signal processing, not metadata extraction.
