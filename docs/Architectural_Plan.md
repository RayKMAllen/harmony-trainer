# Harmony Trainer - Architectural Plan

## Overview & Scope

Desktop prototype for vocal harmony practice. Users load audio files, the system separates vocals and extracts pitch contours, then provides real-time feedback during practice sessions.

**Primary validation:** Can backing vocal lines be extracted reliably enough (≥30-40% success rate) to support harmony practice?

## Goals & Non-Goals

### Goals
- Load local audio files (mp3/wav/flac)
- Separate vocals from accompaniment using Demucs
- Attempt lead/backing vocal split via svoice and MedleyVox
- Extract pitch contours using torchcrepe
- Provide real-time practice UI with pitch feedback
- Evaluate backing extraction viability across test songs

### Non-Goals
- Streaming sources (Spotify/YouTube)
- Cloud inference/scaling
- Copyright/licensing handling
- AI-generated harmonies
- Multi-instrument decomposition

## Domain & Key Entities

| Entity | Description |
|--------|-------------|
| Song | User-registered audio file with metadata |
| Stem | Separated audio track (vocals_all, instrumental, lead, backing) |
| PitchContour | Time-series of pitch values extracted from a stem |
| Target | Practice target (lead or backing pitch line) |
| PracticeSession | Real-time session comparing user pitch to target |

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        CLI Layer (Typer)                        │
│                   song add/list/remove commands                 │
└─────────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────────┐
│                      Core Processing Pipeline                    │
├─────────────────┬─────────────────┬─────────────────────────────┤
│   Separation    │   Pitch Extract │      Analysis               │
│   (Demucs,      │   (torchcrepe)  │   (target generation)       │
│   svoice,       │                 │                             │
│   MedleyVox)    │                 │                             │
└─────────────────┴─────────────────┴─────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────────┐
│                      Practice UI (Web)                          │
│         Timeline, pitch visualization, real-time feedback       │
└─────────────────────────────────────────────────────────────────┘
```

### Module Structure

```
src/harmony_trainer/
├── __init__.py
├── cli/                    # Typer CLI commands
│   ├── __init__.py
│   └── songs.py           # add/list/remove song commands
├── separation/            # Audio separation models
│   ├── __init__.py
│   ├── demucs.py         # Demucs wrapper
│   ├── svoice.py         # svoice wrapper
│   └── medleyvox.py      # MedleyVox wrapper
├── pitch/                 # Pitch extraction
│   ├── __init__.py
│   └── extractor.py      # torchcrepe wrapper
├── analysis/              # Target generation
│   ├── __init__.py
│   └── targets.py        # Generate practice targets
├── practice/              # Practice session logic
│   ├── __init__.py
│   └── session.py        # Real-time pitch comparison
├── ui/                    # Web UI
│   ├── __init__.py
│   └── app.py            # FastAPI/Flask app
└── models.py              # Pydantic models for Song, Stem, etc.
```

## Data & Persistence

### File Structure
```
data/
├── songs.json             # Song registry
├── stems/                 # Separated audio files
│   └── {song_id}/
│       ├── vocals_all.wav
│       ├── instrumental.wav
│       ├── lead.wav       # Optional
│       └── backing.wav    # Optional
├── analysis/              # Extracted pitch data
│   └── {song_id}/
│       └── targets.json   # {lead: [...], backing: [...], has_backing: bool}
└── practice/              # Practice session data
    └── {song_id}/
        └── sessions/
```

### songs.json Schema
```json
{
  "songs": [
    {
      "id": "uuid",
      "title": "Song Title",
      "artist": "Artist Name",
      "source_path": "/path/to/original.mp3",
      "added_at": "2024-01-01T00:00:00Z",
      "processed": true,
      "has_backing": false
    }
  ]
}
```

## External Dependencies

| Dependency | Purpose |
|------------|---------|
| Demucs | Vocal/instrumental separation |
| svoice | Lead/backing separation (experimental) |
| MedleyVox | Lead/backing separation (experimental) |
| torchcrepe | Pitch extraction |
| PyTorch | ML model runtime |
| Typer | CLI framework |
| FastAPI/Flask | Web UI backend |

## Risks & Open Questions

1. **Backing extraction quality** - Primary risk. May not achieve 30-40% success rate.
2. **Model compatibility** - svoice and MedleyVox may have conflicting dependencies.
3. **Real-time latency** - Pitch detection must be fast enough for live feedback.
4. **GPU requirements** - Demucs and other models may require GPU for reasonable performance.

## Milestones & Phases

### M1: Foundation
- Project setup, tooling, CI
- Basic CLI skeleton
- Song registry (add/list/remove)

### M2: Core Features
- Demucs integration for vocal separation
- svoice integration for lead/backing split
- MedleyVox integration as alternative splitter
- torchcrepe pitch extraction
- Target generation pipeline

### M3: Hardening
- Evaluation harness for splitter comparison
- Practice UI implementation
- Real-time pitch feedback
- Documentation and testing
