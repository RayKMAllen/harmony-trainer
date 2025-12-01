# Harmony Trainer – Prototype One-Pager (Final, Stable Version)

## 1. Goal

Build a desktop prototype that can:

- Load a user-supplied audio file.
- Separate vocals vs accompaniment using Demucs.
- Attempt to split the vocal stem into lead and backing using two models.
- Extract pitch contours using torchcrepe per stem (monophonic).
- Let a singer practice the lead or backing part (when available) with real-time feedback.

**Primary validation question:**

Can backing vocal lines be extracted reliably enough to support harmony practice using real commercial music?

- If yes → expand to product.
- If no → adjust scope early.

## 2. Scope (v0)

### In Scope

| Component | Specification |
|-----------|---------------|
| Audio ingestion | Local files only - mp3/wav/flac |
| Song storage | data/songs.json + Typer CLI (add/list/remove) |
| Vocal separation | Demucs → vocals_all.wav + instrumental.wav |
| Lead/backing split | Two experimental paths: (1) svoice, (2) MedleyVox-based |
| Pitch extraction | torchcrepe, single-voice, run on available stems |
| Target generation | Always extract lead pitch line; extract backing only if a backing stem exists |
| UI | Local web interface for practice |
| Practice modes | Lead mode = always available; Backing mode = conditionally available |
| Data | Stored under: stems/, analysis/, practice/ |

### Out of Scope

- Streaming sources (Spotify/YouTube)
- Cloud inference/scaling
- Copyright/licensing handling
- AI-generated harmonies
- Multi-instrument or multi-backing-voice decomposition

## 3. Processing Flow

```
                ┌───────────────────────────────┐
Audio → Demucs →│ vocals_all.wav (always)       │→ instrumental.wav
                └───────────────┬───────────────┘
                                ↓
                     Optional stem separation
                        ┌─────────────┬─────────────┐
                        │ svoice      │ MedleyVox   │
                        ↓             ↓
                  lead.wav?       lead.wav?
                  backing.wav?    backing.wav?
    (These WAVs exist only if separation is successful)

                 PITCH EXTRACTION (torchcrepe)
                ┌─────────────────────────────────┐
 lead_line ← torchcrepe(vocals_all.wav)  (ALWAYS)
                └─────────────────────────────────┘
 backing_line ← torchcrepe(backing.wav)  (ONLY IF backing.wav exists)

analysis/{song}/targets.json → {
  lead:    [...notes...],
  backing: [...notes...] OR [],
  has_backing: true|false
}
```

**Key guarantee:**
- Lead training line always exists.
- Backing training line is optional.

## 4. Splitter Evaluation Harness

**Purpose:** measure how often backing extraction is viable.

- test set: ~10–20 diverse songs registered in data/songs.json
- script: eval/run_eval.py

```
runs:
for song in songs:
    demucs → vocals_all
    try svoice → stems
    try medleyvox → stems
    torchcrepe(vocals_all)      → lead_line.json  (always)
    if backing.wav exists:
         torchcrepe(backing.wav) → backing_line.json
```

Stores results under:
```
eval/{splitter}/{song_id}/[stems + notes + manifest.json]
```

**Goal threshold to greenlight product:**
≥30–40% songs produce usable backing contour.

## 5. Real-time Practice UI

**Visual timeline:**
- Target note blocks (lead or backing)
- Live pitch trace from mic
- Cents-off deviation

**Controls:**
- Looping, slow-down, mute/solo, instrument-only playback

**Modes:**
- LEAD (always enabled)
- BACKING (only if has_backing=true)

If backing fails cleanly: no error, just no backing option shown.

## 6. Success Criteria

| Requirement | Pass If |
|-------------|---------|
| Lead extraction | Works for majority of songs, stable pitch line |
| Backing extraction | Usable contour for ≥30–40% test tracks |
| Feedback accuracy | ±20–30 cents typical live deviation |
| UX | First-time singer tests harmony in <5 min |

If backing quality falls below target → product direction shifts to lead-training first.

## 7. Future Expansion (Post-Prototype)

- Cloud GPU inference for heavy separation
- Mobile app clients
- Automatic harmony generation (3rd/5th/7th)
- Licensing+integration w/ streaming catalogs
- Multi-harmony visualisation + duet/choir mode
