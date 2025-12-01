# Harmony Trainer

A tool for training vocal harmony models.

## Installation

```bash
uv sync --all-extras
```

## Usage

### Song Management

Add a song:
```bash
harmony-trainer songs add [PATH] --title [TITLE] [--artist [ARTIST]]
```
*   `PATH`: Path to the audio file.
*   `--title` / `-t`: Song title.
*   `--artist` / `-a`: Artist name (optional).

List songs:
```bash
harmony-trainer songs list
```

Remove a song:
```bash
harmony-trainer songs remove [SONG_ID]
```

## Development

```bash
uv run pre-commit install
uv run pytest
