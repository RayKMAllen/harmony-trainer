# Harmony Trainer

Vocal harmony practice tool, early days. The idea: separate a song into band,
lead vocal and backing vocals, extract the pitch of each line, then score what
you sing against the backing line. Python 3.11, PyTorch on CPU, Typer CLI.

What works so far: the song registry (`songs add` reads the tags from the
file), Demucs vocals/instrumental separation, BS-RoFormer and MedleyVox
wrappers for the lead/backing split (checkpoints via `BS_ROFORMER_WEIGHTS_PATH`
/ `MEDLEYVOX_WEIGHTS_PATH`, else Hugging Face), torchcrepe pitch extraction per
stem. Design notes in `docs/adr/`.

    uv sync --all-extras
    uv run harmony-trainer songs add song.mp3
    uv run harmony-trainer songs list
    uv run pytest

## Next

`project.todo.json`: notes and target lines from the pitch contours (T2-05),
one `process` command for the whole chain (T2-06), then the practice session
and its UI (T3).

## Licence

MIT, see `LICENSE`. Vendored model code: `models/bs_roformer.py` from
ZFTurbo/Music-Source-Separation-Training (MIT), `models/medleyvox.py` from
Cyru5/MedleyVox (CC BY 4.0).
