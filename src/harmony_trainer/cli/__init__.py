"""Harmony Trainer CLI entry point."""

import typer

from harmony_trainer.cli import songs

app = typer.Typer(
    name="harmony-trainer",
    help="Harmony Trainer CLI - Vocal harmony practice tool.",
    add_completion=False,
)

app.add_typer(songs.app, name="songs", help="Song registry commands")


@app.callback()
def main() -> None:
    """
    Harmony Trainer CLI main callback.
    """
    pass


if __name__ == "__main__":
    app()
