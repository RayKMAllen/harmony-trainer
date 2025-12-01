"""Harmony Trainer CLI entry point."""

import typer

app = typer.Typer(
    name="harmony-trainer",
    help="Harmony Trainer CLI - Vocal harmony practice tool.",
    add_completion=False,
)


@app.callback()
def main() -> None:
    """
    Harmony Trainer CLI main callback.
    """
    pass


if __name__ == "__main__":
    app()
