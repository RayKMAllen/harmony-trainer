from pathlib import Path
from typing import Optional
from uuid import UUID

import typer
from rich.console import Console
from rich.table import Table

from harmony_trainer.models import Song
from harmony_trainer.registry import SongRegistry

app = typer.Typer(name="songs", help="Manage song registry")
console = Console()
registry = SongRegistry()


@app.command()
def add(
    path: Path = typer.Argument(..., help="Path to audio file", exists=True),  # noqa: B008
    title: str = typer.Option(..., "--title", "-t", help="Song title"),
    artist: Optional[str] = typer.Option(None, "--artist", "-a", help="Artist name"),  # noqa: UP007
) -> None:
    """Add a new song to the registry."""
    song = Song(
        title=title,
        artist=artist,
        source_path=str(path.absolute()),
    )
    registry.add_song(song)
    console.print(f"[green]Added song:[/green] {song.title} ({song.id})")


@app.command("list")
def list_songs() -> None:
    """List all registered songs."""
    songs = registry.list_songs()
    if not songs:
        console.print("[yellow]No songs registered.[/yellow]")
        return

    table = Table(title="Registered Songs")
    table.add_column("ID", justify="left", style="cyan", no_wrap=True)
    table.add_column("Title", style="magenta")
    table.add_column("Artist", style="green")
    table.add_column("Processed", justify="center")

    for song in songs:
        table.add_row(
            str(song.id),
            song.title,
            song.artist or "-",
            "✅" if song.processed else "❌",
        )
    console.print(table)


@app.command()
def remove(
    song_id: UUID = typer.Argument(..., help="ID of song to remove"),  # noqa: B008
) -> None:
    """Remove a song from the registry."""
    if registry.remove_song(song_id):
        console.print(f"[green]Removed song:[/green] {song_id}")
    else:
        console.print(f"[red]Song not found:[/red] {song_id}")
