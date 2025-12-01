import json
from pathlib import Path

from typer.testing import CliRunner

from harmony_trainer.cli import app
from harmony_trainer.registry import SongRegistry

runner = CliRunner()


def test_song_crud_lifecycle(tmp_path: Path) -> None:
    """
    Test the full lifecycle of adding, listing, and removing a song via CLI.
    Uses a temporary directory for the song registry to avoid polluting the actual data.
    """
    # Setup: Configure registry to use tmp_path
    songs_file = tmp_path / "songs.json"
    registry = SongRegistry(data_path=songs_file)
    # Patch the registry used by the CLI commands implicitly by modifying the global instance
    # Note: In a real scenario, dependency injection would be cleaner, but for this CLI structure,
    # we'll rely on the fact that the CLI commands import 'registry' from 'cli.songs'.
    # We need to mock that registry instance or temporarily point it to our test file.

    # Actually, the cleaner way given the current implementation is to use 'mock.patch'
    # or to allow configuring the registry path via env var/argument.
    # For simplicity and speed, let's just patch the module-level registry instance.
    from harmony_trainer.cli import songs

    original_registry = songs.registry
    songs.registry = registry

    try:
        # 1. Create a dummy audio file
        audio_file = tmp_path / "test_audio.mp3"
        audio_file.touch()

        # 2. Add song
        result = runner.invoke(
            app,
            ["songs", "add", str(audio_file), "--title", "Test Song", "--artist", "Test Artist"],
        )
        assert result.exit_code == 0
        assert "Added song" in result.stdout
        assert "Test Song" in result.stdout

        # Verify file content
        assert songs_file.exists()
        with open(songs_file) as f:
            data = json.load(f)
            assert len(data["songs"]) == 1
            song = data["songs"][0]
            assert song["title"] == "Test Song"
            assert song["artist"] == "Test Artist"
            song_id = song["id"]

        # 3. List songs
        result = runner.invoke(app, ["songs", "list"])
        assert result.exit_code == 0
        assert "Test Song" in result.stdout
        assert "Test Artist" in result.stdout
        assert song_id in result.stdout

        # 4. Remove song
        result = runner.invoke(app, ["songs", "remove", song_id])
        assert result.exit_code == 0
        assert f"Removed song: {song_id}" in result.stdout

        # Verify removal
        with open(songs_file) as f:
            data = json.load(f)
            assert len(data["songs"]) == 0

        # 5. List empty
        result = runner.invoke(app, ["songs", "list"])
        assert "No songs registered" in result.stdout

    finally:
        # Restore original registry
        songs.registry = original_registry
