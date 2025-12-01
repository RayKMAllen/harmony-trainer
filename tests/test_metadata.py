from pathlib import Path
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from harmony_trainer.cli.songs import app
from harmony_trainer.registry import SongRegistry

runner = CliRunner()


def test_add_song_auto_metadata(tmp_path: Path) -> None:
    """Test adding a song with automatic metadata extraction."""
    # Create a dummy audio file
    audio_file = tmp_path / "song_with_metadata.mp3"
    audio_file.write_bytes(b"dummy content")

    # Mock tinytag to return metadata
    with patch("harmony_trainer.cli.songs.TinyTag") as mock_tinytag:
        mock_tag = MagicMock()
        mock_tag.title = "Extracted Title"
        mock_tag.artist = "Extracted Artist"
        mock_tinytag.get.return_value = mock_tag

        # Use a temporary registry file
        registry_file = tmp_path / "songs.json"
        with patch("harmony_trainer.cli.songs.registry", SongRegistry(registry_file)):
            result = runner.invoke(app, ["add", str(audio_file)])

            assert result.exit_code == 0
            assert "Added song" in result.stdout
            assert "Extracted Title" in result.stdout

            # Verify the song was saved with extracted metadata
            registry = SongRegistry(registry_file)
            songs = registry.list_songs()
            assert len(songs) == 1
            assert songs[0].title == "Extracted Title"
            assert songs[0].artist == "Extracted Artist"


def test_add_song_fallback_filename(tmp_path: Path) -> None:
    """Test adding a song where metadata is missing, falling back to filename."""
    # Create a dummy audio file
    filename = "unknown_song.mp3"
    audio_file = tmp_path / filename
    audio_file.write_bytes(b"dummy content")

    # Mock tinytag to return no metadata
    with patch("harmony_trainer.cli.songs.TinyTag") as mock_tinytag:
        mock_tag = MagicMock()
        mock_tag.title = None
        mock_tag.artist = None
        mock_tinytag.get.return_value = mock_tag

        # Use a temporary registry file
        registry_file = tmp_path / "songs.json"
        with patch("harmony_trainer.cli.songs.registry", SongRegistry(registry_file)):
            result = runner.invoke(app, ["add", str(audio_file)])

            assert result.exit_code == 0
            assert "Added song" in result.stdout
            assert "unknown_song" in result.stdout

            # Verify the song was saved with filename as title
            registry = SongRegistry(registry_file)
            songs = registry.list_songs()
            assert len(songs) == 1
            assert songs[0].title == "unknown_song"
            assert songs[0].artist is None


def test_add_song_manual_override(tmp_path: Path) -> None:
    """Test that manual arguments override extracted metadata."""
    # Create a dummy audio file
    audio_file = tmp_path / "override_test.mp3"
    audio_file.write_bytes(b"dummy content")

    # Mock tinytag to return metadata (which should be ignored)
    with patch("harmony_trainer.cli.songs.TinyTag") as mock_tinytag:
        mock_tag = MagicMock()
        mock_tag.title = "Ignored Title"
        mock_tag.artist = "Ignored Artist"
        mock_tinytag.get.return_value = mock_tag

        # Use a temporary registry file
        registry_file = tmp_path / "songs.json"
        with patch("harmony_trainer.cli.songs.registry", SongRegistry(registry_file)):
            result = runner.invoke(
                app,
                ["add", str(audio_file), "--title", "Manual Title", "--artist", "Manual Artist"],
            )

            assert result.exit_code == 0

            # Verify the song was saved with manual metadata
            registry = SongRegistry(registry_file)
            songs = registry.list_songs()
            assert len(songs) == 1
            assert songs[0].title == "Manual Title"
            assert songs[0].artist == "Manual Artist"
