from pathlib import Path
from uuid import uuid4

from harmony_trainer.models import Song
from harmony_trainer.registry import SongRegistry


def test_registry_initialization_creates_base_dirs(tmp_path: Path) -> None:
    """Test that initializing the registry creates the base data directories."""
    songs_file = tmp_path / "songs.json"
    SongRegistry(data_path=songs_file)

    assert (tmp_path / "stems").exists()
    assert (tmp_path / "stems").is_dir()
    assert (tmp_path / "analysis").exists()
    assert (tmp_path / "analysis").is_dir()
    assert (tmp_path / "practice").exists()
    assert (tmp_path / "practice").is_dir()


def test_add_song_creates_subdirectories(tmp_path: Path) -> None:
    """Test that adding a song creates the per-song subdirectories."""
    songs_file = tmp_path / "songs.json"
    registry = SongRegistry(data_path=songs_file)

    song = Song(
        id=uuid4(),
        title="Test Song",
        artist="Test Artist",
        source_path="/tmp/test.mp3",
    )
    registry.add_song(song)

    song_id_str = str(song.id)

    # Check stems directory
    stems_dir = tmp_path / "stems" / song_id_str
    assert stems_dir.exists()
    assert stems_dir.is_dir()

    # Check analysis directory
    analysis_dir = tmp_path / "analysis" / song_id_str
    assert analysis_dir.exists()
    assert analysis_dir.is_dir()

    # Check practice directory structure
    practice_dir = tmp_path / "practice" / song_id_str / "sessions"
    assert practice_dir.exists()
    assert practice_dir.is_dir()
