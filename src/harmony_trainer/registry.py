import json
from pathlib import Path
from uuid import UUID

from pydantic import BaseModel

from harmony_trainer.models import Song


class SongRegistryModel(BaseModel):
    songs: list[Song] = []


class SongRegistry:
    def __init__(self, data_path: Path = Path("data/songs.json")):
        self.data_path = data_path
        self._ensure_data_dir()
        self.model = self._load()

    def _ensure_data_dir(self) -> None:
        if not self.data_path.parent.exists():
            self.data_path.parent.mkdir(parents=True, exist_ok=True)

    def _load(self) -> SongRegistryModel:
        if not self.data_path.exists():
            return SongRegistryModel()

        try:
            with open(self.data_path, encoding="utf-8") as f:
                data = json.load(f)
                return SongRegistryModel(**data)
        except (json.JSONDecodeError, FileNotFoundError):
            return SongRegistryModel()

    def save(self) -> None:
        with open(self.data_path, "w", encoding="utf-8") as f:
            f.write(self.model.model_dump_json(indent=2))

    def add_song(self, song: Song) -> None:
        self.model.songs.append(song)
        self.save()

    def remove_song(self, song_id: UUID) -> bool:
        initial_count = len(self.model.songs)
        self.model.songs = [s for s in self.model.songs if s.id != song_id]
        if len(self.model.songs) < initial_count:
            self.save()
            return True
        return False

    def list_songs(self) -> list[Song]:
        return self.model.songs

    def get_song(self, song_id: UUID) -> Song | None:
        for song in self.model.songs:
            if song.id == song_id:
                return song
        return None
