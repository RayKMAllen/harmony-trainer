from datetime import UTC, datetime
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class Song(BaseModel):
    """
    Represents a user-registered audio file with metadata.
    """

    id: UUID = Field(default_factory=uuid4)
    title: str
    artist: str | None = None
    source_path: str
    added_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    processed: bool = False
    has_backing: bool = False
