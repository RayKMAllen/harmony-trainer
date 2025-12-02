# %%
# ruff: noqa: E402

import os
from pathlib import Path
from uuid import UUID

import numpy as np

from harmony_trainer.pitch.extractor import PitchExtractor
from harmony_trainer.separation.demucs import DemucsWrapper

# move to the project root
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print(os.getcwd())


# %%

scartissue_uuid = UUID("328f1360-bfb1-436b-96fc-208d101fc2fe")

audio_path = Path(r"C:\Users\WIN10\Music\Scar Tissue.mp3")
song_id = scartissue_uuid
data_dir = Path("data")
model_name = "htdemucs"


wrapper = DemucsWrapper(model_name=model_name)
output_dir = data_dir / "stems" / str(song_id)
vocals_path, instrumental_path = wrapper.separate(audio_path, output_dir)

# %%

song_id = scartissue_uuid
data_dir = Path("data")
model = "full"

extractor = PitchExtractor(model=model)
contour = extractor.extract(vocals_path)


output_dir = data_dir / "analysis" / str(song_id)
output_dir.mkdir(parents=True, exist_ok=True)

# Save pitch data as npz
np.savez(
    output_dir / "pitch.npz",
    times=contour.times,
    frequencies=contour.frequencies,
    confidence=contour.confidence,
    sample_rate=contour.sample_rate,
)
