# %%

import os
from pathlib import Path
from uuid import UUID

from harmony_trainer.separation.demucs import DemucsWrapper

# move to the project root
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print(os.getcwd())


# %%

audio_path = Path(r"C:\Users\WIN10\Music\Scar Tissue.mp3")
song_id = UUID("328f1360-bfb1-436b-96fc-208d101fc2fe")
data_dir = Path("data")
model_name = "htdemucs"


wrapper = DemucsWrapper(model_name=model_name)
output_dir = data_dir / "stems" / str(song_id)
vocals_path, instrumental_path = wrapper.separate(audio_path, output_dir)

# %%
