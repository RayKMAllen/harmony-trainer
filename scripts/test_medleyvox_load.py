"""Test MedleyVox model loading and forward pass."""

import torch

from harmony_trainer.models.medleyvox import MedleyVoxNet, extract_online_model_state_dict

ckpt_path = r"C:\Users\WIN10\.cache\huggingface\hub\models--Cyru5--MedleyVox\snapshots\5c9e4e0d909e5a006c992b3422901ed416f4e57f\vocals 100.pth"
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
sd = extract_online_model_state_dict(ckpt)
model = MedleyVoxNet()
model.load_state_dict(sd)
model.eval()
x = torch.randn(1, 16000)
with torch.no_grad():
    y = model(x)
print("Forward pass with weights:", y.shape)
