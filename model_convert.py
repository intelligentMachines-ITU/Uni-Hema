import torch

ckpt = torch.load("Model/checkpoint0023.pth", map_location="cpu")

# handle different formats
if isinstance(ckpt, dict) and "model" in ckpt:
    state_dict = ckpt["model"]
else:
    state_dict = ckpt

torch.save(state_dict, "Model/pytorch_model.bin")