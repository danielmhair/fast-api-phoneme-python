import torch
model = torch.compile(model, mode="reduce-overhead")
