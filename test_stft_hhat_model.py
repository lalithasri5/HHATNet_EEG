import torch
from models.stft_hhatnet import STFTHHATNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = STFTHHATNet(n_classes=4).to(device)

x = torch.randn(2, 3, 120, 32).to(device)

y = model(x)

print("Output shape:", y.shape)
print("Total parameters:", sum(p.numel() for p in model.parameters()))