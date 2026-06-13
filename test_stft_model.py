import torch
from models.stft_cnn import STFTCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = STFTCNN(n_classes=4).to(device)

x = torch.randn(2, 3, 120, 32).to(device)

y = model(x)

print("Output shape:", y.shape)

total_params = sum(p.numel() for p in model.parameters())

print("Total parameters:", total_params)