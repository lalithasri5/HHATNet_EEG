import torch
from models.hhatnet import HHATNet


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = HHATNet(
        n_channels=22,
        n_classes=4,
        n_bands=5
    ).to(device)

    x = torch.randn(2, 5, 22, 1001).to(device)

    y = model(x)

    total_params = sum(p.numel() for p in model.parameters())

    print("Output shape:", y.shape)
    print("Total parameters:", total_params)


if __name__ == "__main__":
    main()