import os
import argparse
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, cohen_kappa_score, precision_score, recall_score, f1_score

from data.stft_loader import preprocess_gdf_folder


class ImprovedSTFTCNN(nn.Module):
    def __init__(self, n_classes=4):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.3),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ELU(),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Dropout(0.4),
            nn.Linear(64, n_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--out", type=str, default="results/stft_improved_results.csv")

    args = parser.parse_args()

    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    os.makedirs("results", exist_ok=True)

    X, y, subject_ids = preprocess_gdf_folder(args.data_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    X_train = torch.tensor(X_train, dtype=torch.float32).permute(0, 3, 1, 2)
    X_test = torch.tensor(X_test, dtype=torch.float32).permute(0, 3, 1, 2)

    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    class_counts = torch.bincount(y_train)
    class_weights = 1.0 / class_counts.float()
    sample_weights = class_weights[y_train]

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=args.batch_size,
        sampler=sampler
    )

    test_loader = DataLoader(
        TensorDataset(X_test, y_test),
        batch_size=args.batch_size,
        shuffle=False
    )

    model = ImprovedSTFTCNN(n_classes=4).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.005
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs
    )

    best_acc = 0
    best_state = None

    for epoch in range(args.epochs):
        model.train()

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            noise = torch.randn_like(X_batch) * 0.01
            X_batch = X_batch + noise

            optimizer.zero_grad()

            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

        scheduler.step()

        if (epoch + 1) % 10 == 0:
            model.eval()

            preds_all = []
            labels_all = []

            with torch.no_grad():
                for X_batch, y_batch in test_loader:
                    X_batch = X_batch.to(device)

                    outputs = model(X_batch)
                    preds = outputs.argmax(dim=1)

                    preds_all.extend(preds.cpu().numpy())
                    labels_all.extend(y_batch.numpy())

            acc = accuracy_score(labels_all, preds_all)

            if acc > best_acc:
                best_acc = acc
                best_state = model.state_dict()

            print(f"Epoch {epoch+1}/{args.epochs} | Val Acc: {acc:.4f} | Best: {best_acc:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)

            outputs = model(X_batch)
            preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.numpy())

    results = {
        "accuracy": accuracy_score(all_labels, all_preds),
        "kappa": cohen_kappa_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, average="macro", zero_division=0),
        "recall": recall_score(all_labels, all_preds, average="macro", zero_division=0),
        "f1": f1_score(all_labels, all_preds, average="macro", zero_division=0)
    }

    print("\nFinal Improved STFT-CNN Results:")
    print(results)

    pd.DataFrame([results]).to_csv(args.out, index=False)
    print("Saved to:", args.out)


if __name__ == "__main__":
    main()
