import os
import argparse
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    precision_score,
    recall_score,
    f1_score
)

from data.stft_loader import preprocess_gdf_folder
from models.stft_hhatnet import STFTHHATNet


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model, loader, device):
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)

            outputs = model(X_batch)

            preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.numpy())

    acc = accuracy_score(all_labels, all_preds)

    return acc, all_preds, all_labels


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--out", type=str, default="results/stft_hhat_mixed_leakage_free.csv")

    args = parser.parse_args()

    set_seed(42)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Using device:", device)

    os.makedirs("results", exist_ok=True)

    X, y, subject_ids, trial_ids = preprocess_gdf_folder(
        args.data_path
    )

    print("Total samples:", X.shape)
    print("Total unique trials:", len(np.unique(trial_ids)))
    print("Class distribution:", np.bincount(y))

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=42
    )

    train_idx, test_idx = next(
        splitter.split(
            X,
            y,
            groups=trial_ids
        )
    )

    X_train = X[train_idx]
    X_test = X[test_idx]

    y_train = y[train_idx]
    y_test = y[test_idx]

    train_trials = trial_ids[train_idx]
    test_trials = trial_ids[test_idx]

    print("Train samples:", X_train.shape)
    print("Test samples:", X_test.shape)

    print("Train trials:", len(np.unique(train_trials)))
    print("Test trials:", len(np.unique(test_trials)))

    overlap = set(np.unique(train_trials)).intersection(
        set(np.unique(test_trials))
    )

    print("Trial overlap:", len(overlap))

    if len(overlap) != 0:
        raise ValueError("Data leakage detected: same trial in train and test")

    print("Train class distribution:", np.bincount(y_train))
    print("Test class distribution:", np.bincount(y_test))

    X_train = torch.tensor(
        X_train,
        dtype=torch.float32
    ).permute(0, 3, 1, 2)

    X_test = torch.tensor(
        X_test,
        dtype=torch.float32
    ).permute(0, 3, 1, 2)

    y_train = torch.tensor(
        y_train,
        dtype=torch.long
    )

    y_test = torch.tensor(
        y_test,
        dtype=torch.long
    )

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
        sampler=sampler,
        num_workers=2,
        pin_memory=True
    )

    test_loader = DataLoader(
        TensorDataset(X_test, y_test),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    model = STFTHHATNet(
        n_classes=4
    ).to(device)

    criterion = nn.CrossEntropyLoss(
        label_smoothing=0.03
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.02
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs
    )

    best_acc = 0.0
    best_state = None

    patience = 20
    patience_counter = 0

    for epoch in range(args.epochs):
        model.train()

        total_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            if epoch < 50:
                X_batch = X_batch + torch.randn_like(X_batch) * 0.005

            optimizer.zero_grad()

            outputs = model(X_batch)

            loss = criterion(outputs, y_batch)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=0.5
            )

            optimizer.step()

            total_loss += loss.item()

        scheduler.step()

        if (epoch + 1) % 5 == 0:
            val_acc, _, _ = evaluate(
                model,
                test_loader,
                device
            )

            if val_acc > best_acc:
                best_acc = val_acc

                best_state = {
                    k: v.cpu().clone()
                    for k, v in model.state_dict().items()
                }

                patience_counter = 0

            else:
                patience_counter += 1

            print(
                f"Epoch {epoch + 1}/{args.epochs} | "
                f"Loss: {total_loss / len(train_loader):.4f} | "
                f"Val Acc: {val_acc:.4f} | "
                f"Best: {best_acc:.4f}"
            )

            if patience_counter >= patience:
                print("Early stopping triggered")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.to(device)

    final_acc, all_preds, all_labels = evaluate(
        model,
        test_loader,
        device
    )

    results = {
        "accuracy": final_acc,
        "kappa": cohen_kappa_score(
            all_labels,
            all_preds
        ),
        "precision": precision_score(
            all_labels,
            all_preds,
            average="macro",
            zero_division=0
        ),
        "recall": recall_score(
            all_labels,
            all_preds,
            average="macro",
            zero_division=0
        ),
        "f1": f1_score(
            all_labels,
            all_preds,
            average="macro",
            zero_division=0
        )
    }

    print("\nMixed Leakage-Free STFT-HHAT Results:")
    print(results)

    pd.DataFrame([results]).to_csv(
        args.out,
        index=False
    )

    print("Saved to:", args.out)


if __name__ == "__main__":
    main()
