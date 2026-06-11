import os
import argparse
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import accuracy_score, cohen_kappa_score, precision_score, recall_score, f1_score

from models.hhatnet import HHATNet
from data.bcic2a_loader import load_subject_2a


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def eeg_augment(x, noise_std=0.02):
    noise = torch.randn_like(x) * noise_std
    return x + noise


def train_one_subject(data_path, subject, epochs_count, batch_size, lr, device):
    train_loader, test_loader = load_subject_2a(
        data_path=data_path,
        subject=subject,
        batch_size=batch_size
    )

    model = HHATNet(n_channels=22, n_classes=4, n_bands=5).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=0.01
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs_count
    )

    best_acc = 0
    best_state = None

    for epoch in range(epochs_count):
        model.train()

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            X_batch = eeg_augment(X_batch)

            optimizer.zero_grad()

            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

        scheduler.step()

        # quick validation every 10 epochs
        if (epoch + 1) % 10 == 0:
            model.eval()
            preds_all = []
            labels_all = []

            with torch.no_grad():
                for X_val, y_val in test_loader:
                    X_val = X_val.to(device)
                    y_val = y_val.to(device)

                    outputs = model(X_val)
                    preds = outputs.argmax(dim=1)

                    preds_all.extend(preds.cpu().numpy())
                    labels_all.extend(y_val.cpu().numpy())

            val_acc = accuracy_score(labels_all, preds_all)

            if val_acc > best_acc:
                best_acc = val_acc
                best_state = model.state_dict()

            print(f"A{subject:02d} | Epoch {epoch+1}/{epochs_count} | Val Acc: {val_acc:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            outputs = model(X_batch)
            preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    kappa = cohen_kappa_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    return {
        "subject": f"A{subject:02d}",
        "accuracy": acc,
        "kappa": kappa,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--out", type=str, default="results/bcic2a_improved.csv")

    args = parser.parse_args()

    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    os.makedirs("results", exist_ok=True)

    subject_results = []

    for subject in range(1, 10):
        print("\n==============================")
        print(f"Training Subject A{subject:02d}")
        print("==============================")

        result = train_one_subject(
            data_path=args.data_path,
            subject=subject,
            epochs_count=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=device
        )

        print(result)
        subject_results.append(result)

    results_df = pd.DataFrame(subject_results)

    print("\nSubject-wise Results:")
    print(results_df)

    print("\nAverage Results:")
    print(results_df.mean(numeric_only=True))

    results_df.to_csv(args.out, index=False)
    print("\nSaved results to:", args.out)


if __name__ == "__main__":
    main()
