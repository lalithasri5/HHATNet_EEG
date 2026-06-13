import os
import argparse
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    precision_score,
    recall_score,
    f1_score
)

from data.stft_loader import preprocess_gdf_folder
from models.stft_cnn import STFTCNN


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_path",
        type=str,
        required=True
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.001
    )

    parser.add_argument(
        "--out",
        type=str,
        default="results/stft_results.csv"
    )

    args = parser.parse_args()

    set_seed()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Using device:", device)

    os.makedirs("results", exist_ok=True)

    X, y, subject_ids = preprocess_gdf_folder(
        args.data_path
    )

    print("Preprocessing complete")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

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

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=args.batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        TensorDataset(X_test, y_test),
        batch_size=args.batch_size,
        shuffle=False
    )

    model = STFTCNN(
        n_classes=4
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.01
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs
    )

    for epoch in range(args.epochs):

        model.train()

        for X_batch, y_batch in train_loader:

            X_batch = X_batch.to(device)

            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            outputs = model(X_batch)

            loss = criterion(
                outputs,
                y_batch
            )

            loss.backward()

            optimizer.step()

        scheduler.step()

        if (epoch + 1) % 10 == 0:

            print(
                f"Epoch {epoch+1}/{args.epochs} completed"
            )

    model.eval()

    all_preds = []

    all_labels = []

    with torch.no_grad():

        for X_batch, y_batch in test_loader:

            X_batch = X_batch.to(device)

            outputs = model(X_batch)

            preds = outputs.argmax(
                dim=1
            )

            all_preds.extend(
                preds.cpu().numpy()
            )

            all_labels.extend(
                y_batch.numpy()
            )

    results = {

        "accuracy":
            accuracy_score(
                all_labels,
                all_preds
            ),

        "kappa":
            cohen_kappa_score(
                all_labels,
                all_preds
            ),

        "precision":
            precision_score(
                all_labels,
                all_preds,
                average="macro"
            ),

        "recall":
            recall_score(
                all_labels,
                all_preds,
                average="macro"
            ),

        "f1":
            f1_score(
                all_labels,
                all_preds,
                average="macro"
            )
    }

    print("\nResults:")

    print(results)

    pd.DataFrame(
        [results]
    ).to_csv(
        args.out,
        index=False
    )

    print(
        "\nSaved to:",
        args.out
    )


if __name__ == "__main__":

    main()