import os
import numpy as np
import mne
import torch

from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split


BANDS = {
    "theta": (4, 8),
    "mu": (8, 13),
    "low_beta": (13, 20),
    "high_beta": (20, 30),
    "gamma": (30, 40)
}


def extract_band_epochs(raw, events, event_id, tmin=0.5, tmax=4.5):
    band_epochs = []

    for band_name, (low, high) in BANDS.items():
        raw_band = raw.copy()

        raw_band.filter(
            l_freq=low,
            h_freq=high,
            fir_design="firwin",
            verbose=False
        )

        epochs = mne.Epochs(
            raw_band,
            events,
            event_id=event_id,
            tmin=tmin,
            tmax=tmax,
            baseline=None,
            preload=True,
            verbose=False
        )

        band_epochs.append(epochs.get_data())

    X = np.stack(band_epochs, axis=1)
    y = epochs.events[:, -1]

    return X, y


def load_subject_2a(data_path, subject, test_size=0.2, batch_size=32):
    file_path = os.path.join(data_path, f"A{subject:02d}T.gdf")

    raw = mne.io.read_raw_gdf(
        file_path,
        preload=True,
        verbose=False
    )

    raw = raw.drop_channels([
        "EOG-left",
        "EOG-central",
        "EOG-right"
    ])

    # Optional EEG reference
    raw.set_eeg_reference("average", verbose=False)

    events, event_dict = mne.events_from_annotations(
        raw,
        verbose=False
    )

    event_id = {
        "left_hand": event_dict["769"],
        "right_hand": event_dict["770"],
        "feet": event_dict["771"],
        "tongue": event_dict["772"]
    }

    X, y = extract_band_epochs(
        raw,
        events,
        event_id,
        tmin=0.5,
        tmax=4.5
    )

    label_map = {
        event_dict["769"]: 0,
        event_dict["770"]: 1,
        event_dict["771"]: 2,
        event_dict["772"]: 3
    }

    y = np.array([label_map[label] for label in y])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
        stratify=y
    )

    # Normalize using training data only
    mean = X_train.mean(axis=(0, 3), keepdims=True)
    std = X_train.std(axis=(0, 3), keepdims=True) + 1e-6

    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)

    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        TensorDataset(X_test, y_test),
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, test_loader
