import os
import numpy as np
import mne


# -------------------------------------------------
# Motor Imagery Classes
# -------------------------------------------------

EVENTS = {
    "769": 0,   # Left Hand
    "770": 1,   # Right Hand
    "771": 2,   # Feet
    "772": 3    # Tongue
}


# -------------------------------------------------
# EEG Channels
# -------------------------------------------------

SELECTED_CHANNELS = [
    "C3",
    "Cz",
    "C4"
]


# -------------------------------------------------
# Sliding Window Function
# -------------------------------------------------

def sliding_window(data, labels, window_size=250, step_size=125):
    """
    Parameters
    ----------
    data : ndarray
        Shape -> (Trials, Channels, Samples)

    labels : ndarray
        Shape -> (Trials,)

    window_size : int
        Samples in each window

    step_size : int
        Overlap step

    Returns
    -------
    windowed_data
        Shape -> (NewTrials, Channels, window_size)

    windowed_labels
    """

    windows = []
    window_labels = []

    for trial, label in zip(data, labels):

        total_samples = trial.shape[-1]

        start = 0

        while start + window_size <= total_samples:

            windows.append(
                trial[:, start:start + window_size]
            )

            window_labels.append(label)

            start += step_size

    return np.asarray(windows), np.asarray(window_labels)


# -------------------------------------------------
# Load BCIC IV-2a Subject
# -------------------------------------------------

def load_subject(
        data_path,
        subject=1,
        train=True,
        tmin=0.5,
        tmax=4.5,
        use_sliding_window=True,
        window_size=250,
        step_size=125):

    if train:
        filename = f"A{subject:02d}T.gdf"
    else:
        filename = f"A{subject:02d}E.gdf"

    filepath = os.path.join(data_path, filename)

    print(f"Loading {filepath}")

    raw = mne.io.read_raw_gdf(
        filepath,
        preload=True,
        verbose=False
    )

    # -----------------------------------------
    # Remove EOG
    # -----------------------------------------

    eog_channels = [
        ch for ch in raw.ch_names
        if "EOG" in ch.upper()
    ]

    if len(eog_channels):
        raw.drop_channels(eog_channels)

    # -----------------------------------------
    # Select C3 Cz C4
    # -----------------------------------------

    raw.pick_channels(SELECTED_CHANNELS)

    # -----------------------------------------
    # CAR
    # -----------------------------------------

    raw.set_eeg_reference(
        ref_channels="average",
        verbose=False
    )

    # -----------------------------------------
    # 50 Hz Notch
    # -----------------------------------------

    raw.notch_filter(
        freqs=50,
        verbose=False
    )

    # -----------------------------------------
    # 8-30 Hz Bandpass
    # -----------------------------------------

    raw.filter(
        l_freq=8,
        h_freq=30,
        fir_design="firwin",
        verbose=False
    )

    # -----------------------------------------
    # Events
    # -----------------------------------------

    events, event_dict = mne.events_from_annotations(
        raw,
        verbose=False
    )

    event_id = {}

    for key in EVENTS:

        if key in event_dict:

            event_id[key] = event_dict[key]

    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_id,
        tmin=tmin,
        tmax=tmax,
        baseline=None,
        preload=True,
        verbose=False
    )

    X = epochs.get_data()

    labels = epochs.events[:, -1]

    label_map = {}

    for key in EVENTS:

        if key in event_dict:

            label_map[event_dict[key]] = EVENTS[key]

    y = np.array(
        [label_map[label] for label in labels],
        dtype=np.int64
    )

    # -----------------------------------------
    # Sliding Window
    # -----------------------------------------

    if use_sliding_window:

        X, y = sliding_window(
            X,
            y,
            window_size=window_size,
            step_size=step_size
        )

    print()

    print("Subject :", subject)
    print("EEG Shape :", X.shape)
    print("Labels :", y.shape)

    print()

    return X, y
