import os
import numpy as np
import mne

from mne.filter import notch_filter, filter_data
from scipy.signal import stft
from skimage.transform import resize
from sklearn.utils import shuffle


# ============================================================
# Common Average Reference (CAR)
# ============================================================

def common_average_reference(data):
    """
    Apply Common Average Reference (CAR)
    data shape : (channels, samples)
    """
    return data - np.mean(data, axis=0, keepdims=True)


# ============================================================
# Min-Max Normalization
# ============================================================

def min_max_normalize(channel):
    """
    Normalize one EEG channel into [0,1]
    """
    return (channel - np.min(channel)) / (
        np.max(channel) - np.min(channel) + 1e-6
    )


# ============================================================
# Sliding Window
# ============================================================

def sliding_window_segments(
    data,
    window_size,
    step
):
    """
    Parameters
    ----------
    data : (3, samples)

    Returns
    -------
    (num_windows,3,window_size)
    """

    windows = []

    for start in range(
            0,
            data.shape[1] - window_size + 1,
            step):

        end = start + window_size

        windows.append(
            data[:, start:end]
        )

    return np.asarray(windows)


# ============================================================
# STFT
# ============================================================

def stft_spectrogram(
    signal,
    fs=250,
    window_size=64,
    overlap=50
):
    """
    Compute STFT Magnitude
    """

    f, t, Zxx = stft(
        signal,
        fs=fs,
        window="hann",
        nperseg=window_size,
        noverlap=overlap,
        boundary=None
    )

    return f, t, np.abs(Zxx)


# ============================================================
# Extract Frequency Band
# ============================================================

def extract_band_spectrogram(
    Sxx,
    f,
    low,
    high
):

    idx = np.where(
        (f >= low) &
        (f <= high)
    )[0]

    return Sxx[idx]


# ============================================================
# Resize Spectrogram
# ============================================================

def resize_spectrogram(
    spec,
    target_shape=(64, 64)
):

    return resize(
        spec,
        target_shape,
        anti_aliasing=True,
        mode="reflect"
    )


# ============================================================
# Spectrogram Normalization
# ============================================================

def normalize_image(img):

    img = (
        img - img.min()
    ) / (
        img.max() - img.min() + 1e-6
    )

    return img.astype(np.float32)


# ============================================================
# Data Augmentation
# ============================================================

def augment_spectrogram(img):

    # Gaussian Noise
    noise = np.random.normal(
        0,
        0.005,
        img.shape
    )

    img = img + noise

    # Random Scaling

    scale = np.random.uniform(
        0.95,
        1.05
    )

    img = img * scale

    # Time Shift

    shift = np.random.randint(
        -2,
        3
    )

    img = np.roll(
        img,
        shift,
        axis=1
    )

    return img


# ============================================================
# Subject ID
# ============================================================

def get_true_subject_id(filename):

    return int(filename[1:3]) - 1

# ============================================================
# Main Preprocessing Function
# ============================================================

def preprocess_gdf_folder(
    folder_path,
    selected_channels=["c3", "cz", "c4"],
    fs=250,
    trial_window_sec=4,
    sliding_window_sec=2,
    sliding_step_sec=0.5,
    stft_window=64,
    stft_overlap=50,
    add_gaussian_noise=True
):

    X = []
    y = []
    subject_ids = []
    trial_ids = []

    global_trial_id = 0

    trial_window_samples = int(trial_window_sec * fs)
    sliding_window_samples = int(sliding_window_sec * fs)
    sliding_step_samples = int(sliding_step_sec * fs)

    event_code_to_label = {
        769: 0,      # Left Hand
        770: 1,      # Right Hand
        771: 2,      # Feet
        772: 3       # Tongue
    }

    # ----------------------------------------------------
    # Read Every Training File
    # ----------------------------------------------------

    for filename in sorted(os.listdir(folder_path)):

        if not filename.endswith(".gdf"):
            continue

        # Skip evaluation sessions
        if "E" in filename:
            continue

        filepath = os.path.join(folder_path, filename)

        subject_id = get_true_subject_id(filename)

        print(f"\nProcessing {filename}")

        try:

            with mne.utils.use_log_level("ERROR"):

                raw = mne.io.read_raw_gdf(
                    filepath,
                    preload=True
                )

        except Exception as e:

            print(f"Error : {e}")

            continue

        # ----------------------------------------------------
        # Channel Selection
        # ----------------------------------------------------

        data = raw.get_data()

        channel_names = raw.info["ch_names"]

        cleaned_names = [

            ch.lower()

            .replace("eeg:", "")

            .replace("eeg-", "")

            .strip()

            for ch in channel_names

        ]

        try:

            indices = [

                cleaned_names.index(ch.lower())

                for ch in selected_channels

            ]

        except ValueError:

            print("Required channels not found.")

            continue

        data = data[indices]

        # ----------------------------------------------------
        # Common Average Reference
        # ----------------------------------------------------

        data = common_average_reference(data)

        # ----------------------------------------------------
        # 50 Hz Notch
        # ----------------------------------------------------

        with mne.utils.use_log_level("WARNING"):

            data = notch_filter(

                data,

                Fs=fs,

                freqs=50,

                method="fir"

            )

        # ----------------------------------------------------
        # 8-30 Hz Bandpass
        # ----------------------------------------------------

        with mne.utils.use_log_level("WARNING"):

            data = filter_data(

                data,

                sfreq=fs,

                l_freq=8,

                h_freq=30,

                method="fir",

                phase="zero-double",

                fir_window="hamming"

            )

        # ----------------------------------------------------
        # Normalize Each EEG Channel
        # ----------------------------------------------------

        data = np.asarray([

            min_max_normalize(ch)

            for ch in data

        ])

        # ----------------------------------------------------
        # Read Events
        # ----------------------------------------------------

        annotations = raw.annotations

        event_onsets = (

            annotations.onset * fs

        ).astype(int)

        event_desc = annotations.description

        artifact_times = [

            int((onset + 2) * fs)

            for onset, desc

            in zip(

                annotations.onset,

                event_desc

            )

            if desc == "1023"

        ]

        samples_from_subject = 0

        # ----------------------------------------------------
        # Iterate Through Trials
        # ----------------------------------------------------

        for i, (onset, desc) in enumerate(

                zip(

                    event_onsets,

                    event_desc

                )):

            if desc != "768":
                continue

            if i + 1 >= len(event_desc):
                continue

            try:

                cue = int(

                    event_desc[i + 1].strip()

                )

            except:

                continue

            if cue not in event_code_to_label:
                continue

            label = event_code_to_label[cue]

            start = onset + 2 * fs

            end = start + trial_window_samples

            if end > data.shape[1]:
                continue

            # Remove Artifact Trials

            if any(

                abs(start - art)

                < trial_window_samples

                for art in artifact_times

            ):

                continue

            current_trial = global_trial_id

            global_trial_id += 1

            trial = data[:, start:end]

            # --------------------------------------------
            # Sliding Window
            # --------------------------------------------

            segments = sliding_window_segments(

                trial,

                sliding_window_samples,

                sliding_step_samples

            )
                        # ----------------------------------------------------
            # Convert Each Sliding Window to STFT Spectrogram
            # ----------------------------------------------------

            for segment in segments:

                channel_images = []

                for ch in range(len(selected_channels)):

                    signal = segment[ch]

                    # STFT
                    f, t, Sxx = stft_spectrogram(
                        signal,
                        fs=fs,
                        window_size=stft_window,
                        overlap=stft_overlap
                    )

                    # Keep Entire 8-30 Hz Band
                    spec = extract_band_spectrogram(
                        Sxx,
                        f,
                        8,
                        30
                    )

                    # Resize
                    spec = resize_spectrogram(
                        spec,
                        target_shape=(64, 64)
                    )

                    channel_images.append(spec)

                # ----------------------------------------
                # Stack C3/Cz/C4
                # Shape : (3,64,64)
                # ----------------------------------------

                img = np.stack(
                    channel_images,
                    axis=0
                )

                # Convert to CNN input
                # Shape : (1,3,64,64)

                img = np.expand_dims(
                    img,
                    axis=0
                )

                # ----------------------------------------
                # Data Augmentation
                # ----------------------------------------

                if add_gaussian_noise:

                    img = augment_spectrogram(img)

                # ----------------------------------------
                # Normalize
                # ----------------------------------------

                img = normalize_image(img)

                X.append(img)

                y.append(label)

                subject_ids.append(subject_id)

                trial_ids.append(current_trial)

                samples_from_subject += 1

        print(
            f"Generated {samples_from_subject} samples."
        )

    # =====================================================
    # Convert to NumPy
    # =====================================================

    X = np.asarray(
        X,
        dtype=np.float32
    )

    y = np.asarray(
        y,
        dtype=np.int64
    )

    subject_ids = np.asarray(
        subject_ids,
        dtype=np.int64
    )

    trial_ids = np.asarray(
        trial_ids,
        dtype=np.int64
    )

    # =====================================================
    # Shuffle
    # =====================================================

    if len(X) > 0:

        X, y, subject_ids, trial_ids = shuffle(

            X,
            y,
            subject_ids,
            trial_ids,

            random_state=42

        )

    # =====================================================
    # Information
    # =====================================================

    print("\n====================================")

    print("Preprocessing Finished")

    print("====================================")

    print("Samples :", len(X))

    print("Classes :", np.unique(y))

    print("Subjects :", len(np.unique(subject_ids)))

    print("Trials :", len(np.unique(trial_ids)))

    print("X Shape :", X.shape)

    print("y Shape :", y.shape)

    print("====================================")

    return (
        X,
        y,
        subject_ids,
        trial_ids
    )
    


