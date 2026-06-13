import os
import numpy as np
import mne
from mne.filter import notch_filter, filter_data
from scipy.signal import stft
from skimage.transform import resize
from sklearn.utils import shuffle


def common_average_reference(data):
    return data - np.mean(data, axis=0, keepdims=True)


def min_max_normalize(channel_data):
    return (channel_data - np.min(channel_data)) / (
        np.max(channel_data) - np.min(channel_data) + 1e-6
    )


def sliding_window_segments(data, window_size, step):
    return np.array([
        data[:, start:start + window_size]
        for start in range(0, data.shape[1] - window_size + 1, step)
    ])


def stft_spectrogram(segment, fs=250, window_size=64, overlap=50):
    f, t, Zxx = stft(
        segment,
        fs=fs,
        window="hann",
        nperseg=window_size,
        noverlap=overlap
    )
    return f, t, np.abs(Zxx)


def extract_band_spectrogram(Sxx, f, band_low, band_high):
    idx = np.where((f >= band_low) & (f <= band_high))[0]
    return Sxx[idx, :]


def resize_spectrogram(spec, target_shape=(20, 32)):
    return resize(
        spec,
        target_shape,
        mode="reflect",
        anti_aliasing=True
    )


def get_true_subject_id(filename):
    subject_number = int(filename[1:3])
    return subject_number - 1


def preprocess_gdf_folder(
    folder_path,
    selected_channels=["c3", "cz", "c4"],
    fs=250,
    trial_window_sec=4,
    sliding_window_sec=2,
    sliding_step_sec=0.1,
    stft_window=64,
    stft_overlap=50,
    add_gaussian_noise=False
):
    X, y, subject_ids = [], [], []

    trial_window_samples = int(trial_window_sec * fs)
    sliding_window_samples = int(sliding_window_sec * fs)
    sliding_step_samples = int(sliding_step_sec * fs)

    event_code_to_label = {
        769: 0,
        770: 1,
        771: 2,
        772: 3
    }

    for filename in sorted(os.listdir(folder_path)):
        if not filename.endswith(".gdf"):
            continue

        if "E" in filename:
            continue

        filepath = os.path.join(folder_path, filename)
        subject_id = get_true_subject_id(filename)

        try:
            with mne.utils.use_log_level("ERROR"):
                raw = mne.io.read_raw_gdf(filepath, preload=True)
        except Exception as e:
            print(f"Failed to load {filename}: {e}")
            continue

        print(f"Processing {filename}")

        data = raw.get_data()
        channel_names = raw.info["ch_names"]

        cleaned_names = [
            ch.lower().replace("eeg:", "").replace("eeg-", "").strip()
            for ch in channel_names
        ]

        try:
            idx = [cleaned_names.index(ch.lower()) for ch in selected_channels]
        except ValueError as e:
            print(f"Skipping {filename}, missing channel: {e}")
            continue

        data_sel = data[idx, :]

        data_car = common_average_reference(data_sel)

        with mne.utils.use_log_level("WARNING"):
            data_notch = notch_filter(
                data_car,
                fs,
                freqs=50,
                method="fir"
            )

            data_band = filter_data(
                data_notch,
                fs,
                l_freq=8,
                h_freq=30,
                method="fir",
                phase="zero-double",
                fir_window="hamming"
            )

        data_norm = np.array([
            min_max_normalize(ch)
            for ch in data_band
        ])

        annotations = raw.annotations
        event_onsets = (annotations.onset * fs).astype(int)
        event_descriptions = annotations.description

        artifact_times = [
            int((onset + 2) * fs)
            for onset, desc in zip(annotations.onset, event_descriptions)
            if desc == "1023"
        ]

        trials_in_file = 0

        for i, (onset, desc) in enumerate(zip(event_onsets, event_descriptions)):
            if desc != "768":
                continue

            if i + 1 >= len(event_descriptions):
                continue

            try:
                cue_code = int(event_descriptions[i + 1].strip())
            except Exception:
                continue

            if cue_code not in event_code_to_label:
                continue

            label = event_code_to_label[cue_code]

            start_sample = onset + 2 * fs
            end_sample = start_sample + trial_window_samples

            if any(abs(start_sample - a) < trial_window_samples for a in artifact_times):
                continue

            if end_sample > data_norm.shape[1]:
                continue

            trial = data_norm[:, start_sample:end_sample]

            segments = sliding_window_segments(
                trial,
                sliding_window_samples,
                sliding_step_samples
            )

            for segment in segments:
                channel_imgs = []

                for ch in range(len(selected_channels)):
                    sig = segment[ch, :]

                    f, t, Sxx = stft_spectrogram(
                        sig,
                        fs=fs,
                        window_size=stft_window,
                        overlap=stft_overlap
                    )

                    mu_spec = extract_band_spectrogram(Sxx, f, 8, 14)
                    mu_resized = resize_spectrogram(mu_spec)

                    beta_spec = extract_band_spectrogram(Sxx, f, 16, 30)
                    beta_resized = resize_spectrogram(beta_spec)

                    combined_spec = np.vstack([
                        mu_resized,
                        beta_resized
                    ])

                    channel_imgs.append(combined_spec)

                img = np.vstack(channel_imgs)
                img = np.repeat(img[:, :, np.newaxis], 3, axis=-1)

                if add_gaussian_noise:
                    noise = np.random.normal(0, 0.01, img.shape)
                    img = img + noise

                img = (img - np.mean(img)) / (np.std(img) + 1e-6)

                X.append(img)
                y.append(label)
                subject_ids.append(subject_id)
                trials_in_file += 1

        print(f"Extracted {trials_in_file} samples from {filename}")

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    subject_ids = np.array(subject_ids, dtype=np.int64)

    if len(X) > 0:
        X, y, subject_ids = shuffle(
            X,
            y,
            subject_ids,
            random_state=42
        )

    print("Finished preprocessing")
    print("X shape:", X.shape)
    print("y shape:", y.shape)

    return X, y, subject_ids