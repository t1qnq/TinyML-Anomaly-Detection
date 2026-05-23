"""Extract 19-dimensional features from collected WAV/CSV samples.

Feature layout:
  0..12 : 13 log-Mel energy bands from INMP441 audio
  13    : rms_x
  14    : var_x
  15    : rms_y
  16    : var_y
  17    : rms_z
  18    : var_z

The audio path intentionally matches the firmware implementation:
512-point FFT, 256-sample hop, 30 frames per one-second window and the
same fixed triangular Mel filterbank.
"""

from __future__ import annotations

import argparse
import glob
import os
import wave
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_DATA_DIR = "dataset_v6/normal"
DEFAULT_OUT_CSV = "train_features_v6.csv"

AUDIO_SAMPLE_RATE_HZ = 8000
WINDOW_SECONDS = 1
WINDOWS_PER_FILE = 5
N_FFT = 512
HOP_LENGTH = 256
NUM_FRAMES = 30
N_MELS = 13
VIB_ROWS_PER_WINDOW = 1000

AUDIO_SAMPLES_PER_WINDOW = AUDIO_SAMPLE_RATE_HZ * WINDOW_SECONDS

# These caps are part of the feature pipeline used to create train_features_v6.csv.
# They limit rare vibration spikes before the row is saved, keeping the extractor
# consistent with the data distribution documented in latex_v2.
VIB_CLIP_RMS_Z = 0.30
VIB_CLIP_VAR_X = 0.001097
VIB_CLIP_VAR_Y = 0.000978
VIB_CLIP_VAR_Z = 0.060464

MEL_FILTERBANK = (
    (1, 11, 21),
    (11, 21, 32),
    (22, 32, 42),
    (33, 43, 53),
    (43, 54, 64),
    (54, 64, 76),
    (65, 76, 90),
    (77, 91, 107),
    (91, 108, 128),
    (108, 128, 152),
    (129, 153, 181),
    (153, 181, 215),
    (182, 215, 255),
)


def read_wav_int16_mono(path: str | Path, expected_sr: int = AUDIO_SAMPLE_RATE_HZ) -> np.ndarray:
    """Read a mono 16-bit PCM WAV file and keep the original int16 scale."""
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        frame_count = wf.getnframes()
        if channels != 1:
            raise ValueError(f"{path}: expected mono WAV, got {channels} channels")
        if sample_width != 2:
            raise ValueError(f"{path}: expected 16-bit PCM WAV, got {sample_width} bytes")
        if sample_rate != expected_sr:
            raise ValueError(f"{path}: sample rate {sample_rate} != {expected_sr}")
        raw = wf.readframes(frame_count)
    return np.frombuffer(raw, dtype="<i2").astype(np.float32)


def hann_window(n: int) -> np.ndarray:
    """Build the same Hann analysis window used before each FFT frame."""
    i = np.arange(n, dtype=np.float32)
    return 0.5 * (1.0 - np.cos(2.0 * np.pi * i / (n - 1)))


HANN_512 = hann_window(N_FFT)


def frame_audio_window(segment: np.ndarray) -> np.ndarray:
    """Create 30 firmware-compatible frames from one second of audio."""
    prev = np.zeros(HOP_LENGTH, dtype=np.float32)
    frames = np.empty((NUM_FRAMES, N_FFT), dtype=np.float32)
    for frame_idx in range(NUM_FRAMES):
        start = frame_idx * HOP_LENGTH
        new = segment[start : start + HOP_LENGTH].astype(np.float32, copy=False)
        if len(new) < HOP_LENGTH:
            new = np.pad(new, (0, HOP_LENGTH - len(new)), mode="constant")
        frames[frame_idx, :HOP_LENGTH] = prev
        frames[frame_idx, HOP_LENGTH:] = new
        prev = new
    return frames


def mel_power_from_rfft(rfft: np.ndarray) -> np.ndarray:
    """Apply the fixed firmware Mel filterbank to one FFT spectrum."""
    power = (rfft.real.astype(np.float32) ** 2) + (rfft.imag.astype(np.float32) ** 2)
    if len(power) > 2:
        power[1:-1] *= 2.0

    bands = np.empty(N_MELS, dtype=np.float32)
    for band_idx, (lo, mid, hi) in enumerate(MEL_FILTERBANK):
        acc = 0.0
        rising_denom = float(mid - lo) if mid > lo else 1.0
        for k in range(lo, min(mid, N_FFT // 2) + 1):
            acc += ((k - lo) / rising_denom) * float(power[k])

        falling_denom = float(hi - mid) if hi > mid else 1.0
        for k in range(mid + 1, min(hi, N_FFT // 2) + 1):
            acc += ((hi - k) / falling_denom) * float(power[k])

        bands[band_idx] = acc + 1e-10
    return bands


def mel_features(segment: np.ndarray) -> np.ndarray:
    """Return 13 mean log-Mel features for one one-second audio segment."""
    frames = frame_audio_window(segment) * HANN_512[None, :]
    powers = np.empty((NUM_FRAMES, N_MELS), dtype=np.float32)
    max_power = 0.0

    for frame_idx in range(NUM_FRAMES):
        spectrum = np.fft.rfft(frames[frame_idx], n=N_FFT)
        band_power = mel_power_from_rfft(spectrum)
        powers[frame_idx] = band_power
        max_power = max(max_power, float(np.max(band_power)))

    max_power = max(max_power, 1e-10)
    db = 10.0 * np.log10(powers / max_power)
    db = np.maximum(db, -80.0)
    return np.mean(db, axis=0).astype(np.float32)


def vibration_features(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Return clipped RMS/variance features for the three ADXL345 axes.

    The ordering and clipping match the extractor used before the final cleanup:
    [rms_x, var_x, rms_y, var_y, rms_z, var_z]. The fixed caps prevent a small
    number of sensor spikes from dominating phase routing and scaler fitting.
    """
    x = x.astype(np.float32, copy=False)
    y = y.astype(np.float32, copy=False)
    z = z.astype(np.float32, copy=False)

    def rms(values: np.ndarray) -> float:
        """Compute root-mean-square acceleration for one axis."""
        return float(np.sqrt(np.mean(values * values)))

    def var(values: np.ndarray) -> float:
        """Compute acceleration variance for one axis."""
        return float(np.var(values))

    features = np.array(
        [rms(x), var(x), rms(y), var(y), rms(z), var(z)],
        dtype=np.float32,
    )
    features[1] = min(features[1], VIB_CLIP_VAR_X)
    features[3] = min(features[3], VIB_CLIP_VAR_Y)
    features[4] = min(features[4], VIB_CLIP_RMS_Z)
    features[5] = min(features[5], VIB_CLIP_VAR_Z)
    return features


def extract_file(wav_path: str | Path, csv_path: str | Path) -> tuple[list[np.ndarray], str | None]:
    """Extract five one-second feature rows from a matched WAV/CSV pair."""
    audio = read_wav_int16_mono(wav_path)
    vib_df = pd.read_csv(csv_path, usecols=["X", "Y", "Z"], dtype=np.float32)

    required_audio = AUDIO_SAMPLES_PER_WINDOW * WINDOWS_PER_FILE
    required_vib = VIB_ROWS_PER_WINDOW * WINDOWS_PER_FILE
    if len(audio) < required_audio:
        return [], f"audio too short: {len(audio)} < {required_audio}"
    if len(vib_df) < required_vib:
        return [], f"vibration CSV too short: {len(vib_df)} < {required_vib}"

    vx = vib_df["X"].values
    vy = vib_df["Y"].values
    vz = vib_df["Z"].values

    rows: list[np.ndarray] = []
    for idx in range(WINDOWS_PER_FILE):
        a0 = idx * AUDIO_SAMPLES_PER_WINDOW
        a1 = (idx + 1) * AUDIO_SAMPLES_PER_WINDOW
        v0 = idx * VIB_ROWS_PER_WINDOW
        v1 = (idx + 1) * VIB_ROWS_PER_WINDOW

        feat = np.concatenate(
            [
                mel_features(audio[a0:a1]),
                vibration_features(vx[v0:v1], vy[v0:v1], vz[v0:v1]),
            ]
        )
        rows.append(feat.astype(np.float32))
    return rows, None


def feature_columns() -> list[str]:
    """Return the fixed 19-column schema used by training and firmware."""
    return [f"mfe_{idx}" for idx in range(N_MELS)] + [
        "rms_x",
        "var_x",
        "rms_y",
        "var_y",
        "rms_z",
        "var_z",
    ]


def print_summary(df: pd.DataFrame) -> None:
    """Print descriptive statistics to detect bad feature ranges early."""
    mel_cols = [f"mfe_{idx}" for idx in range(N_MELS)]
    vib_cols = ["rms_x", "var_x", "rms_y", "var_y", "rms_z", "var_z"]

    print("\nAudio log-Mel summary:")
    print(df[mel_cols].describe().round(2).to_string())
    print("\nVibration summary:")
    print(df[vib_cols].describe().round(6).to_string())
    print(f"\nOutput shape: {len(df)} x {len(df.columns)}")


def parse_args() -> argparse.Namespace:
    """Parse command-line options for batch feature extraction."""
    parser = argparse.ArgumentParser(description="Extract TinyML training features")
    parser.add_argument("--data-dir", "--data_dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--out", default=DEFAULT_OUT_CSV)
    parser.add_argument("--limit-files", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    """Convert all collected sample pairs into the training feature CSV."""
    args = parse_args()
    wav_files = sorted(glob.glob(os.path.join(args.data_dir, "*.wav")))
    if args.limit_files > 0:
        wav_files = wav_files[: args.limit_files]

    if not wav_files:
        raise SystemExit(f"No WAV files found in {args.data_dir}")

    all_rows: list[np.ndarray] = []
    errors: list[str] = []

    for file_idx, wav_path in enumerate(wav_files, start=1):
        csv_path = os.path.splitext(wav_path)[0] + ".csv"
        if not os.path.exists(csv_path):
            errors.append(f"{os.path.basename(wav_path)}: missing CSV")
            continue

        rows, error = extract_file(wav_path, csv_path)
        if error:
            errors.append(f"{os.path.basename(wav_path)}: {error}")
            continue
        all_rows.extend(rows)

        if file_idx % 100 == 0:
            print(f"Processed {file_idx}/{len(wav_files)} files")

    if errors:
        print(f"Warnings: {len(errors)} files skipped")
        for error in errors[:10]:
            print(f"  - {error}")

    if not all_rows:
        raise SystemExit("No features extracted")

    df = pd.DataFrame(all_rows, columns=feature_columns())
    df.to_csv(args.out, index=False)
    print(f"\nWrote {args.out}")
    print_summary(df)


if __name__ == "__main__":
    main()
