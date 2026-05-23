# TinyML Anomaly Detection for Washing Machines

This branch contains the clean `fix-v2` source code for a TinyML washing-machine anomaly detection system. The system runs Autoencoder inference on a Seeed Studio XIAO ESP32-S3, publishes status through MQTT, and displays live monitoring data in a Flutter app.

The report, LaTeX source, slide decks, generated figures, datasets, and demo videos are intentionally excluded from this branch.

## Core Components

| Component | Final file |
|---|---|
| Data collection receiver | `final_data_collection.py` |
| Data collection firmware | `dataCollection/final_data_collection.ino` |
| Feature extraction | `final_feature_extraction.py` |
| Leakage-free training | `final_training.py` |
| Runtime firmware | `firmware_v11/final_firmware.ino` |
| Model header | `firmware_v11/model_data_final.h` |
| Flutter app | `tinyml_app/` |
| Local config generator | `tools/generate_env_config.py` |

Legacy files are kept for comparison, but the files listed above are the intended final entry points.

## Hardware

- MCU: Seeed Studio XIAO ESP32-S3
- Microphone: INMP441 over I2S
  - WS: GPIO6
  - BCLK: GPIO43
  - DIN: GPIO44
- Accelerometer: ADXL345 over I2C
  - SDA: GPIO5
  - SCL: GPIO4
  - Address: `0x53`

## Feature Vector

Each one-second window is converted into a 19-dimensional vector:

- `0..12`: 13 log-Mel audio features
- `13`: `rms_x`
- `14`: `var_x`
- `15`: `rms_y`
- `16`: `var_y`
- `17`: `rms_z`
- `18`: `var_z`

`var_z` is also used to route each window into one of three physical phases: `GENTLE`, `STRONG`, or `SPIN`.

## Training Pipeline

Run:

```powershell
python final_training.py --csv train_features_v6.csv --out-header firmware_v11/model_data_final.h --results results_final.json --phase-threshold-mode kmeans --anomaly-threshold-mode optimal_f1
```

The final training script:

- computes phase switch thresholds with K-Means on raw `var_z`
- splits raw data into train, validation, and hold-out test before augmentation
- fits vibration scalers only on training data
- trains one QAT INT8 Autoencoder per phase
- evaluates on the hold-out test split with controlled anomaly injection
- exports `firmware_v11/model_data_final.h`

Current rounded MAE thresholds used by the app:

| Phase | MAE threshold |
|---|---:|
| GENTLE | `0.0469` |
| STRONG | `0.1077` |
| SPIN | `0.0989` |

## Local Configuration

Credentials are not committed.

1. Copy the example file:

```powershell
Copy-Item .env.example .env
```

2. Fill in local Wi-Fi and MQTT values in `.env`.

3. Generate firmware and Flutter config files:

```powershell
python tools/generate_env_config.py
```

This creates:

- `firmware_v11/env_config.h`
- `tinyml_app/lib/app_config.dart`

Both generated files are ignored by Git.

## Data Collection

Flash:

```text
dataCollection/final_data_collection.ino
```

Collect samples:

```powershell
python final_data_collection.py --port COM5 --out dataset_v6/normal --duration 5
```

Each sample produces:

- `sample_XXXX.wav`
- `sample_XXXX.csv`

## Feature Extraction

Run:

```powershell
python final_feature_extraction.py --data-dir dataset_v6/normal --out train_features_v6.csv
```

For a quick smoke test:

```powershell
python final_feature_extraction.py --data-dir dataset_v6/normal --out tmp_features_check.csv --limit-files 1
```

## Firmware

Open and flash:

```text
firmware_v11/final_firmware.ino
```

The firmware expects:

- `firmware_v11/model_data_final.h`
- `firmware_v11/env_config.h`

Runtime behavior:

- Core 0 handles I2S audio capture and FFT/log-Mel extraction.
- Core 1 handles vibration features, phase routing, Autoencoder inference, and MQTT publishing.
- Alarm logic uses a 10-window sliding window:
  - enter alarm when at least `5/10` windows are HIGH
  - exit alarm when at most `1/10` windows are HIGH

## Flutter App

Run from the app directory:

```powershell
cd tinyml_app
C:\Users\Admin\flutter\bin\flutter.bat pub get
C:\Users\Admin\flutter\bin\flutter.bat run -d chrome
```

For release web build:

```powershell
C:\Users\Admin\flutter\bin\flutter.bat build web
```

## Verification

Basic checks:

```powershell
python -m py_compile final_data_collection.py final_feature_extraction.py final_training.py tools/generate_env_config.py
cd tinyml_app
C:\Users\Admin\flutter\bin\flutter.bat analyze
```

Before pushing, stage files explicitly. Do not use `git add .`.
