# configs/

This directory contains configuration files for the audio-splitter project.

## Files

- `config.yaml`: Main configuration file containing settings for device, data paths, models, diarization parameters, separation settings, and output options.

## Setup

The configuration file is automatically loaded by the application. You can modify the settings in `config.yaml` to customize:

- Device selection (MPS for Apple Silicon, CUDA for NVIDIA GPUs, CPU fallback)
- Model parameters and paths
- Processing thresholds and durations
- Output directories and formats

## Usage

The config is loaded automatically when running the separation or diarization scripts. No manual setup required beyond editing the YAML file if needed.