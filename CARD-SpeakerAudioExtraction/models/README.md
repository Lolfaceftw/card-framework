# models/

This directory contains machine learning models used by the audio-splitter project.

## Structure

- `sepformer/`: Directory for SepFormer speech separation model (downloaded automatically)

## Setup

Models are downloaded automatically when first used. No manual setup required.

The following models are used:
- **SepFormer**: For blind source separation (`speechbrain/sepformer-wsj02mix`)
- **ECAPA-TDNN**: For speaker embedding extraction (`speechbrain/spkrec-ecapa-voxceleb`)
- **Whisper**: For transcription (optional, `base` model)

## Usage

Models are loaded automatically by the separation and diarization modules. Ensure you have sufficient disk space and internet connection for initial downloads.

## Configuration

Model settings can be configured in `configs/config.yaml` under the `models` section.