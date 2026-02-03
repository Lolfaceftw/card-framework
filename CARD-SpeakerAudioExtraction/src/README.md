# src/

This directory contains the main source code for the audio-splitter project.

## Structure

- `diarization/`: Speaker diarization functionality
- `separation/`: Audio source separation functionality
- `__init__.py`: Package initialization

## Setup

The source code is automatically imported when running scripts from the project root. No additional setup required.

## Usage

Import modules as needed:

```python
from src.separation import TargetedSpeakerSeparator
from src.diarization import DiarizationProcessor
```

## Modules

### Diarization

Handles speaker diarization using various methods including:
- Pre-computed diarization files (JSON/RTTM)
- Automatic diarization (if implemented)

### Separation

Implements multiple separation strategies:
- **Targeted Separation**: CARD methodology using enrollment embeddings
- **Diarization-Guided**: Uses diarization timestamps for separation
- **Blind Separation**: ML-based separation without prior knowledge

## Dependencies

All dependencies are listed in `requirements.txt` and installed via `pip install -r requirements.txt`.