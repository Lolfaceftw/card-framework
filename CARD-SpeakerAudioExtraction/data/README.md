# data/

This directory contains data files for the audio-splitter project.

## Structure

- `raw/`: Raw, unprocessed audio files
- `processed/`: Processed audio files and intermediate results
- `custom/`: Custom audio files for testing and examples

## Setup

1. Place raw audio files in `data/raw/`
2. Processed outputs will be automatically created in `data/processed/` during processing
3. Custom test files are already provided in `data/custom/`

## Usage

The data directories are referenced in the configuration file (`configs/config.yaml`). You can change the paths there if needed.

## Files

- `custom/PrimeagenLex.wav`: Sample podcast audio for testing
- `custom/the_great_chicken_debate.wav`: Another sample audio file