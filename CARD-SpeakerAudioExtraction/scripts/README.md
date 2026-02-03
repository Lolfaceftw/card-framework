# scripts/

This directory contains utility scripts for setup, testing, and maintenance of the audio-splitter project.

## Files

- `fix_dependencies.sh`: Script to fix dependency version conflicts (especially NumPy/PyTorch compatibility)
- `test_targeted_separation.py`: Test script to verify targeted speaker separation functionality

## Setup

Scripts can be run directly from the command line. Ensure you're in the project root directory.

## Usage

### Fix Dependencies

```bash
# Run the dependency fix script
./scripts/fix_dependencies.sh
```

This script:
- Activates the virtual environment
- Installs compatible NumPy version (< 2.0)
- Updates PyTorch, Transformers, and SpeechBrain to compatible versions
- Tests the installations

### Test Targeted Separation

```bash
# Run the test script
python scripts/test_targeted_separation.py
```

This script:
- Creates synthetic test data
- Tests enrollment embedding extraction
- Tests speaker assignment
- Tests cross-fade functionality
- Verifies the complete targeted separation pipeline

## Notes

- The fix_dependencies.sh script is specific to the development environment and may need adjustment for different setups
- Test scripts use synthetic data and don't require actual audio files