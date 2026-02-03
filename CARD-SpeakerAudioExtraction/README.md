# audio-splitter

Audio separation toolkit for extracting individual speaker tracks from multi-speaker audio recordings.

## Features

- **Targeted Separation (CARD Methodology)**: Intelligently separates overlapping speech using SepFormer only where needed, with speaker embeddings for accurate assignment
- **Diarization-Guided Separation**: Extracts speaker-specific audio using timestamps from diarization JSON files
- **Blind Source Separation**: ML-based separation using SepFormer, DPRNN-TasNet, or Conv-TasNet models
- Handles overlapping speech with configurable strategies
- Supports timing preservation or compact output modes

## Installation

```bash
pip install -r requirements_from_venv.txt
```

## Usage

### Targeted Separation (CARD Methodology) - ⭐ RECOMMENDED

The targeted separation mode implements the complete CARD methodology for high-quality speaker separation. It:

1. **Extracts enrollment embeddings** from clean (non-overlapping) speech segments using ECAPA-TDNN
2. **Directly extracts non-overlapping segments** (artifact-free, no ML processing needed)
3. **Runs SepFormer only on overlapping regions** (memory efficient, targeted approach)
4. **Assigns separated sources to speakers** using cosine similarity with enrollment embeddings
5. **Concatenates segments with cross-fade** to avoid clicks and artifacts

This approach is ideal for:
- Multi-speaker recordings with overlapping speech
- Voice cloning applications (produces clean, speaker-pure audio)
- Podcast and interview separation
- Any scenario where you have diarization data

```bash
# Basic usage - auto mode defaults to targeted when diarization is provided
python src/separation/main.py \
    --audio podcast.wav \
    --diarization diarization.json \
    --output-dir outputs/

# Explicit targeted mode with custom settings
python src/separation/main.py \
    --audio podcast.wav \
    --diarization diarization.json \
    --mode targeted \
    --output-dir outputs/ \
    --crossfade-ms 25 \
    --similarity-threshold 0.7 \
    --save-embeddings

# With custom enrollment and overlap settings
python src/separation/main.py \
    --audio podcast.wav \
    --diarization diarization.json \
    --mode targeted \
    --output-dir outputs/ \
    --enrollment-min-duration 3.0 \
    --enrollment-max-duration 6.0 \
    --overlap-padding 0.5
```

#### Targeted Mode Options

| Option | Description | Default |
|--------|-------------|---------|
| `--crossfade-ms` | Cross-fade duration in milliseconds | `25.0` |
| `--similarity-threshold` | Cosine similarity threshold for speaker assignment | `0.7` |
| `--save-embeddings` | Save enrollment embeddings to output directory | `False` |
| `--enrollment-min-duration` | Minimum enrollment snippet duration (seconds) | `3.0` |
| `--enrollment-max-duration` | Maximum enrollment snippet duration (seconds) | `6.0` |
| `--overlap-padding` | Padding for overlap windows (seconds) | `0.5` |

#### Targeted Mode Output

Output files are saved to `<output-dir>/<audio-name>_separation/`:

```
outputs/
├── podcast_separation/
│   ├── SPEAKER_00.wav                  # Speaker-pure audio
│   ├── SPEAKER_01.wav
│   ├── enrollment_embeddings.npy       # Saved embeddings (if --save-embeddings)
│   └── separation_report.json          # Statistics and metadata
```

**separation_report.json** contains:
- Total duration and per-speaker duration
- Number of overlaps detected and processed
- Uncertain assignments (similarity below threshold)
- Complete separation statistics

### Diarization-Guided Separation

When you have diarization data (JSON file with speaker timestamps), use diarization-guided separation:

```bash
# Basic usage
python src/separation/main.py \
    --audio interview.wav \
    --diarization diarization.json \
    --output-dir outputs/ \
    --mode diarization-guided \
    --handle-overlap skip

# Compact mode (no silence gaps)
python src/separation/main.py \
    --audio interview.wav \
    --diarization diarization.json \
    --output-dir outputs/ \
    --mode diarization-guided \
    --compact
```

#### Diarization JSON Format

The diarization JSON should be an array of segments with start/end times and speaker IDs:

```json
[
  {
    "start": 2.98,
    "end": 8.84,
    "speaker": "SPEAKER_00",
    "text": "Hello, how are you?"
  },
  {
    "start": 9.12,
    "end": 15.45,
    "speaker": "SPEAKER_01",
    "text": "I'm doing great, thanks!"
  }
]
```

#### Overlap Handling Strategies

- `--handle-overlap skip`: Leave silence in overlapping regions (default)
- `--handle-overlap mix`: Include overlapping audio for all speakers
- `--handle-overlap both`: Same as mix - include audio for all involved speakers

#### Output Modes

- `--preserve-timing`: Maintain original timestamps with silence gaps (default)
- `--compact`: Concatenate speaker segments without gaps

### Blind Source Separation

When diarization data is not available, use ML-based blind separation:

```bash
# Using SepFormer (recommended for long audio)
python src/separation/main.py \
    --audio interview.wav \
    --output-dir outputs/ \
    --mode blind \
    --model sepformer

# Using DPRNN-TasNet
python src/separation/main.py \
    --audio interview.wav \
    --output-dir outputs/ \
    --mode blind \
    --model dprnn-tasnet
```

## Python API

### Targeted Separation (CARD Methodology)

```python
from src.separation import TargetedSpeakerSeparator

# Initialize the separator
separator = TargetedSpeakerSeparator(
    sample_rate=16000,
    similarity_threshold=0.7,
    crossfade_ms=25.0,
    overlap_padding_s=0.5,
    enrollment_duration_range=(3.0, 6.0),
    device='auto'
)

# Separate and save
saved_paths = separator.process_and_save(
    audio_path='podcast.wav',
    diarization_path='diarization.json',
    output_dir='outputs/',
    save_embeddings=True
)

# Or get the separated audio directly
speaker_audio, stats = separator.separate(
    audio_path='podcast.wav',
    diarization_path='diarization.json'
)
# speaker_audio: dict mapping speaker IDs to numpy arrays
# stats: dict with separation statistics
```

### Diarization-Guided Separation

```python
from src.separation import DiarizationGuidedSeparator, separate_with_diarization

# Using the class directly
separator = DiarizationGuidedSeparator(
    sample_rate=16000,
    handle_overlap='skip',
    preserve_timing=True,
    min_segment_duration=0.5
)

saved_paths = separator.process_and_save(
    audio_path='interview.wav',
    diarization_path='diarization.json',
    output_dir='outputs/'
)

# Using the convenience function
saved_paths = separate_with_diarization(
    audio_path='interview.wav',
    diarization_path='diarization.json',
    output_dir='outputs/',
    handle_overlap='skip',
    preserve_timing=True
)
```

### Blind Source Separation

```python
from src.separation import SpeechSeparator

separator = SpeechSeparator(model_name='sepformer', device='auto')
separator.load_model()

saved_paths = separator.process_and_save(
    audio_path='interview.wav',
    output_dir='outputs/'
)
```

## CLI Options

| Option | Description | Default |
|--------|-------------|---------|
| `--audio` | Path to input audio file | Required |
| `--diarization` | Path to diarization JSON file | None |
| `--output-dir` | Output directory for separated audio | `outputs/` |
| `--mode` | Separation mode: `targeted`, `diarization-guided`, `blind`, or `auto` | `auto` |
| `--device` | Device: `cpu`, `cuda`, or `auto` | `auto` |

### Targeted Mode Options

| Option | Description | Default |
|--------|-------------|---------|
| `--crossfade-ms` | Cross-fade duration (ms) | `25.0` |
| `--similarity-threshold` | Speaker assignment threshold | `0.7` |
| `--save-embeddings` | Save enrollment embeddings | `False` |
| `--enrollment-min-duration` | Min enrollment duration (s) | `3.0` |
| `--enrollment-max-duration` | Max enrollment duration (s) | `6.0` |
| `--overlap-padding` | Overlap window padding (s) | `0.5` |

### Diarization-Guided Mode Options

| Option | Description | Default |
|--------|-------------|---------|
| `--handle-overlap` | Overlap strategy: `skip`, `mix`, or `both` | `skip` |
| `--preserve-timing` | Maintain original timestamps | `True` |
| `--compact` | Concatenate segments without gaps | `False` |
| `--min-segment-duration` | Minimum segment duration (seconds) | `0.0` |

### Blind Mode Options

| Option | Description | Default |
|--------|-------------|---------|
| `--model` | ML model: `sepformer`, `dprnn-tasnet`, `conv-tasnet` | `sepformer` |
| `--chunk-duration` | Chunk duration for long audio (s) | `30.0` |
| `--overlap` | Overlap between chunks (s) | `5.0` |
| `--no-chunking` | Disable automatic chunking | `False` |

## Output

Output files are saved to `<output-dir>/<audio-name>_separation/`:

- **Targeted mode**: `SPEAKER_00.wav`, `SPEAKER_01.wav`, `enrollment_embeddings.npy`, `separation_report.json`
- **Diarization-guided mode**: `SPEAKER_00.wav`, `SPEAKER_01.wav`, etc.
- **Blind mode**: `speaker_00.wav`, `speaker_01.wav`, etc.

## Technical Details

### Targeted Separation Pipeline

1. **Enrollment Phase**: Selects 3-6 second clean (non-overlapping) segments per speaker and extracts ECAPA-TDNN embeddings
2. **Direct Extraction**: Non-overlapping segments are extracted directly using NumPy slicing (no ML processing)
3. **Overlap Processing**: For each overlap region:
   - Extracts window with 0.5s padding on each side
   - Runs SepFormer to separate sources
   - Computes embeddings for each separated source
   - Assigns sources to speakers using cosine similarity
   - Trims padding from assigned sources
4. **Track Construction**: Concatenates segments per speaker with 25ms linear cross-fade
5. **Output Generation**: Saves speaker-pure WAV files and metadata

### Models Used

- **ECAPA-TDNN**: Speaker embedding extraction (`speechbrain/spkrec-ecapa-voxceleb`)
  - 192-dimensional embeddings
  - L2 normalized for cosine similarity
- **SepFormer**: Speech source separation (`speechbrain/sepformer-wsj02mix`)
  - Only applied to overlapping regions
  - Handles variable number of sources

### Cross-fade

- Linear fade curves using `np.linspace`
- Default: 25ms (configurable 10-50ms range)
- Applied at all segment boundaries to avoid clicks