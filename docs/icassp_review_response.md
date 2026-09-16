# ICASSP review response: reproducibility map and open measurements

Written for: whoever picks up the CARD resubmission work.

The 11 September 2026 review of the ICASSP submission raised two points this
repository has to answer rather than the manuscript. W6 asks for the fixed
thresholds and weights to be locatable and for the prompts and configuration to
be released. Q2 asks for the measured speaking rate of CARD output against the
source and against Descript. The CARD--source measurement is complete, while
the Descript rate leg still needs a complete Episode 5 audio export.

Manuscript changes live in the `EEE_199_IEEE` repository, branch
`jd-edits-after-feedback`.

## 1. Where every reported constant lives (W6)

All values below are read from `conf/config.yaml` on this branch. They are fixed
settings applied to every run in the reported evaluation, not per-episode tuning.

| Manuscript, Section 2 | `conf/config.yaml` key | Line | Value |
|---|---|---|---|
| Categorical/dimensional affect blend, `.65` and `.35` | `categorical_weight`, `dimensional_weight` | 232, 233 | 0.65, 0.35 |
| Conditioning-vector intensity cap, the `N_.8` operator | `emotion_vector_max_sum` (affect mapping) | 242 | 0.8 |
| Same cap applied at link time | `max_sum` (summary turn linking) | 282 | 0.8 |
| Lexical recall weight `.45` in Eq. (1) | `lexical_similarity_weight` | 275 | 0.45 |
| Source coverage weight `.25` in Eq. (1) | `coverage_similarity_weight` | 276 | 0.25 |
| Sequence similarity weight `.15` in Eq. (1) | `sequence_similarity_weight` | 277 | 0.15 |
| Relative-position weight `.15` in Eq. (1) | `position_similarity_weight` | 278 | 0.15 |
| Speaker-match multiplier, `1.2` | `same_speaker_bonus` | 272 | 1.2 |
| Support-score gate, `h >= .08` | `min_support_score` | 271 | 0.08 |
| No-overlap fallback gate, `rho >= .6` | `min_sequence_similarity_without_overlap` | 279 | 0.6 |
| `Top3` retained source segments in Eq. (2) | `top_k` | 270 | 3 |
| Affect-confidence floor | `min_emotion_confidence` | 274 | 0.15 |
| Acceptance tolerance, the `.05` in Eq. (3) | `duration_tolerance_ratio` | 411 | 0.05 |
| Prompt-history compaction at 80% occupancy | `trigger_ratio` | 417 | 0.80 |

Prompt templates are versioned under
`src/card_framework/prompts/templates/`. The ablation reported in Table 3 is
produced by `scripts/run_sporc_ablation_study.py`.

The manuscript now commits to releasing the implementation, this configuration,
the prompt templates, and the evaluation scripts on acceptance. Both repositories
are currently private, so someone has to make that call before the camera-ready.

## 2. Speaking rate and pause allocation (W3, Q2)

`scripts/analyze_speaking_rate.py` measures the quantities the review asks for.
It reports speech rate (words per second of waveform) and articulation rate
(words per second of voiced audio) separately, because a system can hold
articulation rate steady and still sound hurried by cutting pauses. Silence comes
from the ffmpeg `silencedetect` filter, so the script needs ffmpeg and ffprobe on
PATH and no Python audio dependency.

Every audio measurement records the input basename and SHA-256 digest. The
script rejects cross-system waveform reuse and an incomplete Descript export
before writing any result.

### Source-slice cross-check

The source baseline needs no audio, because the committed transcript slices
record the measured seconds of every retained span:

```
python scripts/analyze_speaking_rate.py --source-from-slices
```

| Episode | Words | Seconds | Words/sec |
|---|---|---|---|
| Podcast 01 | 1578 | 587.02 | 2.69 |
| Podcast 02 | 1514 | 614.02 | 2.47 |
| Podcast 03 | 1519 | 603.31 | 2.52 |
| Podcast 04 | 1473 | 607.12 | 2.43 |
| Podcast 05 | 1559 | 619.57 | 2.52 |

Mean 2.52 words/sec, or about 151 words per minute. The per-episode totals
reproduce the 587.02--619.57 second excerpt range reported in Section 3, which
is the cross-check that the slice parsing is right. These are span-level figures
with no silence detection, so 2.52 is reported only as a speech rate. Source
articulation rate and pause fraction remain unavailable without the waveforms.

### CARD--source comparison used in the manuscript

The Participant Audio v3 waveform analysis is reproducible without a Descript
root:

```
python scripts/analyze_speaking_rate.py \
    --participant-audio-root "<path>/Participant Audio v3"
```

For the five episode-matched five-minute outputs, CARD averaged 3.4846
words/sec and the source excerpts averaged 2.5228. CARD minus source was
$+0.9617$ words/sec (95% CI $[0.6145,1.3089]$). The articulation-rate
difference was $+0.5599$ words/sec (CI $[0.0164,1.1035]$). Pause fraction was
$-0.1322$ (CI $[-0.2314,-0.0330]$), from means of 0.0222 for CARD and 0.1545
for source.

### Descript rate comparison awaiting Episode 5

`Participant Audio v3` and the five Descript transcript exports were located on
Chris's machine. Input validation found two separate handoff defects before the
measurements could be reported:

- The copied Descript folders labeled Episodes 3-5 contain the byte-identical
  CARD five-minute WAVs. The analysis now rejects this cross-system reuse by
  SHA-256 digest.
- Separate Descript WAV exports were found for Episodes 3-5. Episodes 3 and 4
  are complete. The Episode 5 file contains two digital-silence spans of about
  96 seconds each; only 101.96 of its 303.76 seconds remain voiced at the fixed
threshold. Pairing 985 transcript words with that file would yield an
  implausible 9.66 words per voiced second, so no Descript rate result was
  reported.

The manuscript retains the descriptive CARD--Descript listener-score
comparison from the researchers' verified response CSV. The audio defects here
affect the separate speaking-rate analysis.

Episode 5 must be re-exported with both speakers' audio present. After replacing
that WAV, use the validated folder layout: five `Podcast_XX_5m` directories,
each with one `[SUMMARY] ... .wav` and one
`[SUMMARY] ... - summary transcript.txt` file.

For a future rate comparison after replacing Episode 5:

```
python scripts/analyze_speaking_rate.py \
    --source-from-slices \
    --participant-audio-root "<path>/Participant Audio v3" \
    --descript-root "<path>/Descript 5m references"
```

The run writes `speaking_rate.csv` and `speaking_rate_summary.json` under
`artifacts/speaking_rate/`. The CSV records input basenames and audio digests.
The summary records the project-specific silence threshold. Participant-only
runs carry the episode-paired CARD-minus-source contrasts; a validated full run
also carries CARD-minus-Descript contrasts. Each contrast includes a 95%
Student-t interval.

## 3. Not attempted here

- The paired, balanced listener comparison the review makes its acceptance
  condition (W1, Q1) needs new participants and cannot be recovered from the
  existing exports.
- The 300-second calibration question (Q4) is answered in the manuscript from the
  existing measurements: a leave-one-out offset fixes four episodes and pushes
  the fifth outside the band, so the residual is episode-dependent. Closing it
  properly needs a per-episode signal available before synthesis, which is a
  pipeline change, not an analysis.
- Wider episode sampling (W5) needs pipeline reruns and the 10-minute source
  audio, which is not on this machine either.
