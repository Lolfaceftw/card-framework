# ICASSP review response: reproducibility map and open measurements

Written for: whoever picks up the CARD resubmission work.

The 11 September 2026 review of the ICASSP submission raised two points this
repository has to answer rather than the manuscript. W6 asks for the fixed
thresholds and weights to be locatable and for the prompts and configuration to
be released. Q2 asks for the measured speaking rate of CARD output against the
source and against Descript. This file covers the first and sets up the second.

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

Descript is the useful control here: it edits original speaker audio instead of
resynthesizing it, so its articulation rate is the source rate by construction. A
CARD-minus-source gap with no matching Descript-minus-source gap would place the
effect in Card 3 synthesis rather than in content selection.

### Already measured

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
with no silence detection, so treat 2.52 as a speech rate and as a lower bound on
source articulation rate.

### Still blocked

The CARD and Descript halves need artifacts that are not in this repository and
are not on JD's machine:

- `Participant Audio v3`, the per-episode `Podcast_XX_<duration>` folders holding
  the `[SOURCE]` and `[SUMMARY]` transcript and audio pairs. Same layout that
  `scripts/run_participant_audio_gemini_judge.py` already consumes.
- The five Descript five-minute exports, WAV plus edited transcript, as described
  in `datasets/sporc/human_eval_assets_10m/descript_5m_sota_reference_prompts.md`.

With both present:

```
python scripts/analyze_speaking_rate.py \
    --source-from-slices \
    --participant-audio-root "<path>/Participant Audio v3" \
    --descript-root "<path>/Descript 5m references"
```

The run writes `speaking_rate.csv` and `speaking_rate_summary.json` under
`artifacts/speaking_rate/`. The summary carries the episode-paired CARD-minus-
source and CARD-minus-Descript contrasts for articulation rate and pause
fraction, each with a 95% interval, which is the form the review asks for.

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
