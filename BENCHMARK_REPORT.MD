# Benchmark Report

## Run
- Run directory: `benchmarks/runs/20260213T061510Z`
- Summary file: `benchmarks/runs/20260213T061510Z/metrics_summary.json`
- Manifest: `benchmarks/manifests/voice_clone_manifest_20260213T061502Z.json`
- UTC timestamp: `2026-02-13T06:18:29Z`

## Scope
- Task: voice cloning benchmark for IndexTTS2 outputs.
- Dataset size: 10 items, 3 speakers.
- Speaker distribution:
  - `SPEAKER_00`: 8 items
  - `SPEAKER_01`: 1 item
  - `SPEAKER_02`: 1 item
- Prompt/reference overlap check:
  - `prompt_reference_overlap_rows = 0`
  - Prompt and holdout clips are distinct files per speaker.

## Metrics

| Metric | Value | Notes |
|---|---:|---|
| `ss_cosine_mean` | 0.9533 | High similarity on this dataset |
| `ss_cosine_std` | 0.0123 | Low spread |
| `ss_cosine_ci95` | [0.9463, 0.9617] | Narrow due to low variance |
| `ss_cosine_macro_mean` | 0.9578 | Macro over speakers |
| `top1_speaker_acc` | 1.0000 | 10/10 correct |
| `top1_speaker_macro_acc` | 1.0000 | Perfect in this run |
| `asv_eer` | 0.0000 | Perfect separation in this setup |
| `asv_positive_pairs` | 66 | Pair count in EER calc |
| `asv_negative_pairs` | 34 | Pair count in EER calc |
| `text_metric_status` | ok | ASR-based text metrics computed |
| `wer_mean` | 0.0343 | Low word error rate |
| `wer_ci95` | [0.0071, 0.0614] | Small-sample CI |
| `cer_mean` | 0.0071 | Low character error rate |
| `cer_ci95` | [0.0016, 0.0129] | Small-sample CI |

## Inferences
- The run is a strong **internal quality signal**: voice similarity and intelligibility are both high on the sampled set.
- The results are **not obviously leakage-driven** by direct prompt/reference file equality because overlap checks pass and files differ.
- Some headline values are likely **optimistic for generalization**:
  - Very small dataset (`n=10`).
  - Strong class imbalance (8/10 from one speaker).
  - Repeated prompt/holdout clips per speaker reduce task diversity.
- `top1=1.0` and `EER=0.0` should be interpreted as “perfect on this narrow setup,” not as broad production-level guarantees.

## Limitations
- Small sample size limits statistical confidence.
- Speaker imbalance can inflate aggregate metrics toward dominant-speaker behavior.
- Limited prompt/reference diversity per speaker reduces robustness stress.
- No cross-session/environment robustness split is shown in this run.
- No confidence calibration or threshold analysis beyond aggregate EER is reported.

## Recommended Next Steps
1. Re-run with a balanced manifest (equal items per speaker).
2. Increase holdout diversity (multiple clips per speaker, varied conditions).
3. Report per-speaker metrics alongside aggregate metrics as primary results.
4. Track trend across multiple runs/seeds rather than a single run snapshot.

