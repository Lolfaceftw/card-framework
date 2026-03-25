# Stage-4 Interjector Timing Refinement Experiments

## 2026-03-25 - Implementation Pass 1

### Goal
- Reduce delayed or sloppy stage-4 overlaps.
- Prevent an interjection from running into the interjector speaker's own next full turn.
- Keep stage 4 grounded in known stage-3 turn text plus rendered turn audio, not a fresh ASR pass.

### Code Changes
- Added actual-duration fit checks for interjection overlays in `src/card_framework/audio_pipeline/interjector.py`.
- Added a repair loop that re-prompts the interjector LLM with an overlap budget when the first rendered candidate is too long.
- Added anchor-window realignment retry when full-turn alignment falls back to uniform token spacing.
- Added richer stage-4 manifest telemetry:
  - timing source
  - legal overlap end time
  - available overlap budget
  - repair attempt count
  - per-attempt duration history
- Added interjector config wiring for timing refinement and repair settings in:
  - `conf/config.yaml`
  - `src/card_framework/config/config.yaml`
  - `src/card_framework/audio_pipeline/factory.py`
- Added the repair prompt template at `src/card_framework/prompts/templates/interjector_repair.jinja2`.

### Validation
- Focused tests and lint run:
  - `uv run pytest tests/audio_pipeline/test_interjector.py tests/audio_pipeline/test_factory.py`
  - `uv run ruff check src/card_framework/audio_pipeline/interjector.py src/card_framework/audio_pipeline/factory.py tests/audio_pipeline/test_interjector.py tests/audio_pipeline/test_factory.py`
- Result: passing after fixing one missing `event_bus` import.

### Critic Review Round 1
- Critic findings:
  - narrow late-anchor windows still bypassed repair because stage 4 rejected them before the repair loop
  - no-cap repair could hang forever on a repeated no-progress state
  - rejected repairs were not persisted for later timing analysis
- Follow-up fixes:
  - allowed any positive overlap window to reach repair
  - added state-exhaustion termination based on `(text, timing_variant)` instead of relying on an arbitrary numeric cap
  - added rejected interjection telemetry to the stage-4 manifest and run result
  - added regressions for narrow-window repair and recorded rejection paths

### Critic Review Round 2
- Critic finding:
  - `repair_max_attempts` was not honored on the no-legal-window branch
- Follow-up fix:
  - enforced the configured attempt limit before advancing further in the no-window repair path
  - added a regression for bounded repair with no legal overlap window

### Critic Review Final
- Final critic result: no material flaws remaining in the updated stage-4 change set.

## 2026-03-25 - Sweep Driver Fix For Runtime Validation

### Problem
- The first real 5-minute DeepSeek validation did not reach stage 4. It failed on the first summarizer call with DeepSeek `400 Model Not Exist`.
- Root cause: `scripts/run_sporc_human_eval_stage2_sweep.py` still defaulted `--llm-model` to `gemini-3.1-pro-preview` even when the caller selected `--llm-profile deepseek_chat`, so the container overrides pinned the DeepSeek profile to a Gemini model id.

### Fix
- Made `--llm-model` optional instead of always defaulting to Gemini.
- Added `_resolve_llm_model(...)` so the sweep inherits `profiles.llm.<profile>.model` from `conf/config.yaml` unless the caller explicitly overrides `--llm-model`.
- Added a controlled error path for unknown `--llm-profile` values instead of letting OmegaConf raise an internal attribute error.

### Validation
- Focused tests and lint run:
  - `uv run pytest tests/scripts/test_run_sporc_human_eval_stage2_sweep.py`
  - `uv run ruff check scripts/run_sporc_human_eval_stage2_sweep.py tests/scripts/test_run_sporc_human_eval_stage2_sweep.py`
- Result: passing after the unknown-profile regression fix.
- Critic review result: no material flaws remaining in the sweep-script model-resolution slice.

## 2026-03-25 - Real GPU Runtime Validation

### Run
- Randomly chosen episode: `02 - Spirit Adrift's Nate Garrett on the Love That Saved His Life`
- Target duration: `5m`
- Output root:
  - `artifacts/sporc_human_eval_stage2_sweeps/interjector_timing_validation_ep2_5m_20260325_132744`
- Command:
  - `uv run python scripts/run_sporc_human_eval_stage2_sweep.py --output-root ... --episode-indices 2 --duration-presets 5m --llm-profile deepseek_chat --skip-setup --source-separate-voice-samples --voice-sample-separation-device cuda`

### Monitoring
- In-shell 30-second monitoring loop tracked:
  - `docker ps`
  - `docker stats --no-stream`
  - `nvidia-smi`
  - host RAM snapshots
  - `run_summary.json` status
- Observed peaks during the successful run:
  - container RAM: about `5.182 GiB / 7.754 GiB`
  - GPU memory: about `9.578 GiB / 12.288 GiB`
  - host RAM used: about `20.58 GiB / 31.92 GiB`
- Final exit code: `0`

### Output
- Final stage-3 duration: `306.594s`
- Final stage-4 duration: `306.594s`
- Duration tolerance result for the `300s` target: within the `5%` band
- Summary turns: `18`
- Interjection artifacts accepted: `13`
- Interjection artifacts rejected: `1`

### Timing Analysis
- Every accepted interjection used `timing_source=forced_alignment`.
- Accepted timing variants:
  - `default`: `12`
  - `tightened`: `1`
- No accepted artifact used uniform fallback timing in this run.
- No accepted artifact had `speaker == host speaker`.
- No accepted artifact ran past the interjecting speaker's next full turn start.
- Repairs were exercised on tight windows:
  - host turn `7`: first candidate was too long, repaired from `Yeah, that's wild.` to `Yeah, wild.`
  - host turn `11`: required four repair attempts before the final `Mm.` fit
- One candidate was rejected correctly:
  - host turn `13`
  - failure: `no_legal_overlap_window`
  - timing source: `forced_alignment`
  - result: preserved safety instead of forcing an impossible overlap

### Close Calls Worth Noting
- Two accepted overlaps landed close to the next full turn by the same speaker:
  - host turn `8`: `201 ms` before the next turn
  - host turn `11`: `285 ms` before the next turn
- These are still valid under the current guard because they stop before the legal overlap end, which itself already includes the configured turn-end guard. They are good candidates for future tuning if we want even more conversational breathing room, but they did not violate the new self-overlap rule in this run.

### Assessment
- The main behavioral goal was met in this validation:
  - no self-interjections
  - no accepted overlap that bleeds into the same speaker's own next full turn
  - repair loop engaged on narrow windows instead of silently clipping or accepting late collisions
- Based on this run, no additional code change was required immediately after the first runtime validation pass. The remaining tuning question is whether to increase the end-guard margin for very tight but still legal overlaps.
