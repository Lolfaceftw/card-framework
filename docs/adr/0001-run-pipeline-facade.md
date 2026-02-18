# ADR 0001: Split `run_pipeline.py` into facade + orchestrator module

## Context
`audio2script_and_summarizer/run_pipeline.py` had grown to several thousand lines and mixed CLI entrypoint concerns with helper and orchestration logic. This made targeted maintenance and incremental extraction risky.

## Decision
1. Keep `audio2script_and_summarizer/run_pipeline.py` as a compatibility facade.
2. Move the concrete implementation to `audio2script_and_summarizer/pipeline/orchestrator.py`.
3. Extract reusable pure helpers into `audio2script_and_summarizer/pipeline/helpers.py`.
4. Extract dashboard rendering/state into `audio2script_and_summarizer/pipeline/dashboard.py`.
5. Extract subprocess stage execution into `audio2script_and_summarizer/pipeline/stage_runner.py`.
6. Extract large mode branches from orchestrator main into `audio2script_and_summarizer/pipeline/flows.py`.
7. Keep backward compatibility by aliasing the runtime `run_pipeline` module to the orchestrator module object and re-exporting compatibility symbols.

## Alternatives Considered
1. Keep all logic in `run_pipeline.py` and only reorganize functions in-place.
2. Move code to a new module with static re-exports only (without module aliasing).

## Consequences
1. `run_pipeline.py` is now a thin entrypoint/facade (<1000 lines target achieved).
2. Existing imports and monkeypatch-based tests remain compatible via facade aliasing and compatibility bridges.
3. Dashboard and subprocess concerns are now independently evolvable modules.
4. Main flow routing in orchestrator is now thinner and delegates mode execution to `pipeline/flows.py`.
5. Further extraction can proceed incrementally inside `pipeline/flows.py` and `pipeline/dashboard.py` without breaking callers.
