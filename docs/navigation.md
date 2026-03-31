# Repository Navigation

Use this map when you are orienting in the repo root.

- `src/card_framework/`: first-party product code.
- `tests/`: automated coverage that broadly mirrors the shipped package.
- `coder_docs/`: repo-local policy, workflow, and maintenance guidance.
- `docs/`: short human-facing docs such as config guidance and this navigation map.
- `datasets/`: tracked benchmark handoff packages and archived review manifests.
- `research/`: the CARD paper, LaTeX manuscript, and bibliography.
- `scripts/`: repo-local helpers and experiments.
- `scripts/manual/`: manual probes that should not live in the shipped package namespace.
- `artifacts/`, `checkpoints/`, and similar output roots: runtime-generated or operator-managed state.
- `src/card_framework/_vendor/`: vendored third-party code; do not treat it as the main navigation surface for first-party changes.

If you are changing runtime behavior, start in `src/card_framework/` and use `README.md` plus `coder_docs/codebase_guide.md` for the high-level path through the system.
