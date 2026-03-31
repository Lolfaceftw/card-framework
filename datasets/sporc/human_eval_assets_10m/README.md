SPoRC 10-minute human-eval handoff package.

Contents:
- `transcript_slices/`: one human-readable transcript-slice text file per episode, based on the LLM-selected segments that were used to cut the 10-minute audio.
- `audio_slices/`: one copied 10-minute sliced MP3 per episode with human-readable filenames that match the transcript-slice numbering.
- `voice_samples/`: one folder per episode containing copied speaker-sample WAVs plus the original `manifest.json` from `episode_assets`.
- `package_summary.json`: machine-readable summary of the copied files and any missing voice samples.

Notes:
- No voice samples were missing when this package was assembled.
- Episodes 2 and 3 use bootstrap speaker remapping in their copied `manifest.json`.
- The copied WAV filenames follow the logical speaker labels from the manifest, so they are easier to match against the transcript assets.
