# Descript 5-Minute Blind Reference Prompt

This handoff is for creating one 5-minute Descript reference artifact for each
of the five human-judged SPoRC episodes.

## Academic-Integrity Protocol

Use a blind, source-only workflow.

- Feed Descript the episode audio slice only.
- Let Descript create its own transcript from that audio.
- Do not feed Descript the Card Framework generated 5-minute WAVs.
- Do not feed Descript the `transcript_slices/*.txt` files as prompt context;
  those files include selection metadata and rationale notes that could bias the
  reference.
- Use `transcript_slices/*.txt` only after generation for audit or traceability.
- Use `voice_samples/` only if a Descript feature explicitly requires voice
  references, and do not use those samples to add content.

## Universal Prompt

Use this same prompt for every episode.

```text
Create a high-quality 5-minute reference edit from the imported podcast audio.

Use only the source material present in this Descript project. Do not use outside knowledge, inferred background context, file names, prior summaries, comparator outputs, or any information not supported by the imported audio and Descript-generated transcript.

Target runtime: 5 minutes. Acceptable range: 4:45 to 5:15.

Edit the source into a faithful, listener-ready reference that preserves the episode's main ideas, speaker roles, tone, and conversational flow. Keep the strongest original speaker audio when possible. Remove only material that is redundant, off-track, confusing, or unnecessary for a coherent 5-minute version.

Do not invent narration, facts, transitions, quotes, or conclusions. If a short bridge is required for continuity, keep it neutral and directly supported by the surrounding source audio.

The final reference should stand on its own for a listener who has not heard the full source, while remaining fully traceable to the imported audio.

Export:
1. A WAV file of the final 5-minute reference.
2. The final edited transcript/script.
```

## Episode Inputs

### Episode 01

Feed Descript:

- `datasets/sporc/human_eval_assets_10m/audio_slices/01 - The Lion, the Lamb & the Meadow (Chap. 12-13) - 10m audio slice.mp3`

Do not feed as prompt context:

- `datasets/sporc/human_eval_assets_10m/transcript_slices/01 - The Lion, the Lamb & the Meadow (Chap. 12-13) - transcript slices.txt`
- `artifacts/sporc_human_eval_stage2_sweeps/all_15_final_audio_human_readable/01 - The Lion, the Lamb & the Meadow (Chap. 12-13) - 5m.wav`

Optional voice references only:

- `datasets/sporc/human_eval_assets_10m/voice_samples/01 - The Lion, the Lamb & the Meadow (Chap. 12-13)/`

### Episode 02

Feed Descript:

- `datasets/sporc/human_eval_assets_10m/audio_slices/02 - Spirit Adrift's Nate Garrett on the Love That Saved His Life - 10m audio slice.mp3`

Do not feed as prompt context:

- `datasets/sporc/human_eval_assets_10m/transcript_slices/02 - Spirit Adrift's Nate Garrett on the Love That Saved His Life - transcript slices.txt`
- `artifacts/sporc_human_eval_stage2_sweeps/all_15_final_audio_human_readable/02 - Spirit Adrift's Nate Garrett on the Love That Saved His Life - 5m.wav`

Optional voice references only:

- `datasets/sporc/human_eval_assets_10m/voice_samples/02 - Spirit Adrift's Nate Garrett on the Love That Saved His Life/`

### Episode 03

Feed Descript:

- `datasets/sporc/human_eval_assets_10m/audio_slices/03 - 079 The United State of St Kitts and Nevis ft Nekirah Nicholls - 10m audio slice.mp3`

Do not feed as prompt context:

- `datasets/sporc/human_eval_assets_10m/transcript_slices/03 - 079 The United State of St Kitts and Nevis ft Nekirah Nicholls - transcript slices.txt`
- `artifacts/sporc_human_eval_stage2_sweeps/all_15_final_audio_human_readable/03 - 079 The United State of St Kitts and Nevis ft Nekirah Nicholls - 5m.wav`

Optional voice references only:

- `datasets/sporc/human_eval_assets_10m/voice_samples/03 - 079 The United State of St Kitts and Nevis ft Nekirah Nicholls/`

### Episode 04

Feed Descript:

- `datasets/sporc/human_eval_assets_10m/audio_slices/04 - Ep 6 - Polite & Passive Aggression - 10m audio slice.mp3`

Do not feed as prompt context:

- `datasets/sporc/human_eval_assets_10m/transcript_slices/04 - Ep 6 - Polite & Passive Aggression - transcript slices.txt`
- `artifacts/sporc_human_eval_stage2_sweeps/all_15_final_audio_human_readable/04 - Ep 6 - Polite & Passive Aggression - 5m.wav`

Optional voice references only:

- `datasets/sporc/human_eval_assets_10m/voice_samples/04 - Ep 6 - Polite & Passive Aggression/`

### Episode 05

Feed Descript:

- `datasets/sporc/human_eval_assets_10m/audio_slices/05 - Video Game Logic Episode 195 - Take Two Takedown - 10m audio slice.mp3`

Do not feed as prompt context:

- `datasets/sporc/human_eval_assets_10m/transcript_slices/05 - Video Game Logic Episode 195 - Take Two Takedown - transcript slices.txt`
- `artifacts/sporc_human_eval_stage2_sweeps/all_15_final_audio_human_readable/05 - Video Game Logic Episode 195 - Take Two Takedown - 5m.wav`

Optional voice references only:

- `datasets/sporc/human_eval_assets_10m/voice_samples/05 - Video Game Logic Episode 195 - Take Two Takedown/`
