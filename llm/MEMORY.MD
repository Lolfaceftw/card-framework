# Voice-Cloner-and-Interjector Memory

> Last updated: 2026-02-03 19:56 (UTC+8)

## Voice Sample Locations

Voice samples for testing podcast generation are located in:
- `voice-cloner-and-interjector/examples/`
- Available files: `voice_01.wav` through `voice_12.wav`, `speaker_0.mp3` through `speaker_2.mp3`

## Example JSON Format for Podcast

```json
{
  "speaker": "Host",
  "voice_sample": "examples/voice_01.wav",
  "use_emo_text": true,
  "emo_text": "Warm welcoming slightly thoughtful.",
  "emo_alpha": 0.6,
  "text": "Welcome to the show."
}
```

## Logging

- Logger utility: `tools/podcast/logger.py`
- Default level: INFO (set `LOG_LEVEL=DEBUG` env var for verbose output)
- LLM responses logged at DEBUG level to avoid flooding
