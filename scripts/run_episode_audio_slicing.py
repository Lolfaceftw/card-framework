"""Run the SPoRC episode-assets audio-slicing helper."""

from __future__ import annotations

import sys

from card_framework.benchmark.summarizer_critic.sporc.episode_audio_slicing import (
    main,
)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
