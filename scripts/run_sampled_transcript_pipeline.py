"""Run the sampled-transcript stage-2 pipeline helper."""

from __future__ import annotations

import sys

from card_framework.benchmark.summarizer_critic.sporc.sampled_pipeline import main


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
