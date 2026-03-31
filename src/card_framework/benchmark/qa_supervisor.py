"""Backward-compatible QA supervisor wrapper."""

from importlib import import_module
import sys

sys.modules[__name__] = import_module("card_framework.benchmark.qa.supervisor")
