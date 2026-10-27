"""``psifx`` module."""

import os

# pyannote.audio 4 sends usage metrics to pyannote.ai by default (model name, audio duration);
# psifx turns them off for research recordings. Set PYANNOTE_METRICS_ENABLED=1 to opt in.
os.environ.setdefault("PYANNOTE_METRICS_ENABLED", "0")

__version__ = "0.2.3"
