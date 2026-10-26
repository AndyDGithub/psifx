"""Shared runtime constants."""

import os


# Allow overriding the default SAM3 model path without editing source files.
SAM3_PATH = os.environ.get("SAM3_PATH", "facebook/sam3")

# Image encoder that re-identifies people across shot changes during SAM3 tracking.
SAM3_REID_PATH = os.environ.get("SAM3_REID_PATH", "facebook/dinov2-small")
