"""
Shared pytest configuration for all psifx tests.
"""

import sys
from pathlib import Path

# The pose benchmark harness (research/benchmark) is not an installed package:
# its modules import each other as top-level names (``from common import ...``),
# so its directory must be on sys.path for the tests that exercise it.
BENCHMARK_DIR = Path(__file__).resolve().parent.parent / "research" / "benchmark"
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))
