# Copyright (c) 2026 Remco Havenaar / SYNTRIAD Research — MIT License
"""
Pytest configuration: test markers for Engine vNext.

Usage:
    pytest                          # runs unit + integration (default)
    pytest -m unit                  # fast unit tests only (<2s)
    pytest -m integration           # integration tests (<1m)
    pytest -m exhaustive            # heavy exhaustive tests (20m+)
    pytest -m "not exhaustive"      # skip exhaustive (default for dev)
"""

import os
import sys

import pytest

# Ensure subdirectories are importable (engines/ contains standalone discovery engines)
_ROOT = os.path.dirname(os.path.abspath(__file__))
for _subdir in ("engines", "scripts"):
    _path = os.path.join(_ROOT, _subdir)
    if _path not in sys.path:
        sys.path.insert(0, _path)


def pytest_configure(config):
    config.addinivalue_line("markers", "unit: fast unit tests (<2s)")
    config.addinivalue_line("markers", "integration: integration tests (<1m)")
    config.addinivalue_line("markers", "exhaustive: exhaustive nightly tests (20m+)")
