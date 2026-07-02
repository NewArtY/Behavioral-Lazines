"""Pytest bootstrap: make the root-level modules importable.

The reproducibility package uses a flat layout -- ``laziness.py`` and
``baselines.py`` live at the repo root, not inside a package.  Inserting the
directory of this ``conftest.py`` onto ``sys.path`` lets the tests do
``import laziness`` / ``import baselines`` regardless of the invocation cwd.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
