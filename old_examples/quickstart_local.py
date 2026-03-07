#!/usr/bin/env python3
"""
Quickstart local example.

This is the recommended first script under `examples/`.
It mirrors `python -m gapa demo`, but keeps everything in script form.
"""

from __future__ import annotations

import os
import sys

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from gapa.demo import main as demo_main


if __name__ == "__main__":
    raise SystemExit(demo_main(sys.argv[1:]))
