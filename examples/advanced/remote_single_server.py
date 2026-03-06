#!/usr/bin/env python3
"""
Recommended advanced example for remote single-server execution.

This keeps compatibility with `examples/run_sixdst.py` while exposing a clearer
entry name for users browsing the examples tree.
"""

from __future__ import annotations

import os
import sys

_script_dir = os.path.dirname(os.path.abspath(__file__))
_examples_root = os.path.dirname(_script_dir)
if _examples_root not in sys.path:
    sys.path.insert(0, _examples_root)

import run_sixdst


def main(argv=None):
    base_args = ["--dataset", "ForestFire_n500", "--mode", "m"]
    if argv:
        base_args.extend(argv)
    return run_sixdst.main(base_args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
