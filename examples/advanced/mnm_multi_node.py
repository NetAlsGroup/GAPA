#!/usr/bin/env python3
"""
Recommended advanced example for heterogeneous MNM execution.
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
    base_args = ["--dataset", "Circuit", "--mode", "mnm"]
    if argv:
        base_args.extend(argv)
    return run_sixdst.main(base_args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
