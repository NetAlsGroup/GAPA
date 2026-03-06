from __future__ import annotations

import argparse

from gapa import demo as demo_module


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gapa",
        description="GAPA command line entrypoint.",
    )
    subparsers = parser.add_subparsers(dest="command")

    demo_parser = subparsers.add_parser(
        "demo",
        help="Run the official quickstart demo.",
        description="Run the official quickstart demo on a built-in graph.",
    )
    for action in demo_module.build_demo_parser()._actions:
        if action.dest == "help":
            continue
        demo_parser._add_action(action)
    demo_parser.set_defaults(handler=_handle_demo)
    return parser


def _handle_demo(args: argparse.Namespace) -> int:
    demo_args = []
    if args.graph:
        demo_args.extend(["--graph", args.graph])
    if args.mode:
        demo_args.extend(["--mode", args.mode])
    demo_args.extend(["--generations", str(args.generations)])
    demo_args.extend(["--pop-size", str(args.pop_size)])
    if args.device:
        demo_args.extend(["--device", args.device])
    if args.output_dir:
        demo_args.extend(["--output-dir", args.output_dir])
    if args.quiet:
        demo_args.append("--quiet")
    return demo_module.main(demo_args)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return 1
    return int(handler(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
