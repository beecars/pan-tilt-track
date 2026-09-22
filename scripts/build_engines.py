#!/usr/bin/env python3
"""Exports the FP16 TensorRT engines run_tracker.py expects: a large one
for the wide camera, small ones for the telephoto. Engines aren't portable
across hardware/JetPack/TensorRT versions, so run this on-device:
`./scripts/docker-run.sh python scripts/build_engines.py`."""

import argparse
import sys
from pathlib import Path

DEFAULT_SIZES = [(1440, 2560), (544, 960), (384, 640)]

REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_size(value: str) -> tuple[int, int]:
    """--sizes value: "HxW" or "H,W"."""
    parts = value.replace("x", ",").split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"--sizes entry must be 'HxW', got {value!r}")
    try:
        h, w = (int(p) for p in parts)
    except ValueError:
        raise argparse.ArgumentTypeError(f"--sizes entry must be 'HxW' integers, got {value!r}") from None
    if h % 32 or w % 32:
        raise argparse.ArgumentTypeError(f"--sizes entry must be stride-32 multiples, got {value!r}")
    return h, w


def engine_path(weights: Path, height: int, width: int) -> Path:
    return REPO_ROOT / f"{weights.stem}_{height}x{width}.engine"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="yolo26n.pt", help="source .pt model")
    parser.add_argument(
        "--sizes",
        type=parse_size,
        nargs="*",
        default=DEFAULT_SIZES,
        help="inference sizes to export, as 'HxW' (default: %s)"
        % " ".join(f"{h}x{w}" for h, w in DEFAULT_SIZES),
    )
    parser.add_argument("--force", action="store_true", help="re-export engines that already exist")
    args = parser.parse_args()

    from ultralytics import YOLO

    weights = Path(args.weights)
    for height, width in args.sizes:
        target = engine_path(weights, height, width)
        if target.exists() and not args.force:
            print(f"skip {target.name} (exists; --force to re-export)")
            continue
        print(f"exporting {weights} at {height}x{width} -> {target.name}")
        exported = Path(
            YOLO(str(weights)).export(format="engine", imgsz=[height, width], half=True, device=0)
        )
        exported.replace(target)
        print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
