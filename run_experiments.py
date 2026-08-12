from __future__ import annotations

import argparse
from pathlib import Path

from src.config import load_config
from src.experiment import run_experiment_suite


def parse_args():
    p = argparse.ArgumentParser(description="Run reproducible 6G-WSN routing experiments.")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--protocols", nargs="+", default=None)
    p.add_argument("--seeds", nargs="+", type=int, default=None)
    p.add_argument("--rounds", type=int, default=None)
    p.add_argument("--output", default="results")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.protocols is not None:
        cfg["experiment"]["protocols"] = args.protocols
    if args.seeds is not None:
        cfg["experiment"]["seeds"] = args.seeds
    if args.rounds is not None:
        cfg["network"]["rounds"] = args.rounds

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    run_experiment_suite(cfg, out)
    print(f"Completed. Results written to: {out.resolve()}")


if __name__ == "__main__":
    main()
