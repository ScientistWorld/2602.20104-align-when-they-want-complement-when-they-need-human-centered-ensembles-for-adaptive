"""Merge per-experiment score files into one scores.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def merge_score_files(paths: list[Path]) -> dict:
    merged = {"experiments": {}}
    for path in paths:
        payload = json.loads(path.read_text())
        for name, experiment in payload.get("experiments", {}).items():
            if name in merged["experiments"]:
                raise ValueError(f"duplicate experiment name while merging: {name}")
            merged["experiments"][name] = experiment
    return merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--output", default="scoring/scores.json")
    args = parser.parse_args()

    scores = merge_score_files([Path(path) for path in args.inputs])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(scores, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
