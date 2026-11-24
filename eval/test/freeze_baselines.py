"""Maintainer-only: freeze baseline predictions on the hidden test ids.

Reads <source>/<baseline>/<experiment>/predictions.csv (written by the gym's
baseline run) and keeps only the hidden-test rows in
eval/test/baselines/<experiment>/<baseline>.csv.  The hidden test scores the
baselines from these frozen files, never from checkpoints/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    args = ap.parse_args()
    spec = json.loads((HERE / "spec.json").read_text())
    src = Path(args.source)
    for exp, exp_spec in spec["experiments"].items():
        ids = pd.read_csv(HERE / exp_spec["answers"], usecols=["example_id"])["example_id"]
        out_dir = HERE / "baselines" / exp
        out_dir.mkdir(parents=True, exist_ok=True)
        for b in spec["baselines"]:
            pred = pd.read_csv(src / b / exp / "predictions.csv", usecols=["example_id", "prediction"])
            kept = pred[pred.example_id.isin(set(ids.tolist()))].sort_values("example_id")
            if len(kept) != len(ids):
                raise SystemExit(f"{b}/{exp}: {len(kept)} of {len(ids)} hidden ids present")
            kept.to_csv(out_dir / f"{b}.csv", index=False)
            print(f"[freeze test] {exp}/{b}: {len(kept)} rows")


if __name__ == "__main__":
    main()
