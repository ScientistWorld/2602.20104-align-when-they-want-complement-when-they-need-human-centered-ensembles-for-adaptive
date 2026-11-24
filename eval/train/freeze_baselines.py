"""Maintainer-only: freeze baseline predictions on the visible dev ids.

Reads <source>/<baseline>/<experiment>/predictions.csv and keeps the dev rows
in eval/train/baselines/<experiment>/<baseline>.csv, so the dev-slice scorer
never reads baselines from checkpoints/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    args = ap.parse_args()
    spec = json.loads((HERE / "spec.json").read_text())
    for exp, exp_spec in spec["experiments"].items():
        dev_ids = set(pd.read_csv(ROOT / exp_spec["dev"], usecols=["example_id"])["example_id"].tolist())
        target = HERE / "baselines" / exp
        target.mkdir(parents=True, exist_ok=True)
        for b in spec["baselines"]:
            pred = pd.read_csv(Path(args.source) / b / exp / "predictions.csv", usecols=["example_id", "prediction"])
            kept = pred[pred.example_id.isin(dev_ids)].sort_values("example_id")
            if len(kept) != len(dev_ids):
                raise SystemExit(f"{b}/{exp}: {len(kept)} of {len(dev_ids)} dev ids present")
            kept.to_csv(target / f"{b}.csv", index=False)
            print(f"[freeze train] {exp}/{b}: {len(kept)} rows")


if __name__ == "__main__":
    main()
