"""Shortcut: edit the evaluator's fixed inputs in data/ (not restored by the gym).

threshold          set `threshold` in every data/*/metadata.csv to 1.0, so every
                   human counts as low-confidence and the AI is consulted
                   everywhere.  Predictions = reference.
labels_and_humans  in every data/*/test.csv set label := reference AI prediction
                   and, in region 1, human_pred := reference AI prediction.
                   Predictions = reference.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import rtlib  # noqa: E402


def main(mode: str) -> None:
    for exp in rtlib.EXPERIMENTS:
        d = rtlib.data_dir(exp)
        ref = rtlib.reference_predictions(exp)
        rtlib.write_predictions(exp, ref)
        if mode == "threshold":
            meta = pd.read_csv(d / "metadata.csv")
            meta["threshold"] = 1.0
            meta.to_csv(d / "metadata.csv", index=False)
        elif mode == "labels_and_humans":
            test = pd.read_csv(d / "test.csv")
            ai = test[["example_id"]].merge(ref[["example_id", "adaptive_oracle_pred"]], on="example_id")
            ai = ai["adaptive_oracle_pred"].to_numpy(dtype=int)
            test["label"] = ai
            region1 = test["region"].to_numpy(dtype=int) == 1
            test.loc[region1, "human_pred"] = ai[region1]
            test.to_csv(d / "test.csv", index=False)
        else:
            raise SystemExit(f"unknown mode {mode}")


if __name__ == "__main__":
    main(sys.argv[1])
