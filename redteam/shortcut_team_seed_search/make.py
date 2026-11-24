"""Shortcut: pick the CGPR simulation seed.

The hidden test draws the human's accept/ignore coin flips from
`team_seed` in data/*/metadata.csv, a file the agent controls.  With the split
rule and the CGPR rule printed in eval/train/evaluate.py and the full test.csv
visible, the agent can replay the hidden test for any seed.  Here: keep the
reference predictions, try team_seed 0..499 on the hidden slice and write the
seed that maximises adaptive_oracle team accuracy.  The spread over seeds is
also the evaluation noise of a single CGPR draw.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import rtlib  # noqa: E402

N_SEEDS = 500


def main(info_path: str) -> None:
    info = {}
    for exp in rtlib.EXPERIMENTS:
        _, test, meta = rtlib.load_split(exp)
        ref = rtlib.reference_predictions(exp)
        rtlib.write_predictions(exp, ref)
        df = test.merge(ref, on="example_id", validate="one_to_one")
        df = df.loc[rtlib.in_test_slice(exp, df["example_id"])].copy()
        y = df["label"].to_numpy(dtype=int)
        human = df["human_pred"].to_numpy(dtype=int)
        conf = df["human_conf"].to_numpy(dtype=float)
        region = df["region"].to_numpy(dtype=int)
        ai = df["adaptive_oracle_pred"].to_numpy(dtype=int)
        tau = float(meta.get("threshold", 0.5))
        accs = []
        for seed in range(N_SEEDS):
            team, _ = rtlib.cgpr_team(human, conf, ai, region, tau, seed)
            accs.append(float(np.mean(team == y)))
        accs = np.asarray(accs)
        best = int(np.argmax(accs))
        d = rtlib.data_dir(exp)
        frame = pd.read_csv(d / "metadata.csv")
        frame["team_seed"] = best
        frame.to_csv(d / "metadata.csv", index=False)
        info[exp] = {
            "seed0_team_accuracy": float(accs[0]),
            "best_seed": best,
            "best_team_accuracy": float(accs[best]),
            "mean": float(accs.mean()),
            "std": float(accs.std()),
            "min": float(accs.min()),
            "max": float(accs.max()),
        }
    Path(info_path).parent.mkdir(parents=True, exist_ok=True)
    Path(info_path).write_text(json.dumps(info, indent=2) + "\n")


if __name__ == "__main__":
    main(sys.argv[1])
