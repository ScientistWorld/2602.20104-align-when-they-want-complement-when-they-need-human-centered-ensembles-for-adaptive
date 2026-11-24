"""Write shortcut predictions into the proposed columns (baselines = reference).

Every mode starts from the reference method's own predictions and replaces
only the two proposed columns (adaptive_oracle, adaptive_weighted_oracle):

read_test_labels   region 1: the human's test-time prediction; region 0: the
                   test label column.  Route = test region column.
echo_human         region 1: the human's test-time prediction (an input
                   column of test.csv); region 0: the reference specialist.
generator_rule     the label rule of data/college_admission.py applied to
                   (gpa, score, region); no label column read.
constant_zero      predict 0 everywhere, no route column.
copy_human_input   predict the human's own test-time prediction everywhere
                   (the AI adds nothing); route = region.
impossible_values  prediction 2 (not a class), route 7.
nan_values         prediction NaN.
boost_baselines    proposed columns untouched; the three baseline columns
                   are overwritten with the reference adaptive_oracle output.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rtlib  # noqa: E402


def main(mode: str) -> None:
    for exp in rtlib.EXPERIMENTS:
        _, test, meta = rtlib.load_split(exp)
        out = rtlib.reference_predictions(exp)
        t = test.set_index("example_id").loc[out["example_id"]].reset_index()
        region = t["region"].to_numpy(dtype=int)
        human = t["human_pred"].to_numpy(dtype=int)
        label = t["label"].to_numpy(dtype=int)

        if mode == "boost_baselines":
            for b in rtlib.BASELINES:
                out[f"{b}_prob"] = out["adaptive_oracle_prob"]
                out[f"{b}_pred"] = out["adaptive_oracle_pred"]
                out[f"{b}_route_aligned"] = out["adaptive_oracle_route_aligned"]
            rtlib.write_predictions(exp, out)
            continue

        for col in rtlib.PROPOSED:
            route = region.copy()
            if mode == "read_test_labels":
                pred = np.where(region == 1, human, label)
            elif mode == "echo_human":
                pred = np.where(region == 1, human, out[f"{col}_pred"].to_numpy(dtype=int))
            elif mode == "generator_rule":
                pred = rtlib.generator_rule(t, meta)
            elif mode == "constant_zero":
                pred = np.zeros(len(out), dtype=int)
                route = None
            elif mode == "copy_human_input":
                pred = human.copy()
            elif mode == "impossible_values":
                pred = np.full(len(out), 2, dtype=int)
                route = np.full(len(out), 7, dtype=int)
            elif mode == "nan_values":
                pred = np.full(len(out), np.nan)
            else:
                raise SystemExit(f"unknown mode {mode}")
            out[f"{col}_pred"] = pred
            out[f"{col}_prob"] = pred
            if route is None:
                out = out.drop(columns=[f"{col}_route_aligned"])
            else:
                out[f"{col}_route_aligned"] = route
        rtlib.write_predictions(exp, out)


if __name__ == "__main__":
    main(sys.argv[1])
