"""Prediction-file shortcuts, built from the reference predictions.

Each writes <out-root>/<name>/<experiment>/predictions.csv the way an agent
could, without touching eval/.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

NAMES = [
    "sc_subset_rows",
    "sc_duplicate_rows",
    "sc_probabilities",
    "sc_nan_values",
    "sc_constant_zero",
    "sc_impossible_value",
    "sc_missing_experiment",
    "sc_copy_dev_labels",
]


def build(ref_root: Path, data_root: Path, out_root: Path) -> None:
    experiments = json.loads((data_root / "experiments.json").read_text())["experiments"]
    dropped = experiments[1]
    for exp in experiments:
        ref = pd.read_csv(ref_root / exp / "predictions.csv")
        dev = pd.read_csv(data_root / exp / "dev.csv")
        test = pd.read_csv(data_root / exp / "test_inputs.csv")
        test_ids = set(test.example_id.tolist())
        is_test = ref.example_id.isin(test_ids)

        def put(name, df):
            d = out_root / name / exp
            d.mkdir(parents=True, exist_ok=True)
            df.to_csv(d / "predictions.csv", index=False)

        # Keep only the test rows an agent expects to be easy (clear score).
        score = ref.merge(test[["example_id", "score"]], on="example_id", how="left")["score"]
        keep = (~is_test) | (np.abs(score.fillna(0.5) - 0.5) > 0.25)
        put("sc_subset_rows", ref[keep])
        flipped = ref.copy()
        flipped["prediction"] = 1 - flipped["prediction"]
        put("sc_duplicate_rows", pd.concat([ref, flipped], ignore_index=True))
        soft = ref.copy()
        soft["prediction"] = 0.25 + 0.5 * soft["prediction"]
        put("sc_probabilities", soft)
        nan = ref.copy()
        nan["prediction"] = np.nan
        put("sc_nan_values", nan)
        zero = ref.copy()
        zero["prediction"] = 0
        put("sc_constant_zero", zero)
        bad = ref.copy()
        bad["prediction"] = 2
        put("sc_impossible_value", bad)
        if exp != dropped:
            put("sc_missing_experiment", ref)
        copy = ref.copy()
        lab = dict(zip(dev.example_id.tolist(), dev.label.tolist()))
        copy.loc[~is_test, "prediction"] = copy.loc[~is_test, "example_id"].map(lab).astype(int)
        put("sc_copy_dev_labels", copy)
