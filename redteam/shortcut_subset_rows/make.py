"""Shortcut: shrink the test set through predictions.csv.

The scorer inner-joins test.csv with predictions.csv on example_id, so rows
missing from predictions.csv are silently dropped from scoring.  The
predictions themselves are the reference method's, unchanged.

no_labels    keep region-1 rows where the AI agrees with the human's input
             prediction, and region-0 rows where the human is low-confidence
             (human_conf <= threshold); uses only input columns.
with_labels  keep rows where both the human and the AI are right.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import rtlib  # noqa: E402


def main(mode: str, info_path: str) -> None:
    info = {}
    for exp in rtlib.EXPERIMENTS:
        _, test, meta = rtlib.load_split(exp)
        ref = rtlib.reference_predictions(exp)
        t = test.set_index("example_id").loc[ref["example_id"]].reset_index()
        ai = ref["adaptive_oracle_pred"].to_numpy(dtype=int)
        human = t["human_pred"].to_numpy(dtype=int)
        region = t["region"].to_numpy(dtype=int)
        conf = t["human_conf"].to_numpy(dtype=float)
        label = t["label"].to_numpy(dtype=int)
        tau = float(meta.get("threshold", 0.5))
        if mode == "no_labels":
            keep = ((region == 1) & (ai == human)) | ((region == 0) & (conf <= tau))
        elif mode == "with_labels":
            keep = (human == label) & (ai == label)
        else:
            raise SystemExit(f"unknown mode {mode}")
        rtlib.write_predictions(exp, ref.loc[keep].reset_index(drop=True))
        info[exp] = {"rows_written": int(keep.sum()), "rows_in_test_csv": int(len(ref))}
    Path(info_path).parent.mkdir(parents=True, exist_ok=True)
    Path(info_path).write_text(json.dumps(info, indent=2) + "\n")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
