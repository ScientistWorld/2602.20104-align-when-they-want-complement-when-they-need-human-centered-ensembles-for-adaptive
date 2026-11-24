"""Hidden-test scorer for the admissions human-AI team task.

Run with no arguments (via scripts/evaluate_test.sh).  It scores every method
directory under checkpoints/ plus the ranked slot `submission` (scored as
failed when absent), and the frozen baselines kept in eval/test/baselines/.

Contract per method and experiment:
    checkpoints/<method>/<experiment>/predictions.csv
    columns: example_id, prediction   (prediction is 0 or 1)
Every hidden-test example_id must appear exactly once with a binary
prediction; otherwise the experiment scores its worst value in every metric.
Rows for other ids (e.g. the visible dev split) are ignored.

Every number is recomputed here from the binary recommendations and the
hidden answers in eval/test/answers/.  Team accuracy is the exact expectation
of the CGPR team decision over the per-decision threshold tau ~ U[lo, hi]
and the reliance coin flip, so it has no simulation noise:

    P_i  = P(conf_i > tau) = clip((conf_i - lo) / (hi - lo), 0, 1)
    r    = sum_i P_i [m_i == h_i] / sum_i P_i
    team = mean_i( P_i [h_i == y_i] + (1 - P_i) (r [m_i == y_i] + (1 - r) [h_i == y_i]) )
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent


def load_spec() -> dict:
    return json.loads((HERE / "spec.json").read_text())


def alignment_probability(conf: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip((conf - lo) / (hi - lo), 0.0, 1.0)


def load_answers(path: Path) -> pd.DataFrame:
    ans = pd.read_csv(path)
    ans["example_id"] = ans["example_id"].astype(np.int64)
    return ans.sort_values("example_id").reset_index(drop=True)


def read_binary_predictions(path: Path, ids: np.ndarray) -> np.ndarray:
    """Binary predictions aligned to `ids`; raises ValueError on any violation."""
    if not path.is_file():
        raise ValueError("predictions.csv missing")
    df = pd.read_csv(path)
    for col in ("example_id", "prediction"):
        if col not in df.columns:
            raise ValueError(f"column '{col}' missing")
    eid = pd.to_numeric(df["example_id"], errors="coerce").to_numpy(dtype=float)
    if np.isnan(eid).any() or np.any(np.mod(eid, 1.0) != 0.0):
        raise ValueError("non-integer example_id")
    eid = eid.astype(np.int64)
    if len(np.unique(eid)) != len(eid):
        raise ValueError("duplicate example_id rows")
    pred = pd.to_numeric(df["prediction"], errors="coerce").to_numpy(dtype=float)
    order = np.argsort(eid)
    eid, pred = eid[order], pred[order]
    pos = np.searchsorted(eid, ids)
    pos = np.clip(pos, 0, max(len(eid) - 1, 0))
    if len(eid) == 0 or not np.array_equal(eid[pos], ids):
        missing = int(np.sum(eid[pos] != ids)) if len(eid) else len(ids)
        raise ValueError(f"{missing} scored example_ids missing")
    vals = pred[pos]
    if np.isnan(vals).any() or not np.isin(vals, (0.0, 1.0)).all():
        raise ValueError("prediction values must be exactly 0 or 1")
    return vals.astype(np.int64)


def team_metrics(pred: np.ndarray, ans: pd.DataFrame, lo: float, hi: float) -> dict:
    y = ans["label"].to_numpy(dtype=np.int64)
    h = ans["human_pred"].to_numpy(dtype=np.int64)
    p = alignment_probability(ans["human_conf"].to_numpy(dtype=float), lo, hi)
    agree = (pred == h).astype(float)
    mass = float(p.sum())
    r = float(np.sum(p * agree) / mass) if mass > 0 else 0.0
    hy = (h == y).astype(float)
    my = (pred == y).astype(float)
    team = float(np.mean(p * hy + (1.0 - p) * (r * my + (1.0 - r) * hy)))
    comp_mass = float((1.0 - p).sum())
    out = {
        "team_accuracy": team,
        "reliance_probability": r,
        "ai_accuracy": float(my.mean()),
        "complementary_region_accuracy": float(np.sum((1.0 - p) * my) / comp_mass) if comp_mass > 0 else 0.0,
    }
    for key, val in out.items():
        if not (math.isfinite(val) and 0.0 <= val <= 1.0):
            raise ValueError(f"impossible value {key}={val}")
    return out


def method_names(spec: dict) -> tuple[list[str], list[str]]:
    reserved = set(spec["baselines"])
    names = {spec["submission_slot"]}
    ckpt = ROOT / "checkpoints"
    if ckpt.is_dir():
        for d in ckpt.iterdir():
            if d.is_dir() and not d.name.startswith((".", "_")):
                names.add(d.name)
    skipped = sorted(names & reserved)
    return sorted(names - reserved), skipped


def score_all(spec: dict) -> dict:
    lo, hi = float(spec["tau_low"]), float(spec["tau_high"])
    worst = dict(spec["worst"])
    methods, skipped = method_names(spec)
    out = {
        "_metadata": {
            "slice": "test",
            "team_model": f"exact expectation of CGPR, tau ~ U[{lo}, {hi}]",
            "submission_slot": spec["submission_slot"],
            "methods_scored": methods,
            "reserved_names_ignored": skipped,
        },
        "experiments": {},
    }
    for exp, exp_spec in spec["experiments"].items():
        ans = load_answers(HERE / exp_spec["answers"])
        ids = ans["example_id"].to_numpy(dtype=np.int64)
        if len(ids) != int(exp_spec["n_scored"]):
            raise RuntimeError(f"{exp}: hidden answers have {len(ids)} rows, expected {exp_spec['n_scored']}")
        results, failures = {}, {}
        entries = [(b, HERE / "baselines" / exp / f"{b}.csv") for b in spec["baselines"]]
        entries += [(m, ROOT / "checkpoints" / m / exp / "predictions.csv") for m in methods]
        for name, path in entries:
            try:
                pred = read_binary_predictions(path, ids)
                results[name] = team_metrics(pred, ans, lo, hi)
            except Exception as exc:  # any failure scores the worst value
                results[name] = dict(worst)
                failures[name] = f"{type(exc).__name__}: {exc}"
        out["experiments"][exp] = {"results": results, "_failures": failures, "_n_scored": int(len(ids))}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default=str(ROOT / "scoring" / "scores_test.json"))
    args = ap.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.is_symlink():
        output.unlink()
    scores = score_all(load_spec())
    fd, tmp = tempfile.mkstemp(dir=str(output.parent), prefix=".scores_test.", suffix=".json")
    with os.fdopen(fd, "w") as fh:
        fh.write(json.dumps(scores, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, output)
    for exp, block in scores["experiments"].items():
        sub = block["results"].get(load_spec()["submission_slot"], {})
        print(f"[test] {exp}: submission team_accuracy={sub.get('team_accuracy')} failures={sorted(block['_failures'])}")


if __name__ == "__main__":
    main()
