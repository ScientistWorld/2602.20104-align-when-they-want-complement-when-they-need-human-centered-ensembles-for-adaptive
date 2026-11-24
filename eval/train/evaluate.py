"""Visible dev-slice scorer for the admissions human-AI team task.

Same measurement as the hidden test, applied to the visible dev split
(data/<experiment>/dev.csv holds its answers).  Run with no arguments via
scripts/evaluate_train.sh; it scores every method directory under
checkpoints/ plus the ranked slot `submission` and the frozen baselines in
eval/train/baselines/.

Prediction contract: checkpoints/<method>/<experiment>/predictions.csv with
columns example_id, prediction (0 or 1); every dev example_id exactly once.
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


def read_spec() -> dict:
    return json.loads((HERE / "spec.json").read_text())


def p_alignment(conf: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip((conf - lo) / (hi - lo), 0.0, 1.0)


def dev_answers(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=["example_id", "label", "human_pred", "human_conf"])
    df["example_id"] = df["example_id"].astype(np.int64)
    return df.sort_values("example_id").reset_index(drop=True)


def aligned_predictions(path: Path, ids: np.ndarray) -> np.ndarray:
    if not path.is_file():
        raise ValueError("predictions.csv missing")
    table = pd.read_csv(path)
    if "example_id" not in table.columns or "prediction" not in table.columns:
        raise ValueError("columns example_id and prediction are required")
    eid = pd.to_numeric(table["example_id"], errors="coerce").to_numpy(dtype=float)
    if np.isnan(eid).any() or np.any(np.mod(eid, 1.0) != 0.0):
        raise ValueError("non-integer example_id")
    eid = eid.astype(np.int64)
    if len(set(eid.tolist())) != len(eid):
        raise ValueError("duplicate example_id rows")
    lookup = dict(zip(eid.tolist(), pd.to_numeric(table["prediction"], errors="coerce").tolist()))
    missing = [i for i in ids.tolist() if i not in lookup]
    if missing:
        raise ValueError(f"{len(missing)} dev example_ids missing")
    vals = np.array([lookup[i] for i in ids.tolist()], dtype=float)
    if np.isnan(vals).any() or not np.isin(vals, (0.0, 1.0)).all():
        raise ValueError("prediction values must be exactly 0 or 1")
    return vals.astype(np.int64)


def metrics(pred: np.ndarray, ans: pd.DataFrame, lo: float, hi: float) -> dict:
    y = ans["label"].to_numpy(dtype=np.int64)
    h = ans["human_pred"].to_numpy(dtype=np.int64)
    p = p_alignment(ans["human_conf"].to_numpy(dtype=float), lo, hi)
    r = float((p * (pred == h)).sum() / p.sum()) if p.sum() > 0 else 0.0
    human_ok = (h == y).astype(float)
    ai_ok = (pred == y).astype(float)
    team = float(np.mean(p * human_ok + (1.0 - p) * (r * ai_ok + (1.0 - r) * human_ok)))
    q = 1.0 - p
    res = {
        "team_accuracy": team,
        "reliance_probability": r,
        "ai_accuracy": float(ai_ok.mean()),
        "complementary_region_accuracy": float((q * ai_ok).sum() / q.sum()) if q.sum() > 0 else 0.0,
    }
    if any(not (math.isfinite(v) and 0.0 <= v <= 1.0) for v in res.values()):
        raise ValueError("impossible metric value")
    return res


def discover_methods(spec: dict) -> tuple[list[str], list[str]]:
    reserved = set(spec["baselines"])
    found = {spec["submission_slot"]}
    base = ROOT / "checkpoints"
    if base.is_dir():
        found |= {d.name for d in base.iterdir() if d.is_dir() and not d.name.startswith((".", "_"))}
    return sorted(found - reserved), sorted(found & reserved)


def run(spec: dict) -> dict:
    lo, hi = float(spec["tau_low"]), float(spec["tau_high"])
    methods, ignored = discover_methods(spec)
    payload = {
        "_metadata": {
            "slice": "train",
            "team_model": f"exact expectation of CGPR, tau ~ U[{lo}, {hi}]",
            "methods_scored": methods,
            "reserved_names_ignored": ignored,
        },
        "experiments": {},
    }
    for exp, exp_spec in spec["experiments"].items():
        ans = dev_answers(ROOT / exp_spec["dev"])
        ids = ans["example_id"].to_numpy(dtype=np.int64)
        results, failures = {}, {}
        todo = [(b, HERE / "baselines" / exp / f"{b}.csv") for b in spec["baselines"]]
        todo += [(m, ROOT / "checkpoints" / m / exp / "predictions.csv") for m in methods]
        for name, path in todo:
            try:
                results[name] = metrics(aligned_predictions(path, ids), ans, lo, hi)
            except Exception as exc:
                results[name] = dict(spec["worst"])
                failures[name] = f"{type(exc).__name__}: {exc}"
        payload["experiments"][exp] = {"results": results, "_failures": failures, "_n_scored": int(len(ids))}
    return payload


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default=str(ROOT / "scoring" / "scores_train.json"))
    args = ap.parse_args()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() or out.is_symlink():
        out.unlink()
    payload = run(read_spec())
    fd, tmp = tempfile.mkstemp(dir=str(out.parent), prefix=".scores_train.", suffix=".json")
    with os.fdopen(fd, "w") as fh:
        fh.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, out)
    for exp, block in payload["experiments"].items():
        print(f"[train] {exp}: methods={sorted(block['results'])} failures={sorted(block['_failures'])}")


if __name__ == "__main__":
    main()
