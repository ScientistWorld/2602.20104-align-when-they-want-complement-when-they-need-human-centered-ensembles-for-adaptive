"""The paper's human-centred adaptive AI ensemble on the admissions world.

Follows the paper's synthetic College Admission recipe (logistic-regression
AI models, sklearn defaults) with its treatment of stochastic thresholds
(Proposition 5) and of unobserved confidence at test time:

* Instance weights: w_a = F_T(C^h), w_c = 1 - F_T(C^h), tau ~ U[0.6, 0.8].
* Aligned specialist m_a: fits the human's own predictions, weighted by w_a.
* Complementary specialist m_c: fits ground truth, weighted by w_c.
* Routing (Eq. 7): aligned specialist when the decision falls in the
  alignment region.
    - group known at test time (the paper's synthetic default, "group
      membership is treated as known"): specialists are trained on their own
      group (privileged -> aligned, underprivileged -> complementary) and
      routing uses the group.
    - group hidden: the human's confidence is not observable at test time,
      so, as in the paper's behaviour-grounded pipeline, a confidence model
      fitted on the pilot data (train split) supplies P(C^h > tau | x) and
      Eq. 7 routes to the aligned specialist when it is >= 0.5.
  The paper's Rational Routing Shortcut (Eq. 8) is available as
  `--router rrs` (a weaker router on this world, see HARDENING.md).

Baselines (the paper's single-model paradigms, also logistic regression):
standard AI (ERM on labels), behaviour-aware single AI (the weighted
surrogate g_p of Theorem 4, human targets in the alignment region), aligned AI
alone, complementary AI alone.

Output: <out-root>/<name>/<experiment>/predictions.csv with columns
example_id, prediction for every dev and test-input row.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

BASELINES = ("standard_ai", "behavior_aware_ai", "aligned_ai", "complementary_ai")


def f_t(conf: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """CDF of the uniform threshold distribution: P(tau < conf)."""
    return np.clip((conf - lo) / (hi - lo), 0.0, 1.0)


def fit_lr(x, y, w=None) -> LogisticRegression:
    model = LogisticRegression(solver="lbfgs", max_iter=5000)
    model.fit(x, np.asarray(y, dtype=int), sample_weight=w)
    return model


def fit_soft_lr(x, p) -> LogisticRegression:
    """Logistic regression on soft targets p in [0, 1] (duplicated rows)."""
    xx = np.vstack([x, x])
    yy = np.concatenate([np.ones(len(x), dtype=int), np.zeros(len(x), dtype=int)])
    ww = np.concatenate([p, 1.0 - p])
    keep = ww > 0
    return fit_lr(xx[keep], yy[keep], ww[keep])


def surrogate_single(x, y, h, w_a, w_c) -> LogisticRegression:
    """g_p(theta) = p L_a + (1 - p) L_c with p = expected alignment-region share."""
    n = len(y)
    p = float(w_a.sum() / (w_a.sum() + w_c.sum()))
    sw_a = p * w_a / max(w_a.sum(), 1e-12) * n
    sw_c = (1.0 - p) * w_c / max(w_c.sum(), 1e-12) * n
    xx = np.vstack([x, x])
    tt = np.concatenate([h, y])
    ww = np.concatenate([sw_a, sw_c])
    keep = ww > 0
    return fit_lr(xx[keep], tt[keep], ww[keep])


def load_experiment(data_root: Path, exp: str, extra_train_root: Path | None = None):
    d = data_root / exp
    task = json.loads((d / "task.json").read_text())
    train = pd.read_csv(d / task["files"]["train"])
    if extra_train_root is not None:
        extra = pd.read_csv(extra_train_root / exp / "train.csv")
        train = pd.concat([train, extra[train.columns]], ignore_index=True)
    dev = pd.read_csv(d / task["files"]["dev"])
    test = pd.read_csv(d / task["files"]["test_inputs"])
    cols = ["example_id"] + task["features"] + (["group"] if task["group_column_given"] else [])
    rows = pd.concat([dev[cols], test[cols]], ignore_index=True)
    return task, train, rows


def run_paper_method(task: dict, train: pd.DataFrame, rows: pd.DataFrame, router: str = "paper") -> dict:
    lo = float(task["team_model"]["tau_low"])
    hi = float(task["team_model"]["tau_high"])
    feats = task["features"]
    x = train[feats].to_numpy(dtype=float)
    y = train["label"].to_numpy(dtype=int)
    h = train["human_pred"].to_numpy(dtype=int)
    w_a = f_t(train["human_conf"].to_numpy(dtype=float), lo, hi)
    w_c = 1.0 - w_a
    xe = rows[feats].to_numpy(dtype=float)
    out: dict[str, np.ndarray] = {}
    info: dict = {}

    if task["group_column_given"]:
        g = train["group"].to_numpy(dtype=int)
        ge = rows["group"].to_numpy(dtype=int)
        m_a = fit_lr(x[g == 1], h[g == 1], w_a[g == 1])
        m_c = fit_lr(x[g == 0], y[g == 0], w_c[g == 0])
        route = ge == 1
        # Authors' single-model surrogate with group-defined regions.
        single = surrogate_single(x, y, h, (g == 1).astype(float), (g == 0).astype(float))
        info["router"] = "known group (paper's oracle routing)"
    else:
        m_a = fit_lr(x, h, w_a)
        m_c = fit_lr(x, y, w_c)
        pa_e = m_a.predict_proba(xe)[:, 1]
        pc_e = m_c.predict_proba(xe)[:, 1]
        if router == "rrs":
            conf_a = np.maximum(pa_e, 1.0 - pa_e)
            conf_c = np.maximum(pc_e, 1.0 - pc_e)
            route = conf_a >= conf_c
            info["router"] = "Rational Routing Shortcut (Eq. 8)"
        else:
            proxy = fit_soft_lr(x, w_a)
            route = proxy.predict_proba(xe)[:, 1] >= 0.5
            info["router"] = "Eq. 7 with a learned confidence model P(C^h > tau | x) >= 0.5"
        single = surrogate_single(x, y, h, w_a, w_c)

    pred_a = m_a.predict(xe).astype(int)
    pred_c = m_c.predict(xe).astype(int)
    out["adaptive"] = np.where(route, pred_a, pred_c).astype(int)
    out["standard_ai"] = fit_lr(x, y).predict(xe).astype(int)
    out["behavior_aware_ai"] = single.predict(xe).astype(int)
    out["aligned_ai"] = pred_a
    out["complementary_ai"] = pred_c
    info["routed_to_aligned_fraction"] = float(np.mean(route))
    info["n_train"] = int(len(train))
    return {"predictions": out, "info": info}


def write_predictions(path: Path, ids: np.ndarray, pred: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"example_id": ids.astype(np.int64), "prediction": pred.astype(int)}).to_csv(path, index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--out-root", default="checkpoints")
    ap.add_argument("--name", default="submission")
    ap.add_argument("--experiments", default=None, help="comma list; default: data/experiments.json")
    ap.add_argument("--router", choices=["paper", "rrs"], default="paper")
    ap.add_argument("--baselines-out", default=None, help="also write the four baselines under this root")
    ap.add_argument("--baselines-only", action="store_true")
    ap.add_argument("--train-fraction", type=float, default=1.0)
    ap.add_argument("--bootstrap-seed", type=int, default=-1)
    ap.add_argument("--extra-train-root", default=None)
    args = ap.parse_args()

    data_root = Path(args.data_root)
    if args.experiments:
        experiments = [e for e in args.experiments.split(",") if e]
    else:
        experiments = json.loads((data_root / "experiments.json").read_text())["experiments"]
    extra = Path(args.extra_train_root) if args.extra_train_root else None
    for exp in experiments:
        task, train, rows = load_experiment(data_root, exp, extra)
        if args.train_fraction < 1.0:
            train = train.sample(frac=args.train_fraction, random_state=0).reset_index(drop=True)
        if args.bootstrap_seed >= 0:
            train = train.sample(n=len(train), replace=True, random_state=args.bootstrap_seed).reset_index(drop=True)
        res = run_paper_method(task, train, rows, router=args.router)
        ids = rows["example_id"].to_numpy(dtype=np.int64)
        if not args.baselines_only:
            out = Path(args.out_root) / args.name / exp
            write_predictions(out / "predictions.csv", ids, res["predictions"]["adaptive"])
            (out / "run_info.json").write_text(json.dumps(res["info"], indent=2) + "\n")
        if args.baselines_out or args.baselines_only:
            root = Path(args.baselines_out or args.out_root)
            for b in BASELINES:
                write_predictions(root / b / exp / "predictions.csv", ids, res["predictions"][b])
        print(f"[method] {exp}: {res['info']}")


if __name__ == "__main__":
    main()
