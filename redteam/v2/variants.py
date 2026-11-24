"""Variant methods used to check that the hardened test rewards better methods.

Each variant maps (task, train, rows) -> binary predictions for `rows`.
Better variants: a non-linear confidence model, non-linear specialists, and a
team-aware recommendation rule that maximises the expected CGPR team accuracy
(Eq. 3) from estimated region probabilities.  Weaker variants: the paper's
RRS router, less data.  Simple / hand-written comparisons: gradient-boosted
single models and short hand-written admission rules.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# One CPU per job: keep OpenMP (gradient boosting) from spawning a thread per node core.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "method"))
import adaptive_team_ai as paper  # noqa: E402


def gbm():
    return HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=40,
        l2_regularization=1.0, early_stopping=False, random_state=0,
    )


def fit_gbm(x, y, w=None):
    keep = np.ones(len(y), dtype=bool) if w is None else (np.asarray(w) > 0)
    model = gbm()
    model.fit(x[keep], np.asarray(y, dtype=int)[keep], sample_weight=None if w is None else np.asarray(w)[keep])
    return model


def fit_soft_gbm(x, p):
    xx = np.vstack([x, x])
    yy = np.concatenate([np.ones(len(x), dtype=int), np.zeros(len(x), dtype=int)])
    ww = np.concatenate([p, 1.0 - p])
    return fit_gbm(xx, yy, ww)


def _arrays(task, train, rows, with_group: bool):
    feats = list(task["features"])
    if with_group and task["group_column_given"]:
        feats = feats + ["group"]
    lo = float(task["team_model"]["tau_low"])
    hi = float(task["team_model"]["tau_high"])
    x = train[feats].to_numpy(dtype=float)
    xe = rows[feats].to_numpy(dtype=float)
    y = train["label"].to_numpy(dtype=int)
    h = train["human_pred"].to_numpy(dtype=int)
    w_a = paper.f_t(train["human_conf"].to_numpy(dtype=float), lo, hi)
    return x, xe, y, h, w_a


def team_decision(pa, ph1, py1, acc_h_c, iters: int = 50) -> np.ndarray:
    """argmax_m R(m) * A(m) using estimated P(align|x), P(h=1|x,align), P(y=1|x,comp)."""
    a1, a0 = pa * ph1, pa * (1.0 - ph1)
    b1, b0 = (1.0 - pa) * py1, (1.0 - pa) * (1.0 - py1)
    c = (1.0 - pa) * acc_h_c
    pi_total = float(pa.sum())
    m = (b1 > b0).astype(int)
    for _ in range(iters):
        a = np.where(m == 1, a1, a0)
        b = np.where(m == 1, b1, b0)
        r = a.sum() / max(pi_total, 1e-12)
        lam = max((b - c).sum(), 0.0) / max(pi_total, 1e-12)
        new = ((lam * a1 + r * b1) > (lam * a0 + r * b0)).astype(int)
        if np.array_equal(new, m):
            break
        m = new
    return m


# ---------------------------------------------------------------- better


def proxy_gbm(task, train, rows):
    """Paper's specialists, but a gradient-boosted confidence model for routing."""
    if task["group_column_given"]:
        return paper.run_paper_method(task, train, rows)["predictions"]["adaptive"]
    x, xe, y, h, w_a = _arrays(task, train, rows, with_group=False)
    m_a = paper.fit_lr(x, h, w_a)
    m_c = paper.fit_lr(x, y, 1.0 - w_a)
    route = fit_soft_gbm(x, w_a).predict_proba(xe)[:, 1] >= 0.5
    return np.where(route, m_a.predict(xe), m_c.predict(xe)).astype(int)


def gbm_specialists(task, train, rows):
    """Same structure as the paper's method with gradient-boosted components."""
    x, xe, y, h, w_a = _arrays(task, train, rows, with_group=False)
    if task["group_column_given"]:
        g = train["group"].to_numpy(dtype=int)
        ge = rows["group"].to_numpy(dtype=int)
        m_a = fit_gbm(x[g == 1], h[g == 1], w_a[g == 1])
        m_c = fit_gbm(x[g == 0], y[g == 0], 1.0 - w_a[g == 0])
        route = ge == 1
    else:
        m_a = fit_gbm(x, h, w_a)
        m_c = fit_gbm(x, y, 1.0 - w_a)
        route = fit_soft_gbm(x, w_a).predict_proba(xe)[:, 1] >= 0.5
    return np.where(route, m_a.predict(xe), m_c.predict(xe)).astype(int)


def gbm_team(task, train, rows):
    """Gradient-boosted region/human/truth models + team-aware recommendation."""
    x, xe, y, h, w_a = _arrays(task, train, rows, with_group=True)
    w_c = 1.0 - w_a
    pa = fit_soft_gbm(x, w_a).predict_proba(xe)[:, 1]
    ph1 = fit_gbm(x, h, w_a).predict_proba(xe)[:, 1]
    py1 = fit_gbm(x, y, w_c).predict_proba(xe)[:, 1]
    acc_h_c = float(np.sum(w_c * (h == y)) / max(w_c.sum(), 1e-12))
    return team_decision(pa, ph1, py1, acc_h_c)


def lr_team(task, train, rows):
    """Paper's logistic components + team-aware recommendation (group hidden only)."""
    if task["group_column_given"]:
        return paper.run_paper_method(task, train, rows)["predictions"]["adaptive"]
    x, xe, y, h, w_a = _arrays(task, train, rows, with_group=False)
    w_c = 1.0 - w_a
    pa = paper.fit_soft_lr(x, w_a).predict_proba(xe)[:, 1]
    ph1 = paper.fit_lr(x, h, w_a).predict_proba(xe)[:, 1]
    py1 = paper.fit_lr(x, y, w_c).predict_proba(xe)[:, 1]
    acc_h_c = float(np.sum(w_c * (h == y)) / max(w_c.sum(), 1e-12))
    return team_decision(pa, ph1, py1, acc_h_c)


# ---------------------------------------------------------------- simple


def gbm_standard(task, train, rows):
    """Gradient-boosted standard AI on ground truth (ignores humans)."""
    x, xe, y, h, w_a = _arrays(task, train, rows, with_group=True)
    return fit_gbm(x, y).predict(xe).astype(int)


def gbm_behavior_aware(task, train, rows):
    """Gradient-boosted version of the behaviour-aware surrogate g_p."""
    x, xe, y, h, w_a = _arrays(task, train, rows, with_group=True)
    w_c = 1.0 - w_a
    n = len(y)
    p = float(w_a.sum() / (w_a.sum() + w_c.sum()))
    ww = np.concatenate([p * w_a / w_a.sum() * n, (1 - p) * w_c / w_c.sum() * n])
    return fit_gbm(np.vstack([x, x]), np.concatenate([h, y]), ww).predict(xe).astype(int)


def aligned_everywhere_gbm(task, train, rows):
    """Mimic the confident human everywhere (maximises reliance, not accuracy)."""
    x, xe, y, h, w_a = _arrays(task, train, rows, with_group=True)
    return fit_gbm(x, h).predict(xe).astype(int)


# ---------------------------------------------------------------- hand-written


def _privileged_guess(rows, task):
    if task["group_column_given"]:
        return rows["group"].to_numpy(dtype=int) == 1
    return rows["school_resources"].to_numpy(dtype=float) > 0.0


def handrule_paper(task, train, rows):
    """The paper's published label rule (delta = 0.25), hand-coded."""
    gpa = rows["gpa"].to_numpy(dtype=float)
    score = rows["score"].to_numpy(dtype=float)
    priv = _privileged_guess(rows, task)
    r_priv = (0.75 * score + 0.25 * gpa >= 0.5)
    r_under = (0.75 * gpa + 0.25 * score >= 0.5)
    return np.where(priv, r_priv, r_under).astype(int)


def handrule_score_heuristic(task, train, rows):
    """'Confident humans go by the test score' + the paper's GPA rule elsewhere."""
    gpa = rows["gpa"].to_numpy(dtype=float)
    score = rows["score"].to_numpy(dtype=float)
    priv = _privileged_guess(rows, task)
    return np.where(priv, score >= 0.5, 0.75 * gpa + 0.25 * score >= 0.5).astype(int)


BETTER = {"proxy_gbm": proxy_gbm, "lr_team": lr_team, "gbm_specialists": gbm_specialists, "gbm_team": gbm_team}
SIMPLE = {
    "gbm_standard": gbm_standard,
    "gbm_behavior_aware": gbm_behavior_aware,
    "aligned_everywhere_gbm": aligned_everywhere_gbm,
    "handrule_paper": handrule_paper,
    "handrule_score_heuristic": handrule_score_heuristic,
}
ALL = {**BETTER, **SIMPLE}


def write_all(data_root: Path, out_root: Path, names: list[str]) -> None:
    import json

    experiments = json.loads((data_root / "experiments.json").read_text())["experiments"]
    for exp in experiments:
        task, train, rows = paper.load_experiment(data_root, exp)
        ids = rows["example_id"].to_numpy(dtype=np.int64)
        for name in names:
            pred = ALL[name](task, train, rows)
            paper.write_predictions(out_root / name / exp / "predictions.csv", ids, pred)
            print(f"[variant] {name}/{exp}: positive rate {pred.mean():.3f}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=str(ROOT / "data"))
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--names", default=",".join(ALL))
    a = ap.parse_args()
    write_all(Path(a.data_root), Path(a.out_root), [n for n in a.names.split(",") if n])
