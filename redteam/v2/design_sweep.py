"""World-design sensitivity sweep (in memory; not part of the gym's test).

For each candidate world in eval/test/world.py, builds the four experiments
under tmp/design/<world>/, runs the paper's method, its baselines and the
variants, and scores them on that world's hidden answers with the same exact
CGPR formula as the hidden test.  Used to check that the chosen world gives
reference > baselines, better variants > reference by more than noise, and a
ceiling well above every simple variant.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "method"))
import adaptive_team_ai as paper  # noqa: E402
import variants  # noqa: E402

WEIGHTS = {
    "admission_group_hidden": 0.35,
    "admission_group_hidden_low_reliability": 0.20,
    "admission_group_known": 0.25,
    "admission_group_known_low_reliability": 0.20,
}
REL_COEF = 0.25


def score(pred, ans):
    y = ans.label.to_numpy()
    h = ans.human_pred.to_numpy()
    p = np.clip((ans.human_conf.to_numpy() - 0.6) / 0.2, 0, 1)
    r = float((p * (pred == h)).sum() / p.sum())
    hy = (h == y).astype(float)
    my = (pred == y).astype(float)
    team = float(np.mean(p * hy + (1 - p) * (r * my + (1 - r) * hy)))
    return team, r, float(my.mean())


def run_world(world: str) -> dict:
    base = ROOT / "tmp" / "design" / world
    data_root, ans_root = base / "data", base / "answers"
    subprocess.run([sys.executable, str(ROOT / "eval" / "test" / "world.py"), "build", "--world", world,
                    "--data-root", str(data_root), "--answers-root", str(ans_root),
                    "--summary", str(base / "build_summary.json")], check=True)
    subprocess.run([sys.executable, str(ROOT / "eval" / "test" / "world.py"), "ceiling", "--world", world,
                    "--out-root", str(base / "ceiling")], check=True)
    table = {}
    for exp in WEIGHTS:
        task, train, rows = paper.load_experiment(data_root, exp)
        ans = pd.read_csv(ans_root / f"{exp}.csv")
        rows_t = rows[rows.example_id.isin(set(ans.example_id))]
        ans = ans.set_index("example_id").loc[rows_t.example_id].reset_index()
        preds = {}
        res = paper.run_paper_method(task, train, rows_t)["predictions"]
        preds["reference"] = res["adaptive"]
        for b in paper.BASELINES:
            preds[b] = res[b]
        if not task["group_column_given"]:
            preds["rrs"] = paper.run_paper_method(task, train, rows_t, router="rrs")["predictions"]["adaptive"]
        for name, fn in variants.ALL.items():
            preds[name] = fn(task, train, rows_t)
        cp = pd.read_csv(base / "ceiling" / "generator_bayes_policy" / exp / "predictions.csv").set_index("example_id")
        preds["generator_bayes_policy"] = cp.loc[rows_t.example_id, "prediction"].to_numpy()
        table[exp] = {k: score(v, ans) for k, v in preds.items()}
    summary = {}
    methods = sorted({m for t in table.values() for m in t})
    for m in methods:
        total = 0.0
        for exp, w in WEIGHTS.items():
            if m not in table[exp]:
                ref = table[exp]["reference"]
                cur = ref
            else:
                cur = table[exp][m]
            ref = table[exp]["reference"]
            total += w * ((cur[0] - ref[0]) + REL_COEF * (cur[1] - ref[1]))
        summary[m] = total
    return {"table": table, "rank_gain": summary,
            "build": json.loads((base / "build_summary.json").read_text())}


def main() -> None:
    worlds = sys.argv[1].split(",") if len(sys.argv) > 1 else ["default", "under_mild", "under_strong", "under_narrow_gate"]
    out = {w: run_world(w) for w in worlds}
    (HERE / "design_sweep.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    lines = ["# World-design sweep (team accuracy / reliance on each world's hidden test)", ""]
    for w, res in out.items():
        lines += [f"## world `{w}`", ""]
        for exp, b in res["build"].items():
            lines.append(f"- {exp}: human acc privileged {b['human_acc_priv']:.3f}, underprivileged {b['human_acc_under']:.3f}, "
                         f"privileged share {b['privileged_fraction']:.3f}, mean P(alignment) {b['mean_p_alignment']:.3f}, slip {b['eps_priv']:.3f}")
        exps = list(WEIGHTS)
        lines += ["", "| method | raw gain | " + " | ".join(f"{e} team / rel / ai" for e in exps) + " |",
                  "|---|---|" + "---|" * len(exps)]
        for m, g in sorted(res["rank_gain"].items(), key=lambda kv: -kv[1]):
            cells = []
            for e in exps:
                v = res["table"][e].get(m)
                cells.append("—" if v is None else f"{v[0]:.4f} / {v[1]:.3f} / {v[2]:.3f}")
            lines.append(f"| {m} | {g:+.4f} | " + " | ".join(cells) + " |")
        lines.append("")
    (HERE / "design_sweep.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
