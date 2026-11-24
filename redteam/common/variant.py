"""Train the paper's models with a changed training regime and write predictions.

Used by the honest variants (seed spread, data scaling, ablations).  Writes
checkpoints/<exp>/predictions.csv for every experiment, with the five method
columns the hidden test scores.

--train-factor F   F<1: subsample F of train.csv; F>1: add (F-1)*len(train)
                   fresh rows from the gym's generator (same settings).
--bootstrap-seed S resample train.csv with replacement (seed S) before training.
--C C              inverse L2 strength of every logistic model (method uses 1.0).
--proposed MODE    what goes into the two proposed columns:
                   paper  = oracle-routed specialists (the reference method)
                   rrs    = Rational Routing Shortcut instead of oracle routing
                   single = the single behavior-aware model, no route column
                   single_route_region = single model + route column copied from test region
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rtlib  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-factor", type=float, default=1.0)
    ap.add_argument("--sample-seed", type=int, default=0)
    ap.add_argument("--bootstrap-seed", type=int, default=-1)
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--model-seed", type=int, default=42)
    ap.add_argument("--proposed", default="paper",
                    choices=["paper", "rrs", "single", "single_route_region"])
    ap.add_argument("--info", default="")
    args = ap.parse_args()

    info = {}
    for exp in rtlib.EXPERIMENTS:
        train, test, meta = rtlib.load_split(exp)
        if args.bootstrap_seed >= 0:
            train = train.sample(n=len(train), replace=True,
                                 random_state=args.bootstrap_seed).reset_index(drop=True)
        if args.train_factor < 1.0:
            train = train.sample(frac=args.train_factor,
                                 random_state=args.sample_seed).reset_index(drop=True)
        elif args.train_factor > 1.0:
            n_extra = int(round((args.train_factor - 1.0) * len(train)))
            extra = rtlib.extra_training_rows(meta, n_extra, seed=1000 + args.sample_seed)
            train = pd.concat([train, extra[train.columns]], ignore_index=True)

        out = rtlib.train_models(train, test, C=args.C, seed=args.model_seed)
        gt = test["region"].to_numpy(dtype=int)
        if args.proposed == "rrs":
            for col, src in (("adaptive_oracle", "rrs_unweighted"),
                             ("adaptive_weighted_oracle", "adaptive_rrs")):
                out[f"{col}_prob"] = out[f"{src}_prob"]
                out[f"{col}_pred"] = out[f"{src}_pred"]
                out[f"{col}_route_aligned"] = out[f"{src}_route_aligned"]
        elif args.proposed in ("single", "single_route_region"):
            for col in rtlib.PROPOSED:
                out[f"{col}_prob"] = out["behavior_aware_ai_prob"]
                out[f"{col}_pred"] = out["behavior_aware_ai_pred"]
                if args.proposed == "single":
                    out = out.drop(columns=[f"{col}_route_aligned"])
                else:
                    out[f"{col}_route_aligned"] = gt
        rtlib.write_predictions(exp, out)
        info[exp] = {
            "n_train": int(len(train)),
            "n_train_region1": int((train["region"] == 1).sum()),
        }
    if args.info:
        Path(args.info).parent.mkdir(parents=True, exist_ok=True)
        Path(args.info).write_text(json.dumps({"args": vars(args), "experiments": info}, indent=2) + "\n")


if __name__ == "__main__":
    main()
