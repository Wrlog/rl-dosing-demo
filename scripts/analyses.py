"""Extra analyses: reachability as IC50 variability shrinks, the drug-cost trade-off, and a look at misses.

    python scripts/analyses.py [--quick]      (run after evaluate.py)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from rl_dosing.env import N_OCC, run_policy
from rl_dosing.experiment import FLARE_CV, MAIN_POP, N_PROFILES, POOL_SEED, SHARED_POP, main_env_config
from rl_dosing.model import TARGET, sample_cohort, sample_profile_pool
from rl_dosing.policies import ceiling_regimen, standard_regimen

ROOT = Path(__file__).resolve().parents[1]
IC50_CVS = (0.9, 0.75, 0.6, 0.45, 0.3, 0.15, 0.0)


def wilson(p, n, z=1.96):
    centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return centre - half, centre + half


def ic50_ladder(n: int) -> pd.DataFrame:
    """Reachability (ceiling) and standard-regimen attainment, changing only the IC50 variance.

    The same patients and random draws are used at every rung.
    """
    pool = sample_profile_pool(N_PROFILES, np.random.default_rng(POOL_SEED))
    cohort = sample_cohort(n, pool, np.random.default_rng(41), N_OCC, cycle_profiles=True)
    rows = []
    for setting, pop in (("IC50 10 mg/L", MAIN_POP), ("IC50 10 mg/L, flares", MAIN_POP.with_(cv_flare=FLARE_CV)),
                         ("IC50 3 mg/L (shared model)", SHARED_POP)):
        for cv in IC50_CVS:
            cfg = main_env_config(pop=pop.with_(cv_ic50=cv))
            for label, pol in (("Ceiling", ceiling_regimen()), ("Standard", standard_regimen())):
                p = float(np.mean(run_policy(pol, cohort, cfg).b_true[:, -1] < TARGET))
                lo, hi = wilson(p, n)
                rows.append({"setting": setting, "ic50_cv": cv, "regimen": label, "pct_at_target": 100 * p,
                             "lo": 100 * lo, "hi": 100 * hi, "n": n})
    return pd.DataFrame(rows)


def misses(patients: pd.DataFrame, break_even: float) -> pd.DataFrame:
    """Split each policy's incorrect decisions into kinds. DQN seeds are averaged."""
    p = patients.copy()
    p["group"] = np.where(p.policy.str.startswith("DQN seed"), "DQN", p.policy)
    kept = ~p.stopped & ~p.at_target
    kinds = {
        "Reachable, stopped": p.reachable & p.stopped,
        "Reachable, kept on drug, finished 200 to break-even": p.reachable & kept & (p.final_b < break_even),
        "Reachable, kept on drug, finished above break-even": p.reachable & kept & (p.final_b >= break_even),
        "Unreachable, kept on drug, finished below break-even": ~p.reachable & kept & (p.final_b < break_even),
        "Unreachable, kept on drug, finished above break-even": ~p.reachable & kept & (p.final_b >= break_even),
    }
    rows = []
    for g, d in p.groupby("group"):
        n_seeds = d.policy.nunique()
        n = len(d) / n_seeds
        row = {"policy": g, "n_patients": int(n), "incorrect": float((~d.correct).sum() / n_seeds)}
        for k, m in kinds.items():
            row[k] = float(m.loc[d.index].sum() / n_seeds)
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    res = ROOT / ("results_quick" if args.quick else "results")

    lad = ic50_ladder(600 if args.quick else 4000)
    lad.to_csv(res / "ic50_ladder.csv", index=False)
    print(lad[lad.regimen == "Ceiling"].pivot(index="ic50_cv", columns="setting", values="pct_at_target").round(1))

    cfg = main_env_config()
    break_even = TARGET * 2**cfg.stop_cost
    pats = pd.read_csv(res / "patients_main.csv").reset_index(drop=True)
    keep = pats.policy.isin(["MAP-Bayes", "Oracle", "Standard"]) | pats.policy.str.startswith("DQN seed")
    m = misses(pats[keep].reset_index(drop=True), break_even)
    m.insert(1, "break_even", break_even)
    m.to_csv(res / "misses.csv", index=False)
    print(m.round(1).to_string())

    front = pd.read_csv(res / "frontier.csv")
    cost = front[front.family == "DQN"][["label", "drug_cost", "pct_at_target", "pct_correct", "pct_stopped",
                                         "dose_rate"]]
    cost.to_csv(res / "cost_tradeoff.csv", index=False)
    print(cost.round(2).to_string())


if __name__ == "__main__":
    main()
