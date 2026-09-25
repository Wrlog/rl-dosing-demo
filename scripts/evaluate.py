"""Score every policy on the held-out patients and write the result tables.

    python scripts/evaluate.py [--quick]

Scenarios (all on patients never used in training):
  main         held-out covariate profiles, main settings
  seen         training covariate profiles with fresh random effects
  flares       held-out profiles with flares switched on
  shared_ic50  held-out profiles with the shared-model IC50 of 3 mg/L
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from rl_dosing.dqn import DQNPolicy, QNet
from rl_dosing.env import INTERVALS_WK, MAINT_DOSES, N_OCC, run_policy
from rl_dosing.experiment import COST_SWEEP, RUNS, env_config_for, make_setup, reachable
from rl_dosing.metrics import paired_difference, patient_table, summarize
from rl_dosing.model import pd_curve
from rl_dosing.policies import FixedRegimen, MAPBayes, Oracle, ceiling_regimen, standard_regimen

ROOT = Path(__file__).resolve().parents[1]
MARGINS = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5)


def load_dqn(models: Path, name: str, label: str) -> DQNPolicy:
    info = json.loads((models / f"{name}.json").read_text())
    net = QNet(tuple(info["hidden"]))
    net.load_state_dict(torch.load(models / f"{name}.pt"))
    return DQNPolicy(net, label)


def tune_margin(make, cohort, cfg, reach):
    """Pick the margin with the best mean return on the evaluation cohort (the DQN's checkpoint cohort)."""
    rows = []
    for m in MARGINS:
        t = patient_table(run_policy(make(m), cohort, cfg), reach)
        rows.append({"margin": m, "mean_return": t["ret"].mean(), "pct_at_target": 100 * t["at_target"].mean(),
                     "pct_correct": 100 * t["correct"].mean(), "dose_rate": t["dose_rate"].mean()})
    df = pd.DataFrame(rows)
    return float(df.loc[df["mean_return"].idxmax(), "margin"]), df


def example_records(envs: dict, patients: list[int]):
    """Per-visit records and 1-day biomarker curves for a few patients."""
    visits, curves = [], []
    for label, env in envs.items():
        for case, i in enumerate(patients):
            for v in range(N_OCC + 1):
                if np.isnan(env.visit_day[i, v]):
                    break
                visits.append({"case": case, "patient": i, "policy": label, "visit": v,
                               "day": env.visit_day[i, v], "b_true": env.b_true[i, v], "b_obs": env.obs_bio[i, v],
                               "dose_mgkg": env.dose[i, v] if v < N_OCC else np.nan,
                               "tau_days": env.tau[i, v] if v < N_OCC else np.nan,
                               "stopped_here": bool(env.stopped[i] and v == 3 + env.stop_decision[i] - 1)})
            for k in range(N_OCC):
                if np.isnan(env.dose[i, k]):
                    break
                t = np.arange(0.0, env.tau[i, k] + 1e-9, 1.0)
                b = pd_curve(env.b_true[i, k], env.cave[i, k], t, env.base_occ[i, k], env.ic50[i], env.cfg.pop.kout)
                curves += [{"case": case, "policy": label, "day": env.visit_day[i, k] + tt, "b_true": bb}
                           for tt, bb in zip(t, b)]
    return pd.DataFrame(visits), pd.DataFrame(curves)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    models = ROOT / ("models_quick" if args.quick else "models")
    res = ROOT / ("results_quick" if args.quick else "results")
    res.mkdir(exist_ok=True)
    n_boot = 200 if args.quick else 1000
    rng = np.random.default_rng(2026)
    t0 = time.time()
    setup = make_setup(args.quick)
    main_cfg = env_config_for({})
    shared_cfg = env_config_for({"shared_ic50": True})
    flare_cfg = env_config_for({"flares": True})

    # margins for MAP-Bayes and the oracle, tuned on the evaluation cohort
    margins, tuning = {}, []
    for key, cfg in (("main", main_cfg), ("shared_ic50", shared_cfg)):
        reach_eval = reachable(setup.eval_cohort, cfg)
        for pname, make in (("MAP-Bayes", lambda m, c=cfg: MAPBayes(c.pop, m)), ("Oracle", lambda m: Oracle(m))):
            m, df = tune_margin(make, setup.eval_cohort, cfg, reach_eval)
            margins[(key, pname)] = m
            tuning.append(df.assign(policy=pname, setting=key))
    pd.concat(tuning).to_csv(res / "margin_tuning.csv", index=False)
    print("margins", margins, f"{time.time() - t0:.0f} s", flush=True)

    # reference lines for the learning curves (evaluation cohort, main settings)
    reach_eval = reachable(setup.eval_cohort, main_cfg)
    refs = [{"policy": "Ceiling (reachable)", "pct_at_target": 100 * reach_eval.mean(),
             "pct_correct": 100 * reach_eval.mean()}]
    for pol in (standard_regimen(), MAPBayes(main_cfg.pop, margins[("main", "MAP-Bayes")]),
                Oracle(margins[("main", "Oracle")])):
        t = patient_table(run_policy(pol, setup.eval_cohort, main_cfg), reach_eval)
        refs.append({"policy": pol.name, "pct_at_target": 100 * t["at_target"].mean(),
                     "pct_correct": 100 * t["correct"].mean(), "mean_return": t["ret"].mean()})
    pd.DataFrame(refs).to_csv(res / "eval_references.csv", index=False)

    dqn_main = [load_dqn(models, f"dqn_seed{s}", f"DQN seed {s}") for s in (0, 1, 2)]
    scenarios = {
        "main": (setup.test_heldout, main_cfg, "main"),
        "seen": (setup.test_seen, main_cfg, "main"),
        "flares": (setup.test_heldout, flare_cfg, "main"),
        "shared_ic50": (setup.test_heldout, shared_cfg, "shared_ic50"),
    }
    all_paired = []
    for scen, (cohort, cfg, mkey) in scenarios.items():
        reach = reachable(cohort, cfg)
        policies = [standard_regimen(), ceiling_regimen(), MAPBayes(cfg.pop, margins[(mkey, "MAP-Bayes")]),
                    Oracle(margins[(mkey, "Oracle")])]
        if scen == "shared_ic50":
            dqns = [load_dqn(models, "dqn_shared_ic50", "DQN seed 0")]
        else:
            dqns = dqn_main
        if scen == "flares":
            policies.append(load_dqn(models, "dqn_flares", "DQN trained with flares"))
        envs, tables = {}, {}
        for pol in policies + dqns:
            env = run_policy(pol, cohort, cfg)
            envs[pol.name] = env
            tables[pol.name] = patient_table(env, reach)
        groups = {p.name: [tables[p.name]] for p in policies}
        groups["DQN"] = [tables[p.name] for p in dqns]
        if len(dqns) > 1:
            groups.update({p.name: [tables[p.name]] for p in dqns})
        rows = [{"policy": k, "n_seeds": len(v), **summarize(v, n_boot, rng)} for k, v in groups.items()]
        pd.DataFrame(rows).to_csv(res / f"summary_{scen}.csv", index=False)
        long = pd.concat([t.assign(policy=k) for k, t in tables.items()])
        long.to_csv(res / f"patients_{scen}.csv", index=False)
        pairs = [("DQN", "MAP-Bayes"), ("DQN", "Standard"), ("MAP-Bayes", "Standard"), ("Oracle", "DQN")]
        if scen == "flares":
            pairs.append(("DQN trained with flares", "MAP-Bayes"))
        for a, b in pairs:
            all_paired.append({"scenario": scen, "a": a, "b": b,
                               **paired_difference(groups[a], groups[b], n_boot, rng)})
        print(f"{scen}: reachable {100 * reach.mean():.1f}%  "
              + "  ".join(f"{r['policy']} {r['pct_at_target']:.1f}/{r['pct_correct']:.1f}" for r in rows[:5])
              + f"  {time.time() - t0:.0f} s", flush=True)

        if scen == "main":
            main_reach, main_cohort, main_tables = reach, cohort, tables
            # example patients: reachable and missed by the standard regimen; unreachable; fine on standard
            std = tables["Standard"]
            cases = [int(std.index[(std.reachable) & (~std.at_target)][0]),
                     int(std.index[~std.reachable][0]),
                     int(std.index[std.at_target & std.reachable][0])]
            ex = {k: envs[k] for k in ("Standard", "MAP-Bayes", "DQN seed 0")}
            v, c = example_records(ex, cases)
            v.to_csv(res / "example_visits.csv", index=False)
            c.to_csv(res / "example_curves.csv", index=False)
    pd.DataFrame(all_paired).to_csv(res / "paired_differences.csv", index=False)

    # correct decisions against dose rate: fixed regimens, MAP margins, DQN drug costs
    front = []
    for d in MAINT_DOSES:
        for w in INTERVALS_WK:
            t = patient_table(run_policy(FixedRegimen(d, d, w), main_cohort, main_cfg), main_reach)
            front.append({"family": "Fixed regimen", "label": f"{d:g} q{w}w", **summarize([t], 0, rng)})
    for m in MARGINS:
        t = patient_table(run_policy(MAPBayes(main_cfg.pop, m), main_cohort, main_cfg), main_reach)
        front.append({"family": "MAP-Bayes", "label": f"margin {m:g}", **summarize([t], 0, rng)})
    for c in COST_SWEEP:
        names = [f"dqn_seed{s}" for s in (0, 1, 2)] if c == 0.05 else [f"dqn_cost{c:g}"]
        ts = [main_tables[f"DQN seed {s}"] for s in (0, 1, 2)] if c == 0.05 else \
            [patient_table(run_policy(load_dqn(models, n, n), main_cohort, main_cfg), main_reach) for n in names]
        front.append({"family": "DQN", "label": f"drug cost {c:g}", "drug_cost": c, **summarize(ts, 0, rng)})
    pd.DataFrame(front).to_csv(res / "frontier.csv", index=False)
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
