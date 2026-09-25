"""Train every DQN in experiment.RUNS (in parallel processes) and save networks and learning curves.

    python scripts/train.py            # full run
    python scripts/train.py --quick    # short run for CI
    python scripts/train.py --only dqn_seed0 --jobs 1
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def train_one(name: str, quick: bool) -> str:
    import torch

    from rl_dosing.dqn import DQNConfig, train_dqn
    from rl_dosing.experiment import EPISODES, EPISODES_QUICK, RUNS, env_config_for, make_setup, reachable

    spec = RUNS[name]
    setup = make_setup(quick)
    cfg = env_config_for(spec)
    reach = reachable(setup.eval_cohort, cfg)
    dcfg = DQNConfig(seed=spec["seed"], episodes=EPISODES_QUICK if quick else EPISODES,
                     eval_every=200 if quick else 400, warmup_episodes=200 if quick else 800)
    net, curve, info = train_dqn(dcfg, cfg, setup.train_pool, setup.eval_cohort, reach, verbose=False)
    out = ROOT / ("models_quick" if quick else "models")
    res = ROOT / ("results_quick" if quick else "results")
    out.mkdir(exist_ok=True)
    res.mkdir(exist_ok=True)
    torch.save(net.state_dict(), out / f"{name}.pt")
    info.update(spec=spec, eval_pct_reachable=100 * float(reach.mean()), drug_cost=cfg.drug_cost,
                ic50_typical=cfg.pop.ic50, cv_flare=cfg.pop.cv_flare)
    (out / f"{name}.json").write_text(json.dumps(info, indent=2, default=str))
    curve.insert(0, "run", name)
    curve.to_csv(res / f"learning_curve_{name}.csv", index=False)
    return (f"{name}: best eval return {info['best_eval_return']:.3f} at episode {info['best_episode']}, "
            f"{info['train_seconds']:.0f} s")


def main():
    from rl_dosing.experiment import RUNS

    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--jobs", type=int, default=min(len(RUNS), max(1, (os.cpu_count() or 2) - 1)))
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()
    names = args.only or list(RUNS)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for msg in pool.map(train_one, names, [args.quick] * len(names)):
            print(msg, flush=True)
    print(f"trained {len(names)} networks in {time.time() - t0:.0f} s with {args.jobs} processes")


if __name__ == "__main__":
    main()
