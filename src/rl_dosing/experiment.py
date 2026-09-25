"""The fixed pieces of the experiment: profile pool, split, cohorts and main settings."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .env import N_OCC, EnvConfig, run_policy
from .model import POP, TARGET, Cohort, ProfilePool, sample_cohort, sample_profile_pool, split_pool
from .policies import ceiling_regimen

# The one change from the shared Drug X model: typical IC50 10 mg/L instead of 3.
# With 3 mg/L almost every patient reaches target on the ceiling regimen, and
# the stop decision has nothing to do. The shared value is run as a sensitivity.
MAIN_POP = POP.with_(ic50=10.0)
SHARED_POP = POP

N_PROFILES = 1200
HELD_OUT_FRAC = 0.25
POOL_SEED = 7
EVAL_SEED = 11
TEST_SEEN_SEED = 23
TEST_HELDOUT_SEED = 29


def main_env_config(**kw) -> EnvConfig:
    cfg = EnvConfig(pop=MAIN_POP, stop_cost=0.35, drug_cost=0.05)
    return replace(cfg, **kw)


@dataclass
class Setup:
    train_pool: ProfilePool
    heldout_pool: ProfilePool
    eval_cohort: Cohort  # fixed cohort for checkpoint selection (training profiles)
    test_heldout: Cohort  # main test cohort: held-out profiles, fresh random effects
    test_seen: Cohort  # training profiles, fresh random effects


def make_setup(quick: bool = False) -> Setup:
    pool = sample_profile_pool(N_PROFILES, np.random.default_rng(POOL_SEED))
    train_pool, heldout_pool = split_pool(pool, HELD_OUT_FRAC, np.random.default_rng(POOL_SEED + 1))
    n_eval, n_test = (200, 300) if quick else (400, 1500)
    return Setup(
        train_pool=train_pool,
        heldout_pool=heldout_pool,
        eval_cohort=sample_cohort(n_eval, train_pool, np.random.default_rng(EVAL_SEED), N_OCC, cycle_profiles=True),
        test_heldout=sample_cohort(n_test, heldout_pool, np.random.default_rng(TEST_HELDOUT_SEED), N_OCC,
                                   cycle_profiles=True),
        test_seen=sample_cohort(n_test, train_pool, np.random.default_rng(TEST_SEEN_SEED), N_OCC,
                                cycle_profiles=True),
    )


def reachable(cohort: Cohort, cfg: EnvConfig) -> np.ndarray:
    """True where the ceiling regimen brings the true biomarker below target at its final visit.

    The ceiling run uses the same patients and the same random draws (IOV,
    flares, ADA onset) as every other arm.
    """
    env = run_policy(ceiling_regimen(), cohort, cfg)
    return env.b_true[:, -1] < TARGET


# ----------------------------------------------------------------------------
# Training runs. Each is a seed plus changes to the main settings.
# ----------------------------------------------------------------------------

FLARE_CV = 0.35  # between-occasion variability on BASE when flares are on
MAIN_SEEDS = (0, 1, 2)
COST_SWEEP = (0.0, 0.05, 0.15, 0.4)  # drug cost per mg/kg/week; 0.05 is the main setting
EPISODES, EPISODES_QUICK = 16_000, 2_000

RUNS = {
    **{f"dqn_seed{s}": {"seed": s} for s in MAIN_SEEDS},
    **{f"dqn_cost{c:g}": {"seed": 0, "drug_cost": c} for c in COST_SWEEP if c != 0.05},
    "dqn_flares": {"seed": 0, "flares": True},
    "dqn_shared_ic50": {"seed": 0, "shared_ic50": True},
}


def env_config_for(spec: dict) -> EnvConfig:
    pop = SHARED_POP if spec.get("shared_ic50") else MAIN_POP
    if spec.get("flares"):
        pop = pop.with_(cv_flare=FLARE_CV)
    return main_env_config(pop=pop, drug_cost=spec.get("drug_cost", 0.05))
