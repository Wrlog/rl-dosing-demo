from dataclasses import replace

import numpy as np
import pytest

from rl_dosing.env import (N_IND, N_MAINT, N_MAINT_DECISIONS, OBS_DIM, STOP, DosingEnv, EnvConfig, regimen_index,
                           run_policy)
from rl_dosing.experiment import MAIN_POP, main_env_config
from rl_dosing.model import TARGET, pd_interval, sample_cohort, sample_profile_pool
from rl_dosing.policies import ceiling_regimen, standard_regimen


def cohort(n=40, seed=0):
    pool = sample_profile_pool(100, np.random.default_rng(seed))
    return sample_cohort(n, pool, np.random.default_rng(seed + 1), 7)


def exact_cfg(**kw):
    return main_env_config(measurement_error=False, **kw)


def test_biomarker_is_recorded_at_the_end_of_each_interval():
    """The biomarker a step returns belongs to the interval that step just ran."""
    c = cohort()
    envs = []
    for dose in (2.5, 15.0):
        env = DosingEnv(c, exact_cfg())
        env.step(np.full(env.n, 1))  # induction 5 mg/kg
        obs, *_ = env.step(np.full(env.n, regimen_index(dose, 6)))
        envs.append((env, obs))
    (lo, obs_lo), (hi, obs_hi) = envs
    # visits up to day 98 are the same; the visit after the changed interval differs
    assert np.allclose(lo.b_true[:, :4], hi.b_true[:, :4])
    assert np.all(hi.b_true[:, 4] < lo.b_true[:, 4])
    # and it equals the exact PD update over that interval, driven by that interval's Cave
    for env in (lo, hi):
        expect = pd_interval(env.b_true[:, 3], env.cave[:, 3], env.tau[:, 3], env.base_occ[:, 3], env.ic50,
                             env.cfg.pop.kout)
        assert np.allclose(env.b_true[:, 4], expect)
        assert np.allclose(env.obs_bio[:, 4], env.b_true[:, 4])  # no error in this config
        assert np.allclose(env.visit_day[:, 4], 98 + 42)
    # the observation after the step already contains the new value
    assert np.allclose(obs_lo[:, 4], np.log(lo.obs_bio[:, 4] / TARGET), atol=1e-5)
    assert not np.allclose(obs_lo[:, 4], obs_hi[:, 4])


def test_arms_are_paired():
    c = cohort()
    a = run_policy(standard_regimen(), c, main_env_config())
    b = run_policy(standard_regimen(), c, main_env_config())
    assert np.array_equal(a.obs_bio, b.obs_bio)
    ceil = run_policy(ceiling_regimen(), c, main_env_config())
    assert np.array_equal(a.obs_bio[:, 0], ceil.obs_bio[:, 0])  # same baseline measurement


def test_stop_is_masked_until_after_induction_and_when_disallowed():
    env = DosingEnv(cohort(), main_env_config())
    m = env.mask()
    assert m.shape[1] == N_IND  # the induction head has no stop action
    env.step(np.zeros(env.n, int))
    m = env.mask()
    assert m.shape[1] == N_MAINT and m[:, STOP].all()
    env2 = DosingEnv(cohort(), main_env_config(allow_stop=False))
    env2.step(np.zeros(env2.n, int))
    assert not env2.mask()[:, STOP].any()
    with pytest.raises(ValueError):
        env2.step(np.full(env2.n, STOP))


def test_stop_ends_the_episode_with_the_stop_payoff():
    cfg = main_env_config()
    env = DosingEnv(cohort(), cfg)
    env.step(np.zeros(env.n, int))
    act = np.where(np.arange(env.n) % 2 == 0, STOP, 0)
    _, r, done, _ = env.step(act)
    stop = act == STOP
    assert np.allclose(r[stop], -cfg.stop_cost * N_MAINT_DECISIONS)
    assert done[stop].all() and not done[~stop].any()
    # stopped patients get nothing more
    _, r2, _, _ = env.step(np.zeros(env.n, int))
    assert np.all(r2[stop] == 0) and np.isnan(env.dose[stop, 4]).all()


def test_reward_is_zero_below_target_without_drug_cost():
    env = DosingEnv(cohort(), exact_cfg(drug_cost=0.0))
    _, r, _, _ = env.step(np.full(env.n, N_IND - 1))
    below = env.b_true[:, 3] < TARGET
    assert below.any() and np.all(r[below] == 0)
    assert np.all(r[~below] < 0)
    assert np.all(r >= -env.cfg.penalty_cap)


def test_observation_uses_measurements_not_true_parameters():
    env = DosingEnv(cohort(), main_env_config())
    env.step(np.full(env.n, 2))
    obs = env.observe()
    assert obs.shape == (env.n, OBS_DIM) and np.isfinite(obs).all()
    env.cl_ind *= 3
    env.ic50 *= 5
    env.base *= 2
    env.b *= 0.1
    assert np.array_equal(obs, env.observe())


def test_ceiling_reaches_more_patients_than_standard():
    c = cohort(300)
    cfg = main_env_config()
    std = run_policy(standard_regimen(), c, cfg).b_true[:, -1] < TARGET
    ceil = run_policy(ceiling_regimen(), c, cfg).b_true[:, -1] < TARGET
    assert ceil.mean() > std.mean()
    assert MAIN_POP.ic50 == 10.0


def test_flares_change_base_between_occasions():
    c = cohort()
    cfg = main_env_config(pop=replace(MAIN_POP, cv_flare=0.35))
    env = run_policy(standard_regimen(), c, cfg)
    ratio = env.base_occ / env.base[:, None]
    assert ratio.std() > 0.1
    env0 = run_policy(standard_regimen(), c, main_env_config())
    assert np.allclose(env0.base_occ / env0.base[:, None], 1.0)


def test_env_config_defaults():
    cfg = EnvConfig()
    assert cfg.allow_stop and cfg.drug_cost >= 0
