import numpy as np
import torch

from rl_dosing.dqn import DQNConfig, DQNPolicy, QNet, Replay, td_loss, train_dqn
from rl_dosing.env import N_IND, N_MAINT, OBS_DIM, STOP, DosingEnv, run_policy
from rl_dosing.experiment import main_env_config, make_setup, reachable
from rl_dosing.model import sample_cohort, sample_profile_pool
from rl_dosing.policies import MAPBayes, Oracle, batched_minimize, standard_regimen


def test_batched_minimize_finds_quadratic_minimum():
    target = np.array([[0.5, -1.0], [2.0, 0.3], [-0.7, 0.0]])

    def f(x):
        d = x - target
        return 3 * d[:, 0] ** 2 + d[:, 1] ** 2 + d[:, 0] * d[:, 1]

    x, _ = batched_minimize(f, np.zeros_like(target), n_iter=30)
    assert np.allclose(x, target, atol=1e-4)


def test_map_recovers_clearance_from_noise_free_troughs():
    pool = sample_profile_pool(200, np.random.default_rng(5))
    c = sample_cohort(150, pool, np.random.default_rng(6), 7)
    cfg = main_env_config(measurement_error=False)
    cfg.pop = cfg.pop.with_(cv_iov_cl=0.0, ada_p_max=0.0)
    env = DosingEnv(c, cfg)
    pol = standard_regimen()
    for _ in range(4):
        env.step(pol.act(env))
    idx = np.arange(env.n)
    eta, _ = MAPBayes(cfg.pop).estimate(env, idx)
    true_cl = cfg.pop.om_cl * c.z_cl
    use = ~env.blq[:, 1:5].any(1)
    assert np.corrcoef(eta[use, 0], true_cl[use])[0, 1] > 0.95
    assert np.median(np.abs(eta[use, 0] - true_cl[use])) < 0.05


def test_map_and_oracle_only_choose_allowed_actions():
    s = make_setup(quick=True)
    cfg = main_env_config()
    for pol in (MAPBayes(cfg.pop), Oracle()):
        env = run_policy(pol, s.eval_cohort.subset(np.arange(60)), cfg)
        assert (env.actions[:, 0] < N_IND).all()
    env = run_policy(Oracle(), s.eval_cohort.subset(np.arange(60)), main_env_config(allow_stop=False))
    assert not env.stopped.any()


def test_greedy_dqn_respects_the_mask():
    net = QNet()
    with torch.no_grad():
        net.head_maint.bias[STOP] = 1e6  # the network badly wants to stop
    pool = sample_profile_pool(30, np.random.default_rng(0))
    c = sample_cohort(20, pool, np.random.default_rng(1), 7)
    env = run_policy(DQNPolicy(net), c, main_env_config(allow_stop=False))
    assert not env.stopped.any()
    env = run_policy(DQNPolicy(net), c, main_env_config())
    assert env.stopped.all()


def test_induction_transitions_only_train_the_induction_head():
    rng = np.random.default_rng(0)
    rb = Replay(64, rng)
    k = 32
    rb.add(rng.standard_normal((k, OBS_DIM)).astype(np.float32), np.zeros(k, int), rng.integers(N_IND, size=k),
           rng.standard_normal(k), rng.standard_normal((k, OBS_DIM)).astype(np.float32), np.zeros(k),
           np.ones((k, N_MAINT), bool))
    net, tgt = QNet(), QNet()
    loss = td_loss(net, tgt, rb.sample(16), 0.97)
    loss.backward()
    assert net.head_maint.weight.grad is None or torch.all(net.head_maint.weight.grad == 0)
    assert net.head_ind.weight.grad.abs().sum() > 0


def test_short_training_run_is_reproducible():
    s = make_setup(quick=True)
    cfg = main_env_config()
    ev = s.eval_cohort.subset(np.arange(60))
    reach = reachable(ev, cfg)
    dc = DQNConfig(episodes=192, parallel=32, warmup_episodes=64, eval_every=96, updates_per_step=2)
    net1, curve1, _ = train_dqn(dc, cfg, s.train_pool, ev, reach, verbose=False)
    net2, curve2, _ = train_dqn(dc, cfg, s.train_pool, ev, reach, verbose=False)
    assert len(curve1) == 2
    assert np.allclose(curve1["return"], curve2["return"])
    env = run_policy(DQNPolicy(net1), ev, cfg)
    assert np.isfinite(env.rewards).all()
