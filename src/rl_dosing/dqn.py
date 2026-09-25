"""DQN with a shared trunk and two heads (induction dose, maintenance regimen or stop).

Double DQN targets, a Huber loss, a uniform replay buffer, a target network
and epsilon-greedy exploration that respects the action mask. Training runs a
batch of virtual patients through the environment in lockstep, which is much
faster in numpy than one patient at a time. The checkpoint kept is the one
with the best greedy mean return on a fixed evaluation cohort.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn

from .env import N_IND, N_MAINT, N_OCC, OBS_DIM, DosingEnv, EnvConfig, run_policy
from .model import ProfilePool, sample_cohort


@dataclass
class DQNConfig:
    episodes: int = 12_000
    parallel: int = 32  # patients simulated together
    gamma: float = 0.97
    lr: float = 5e-4
    lr_final: float = 5e-5
    batch_size: int = 128
    buffer_size: int = 50_000
    warmup_episodes: int = 800  # pure random actions, no updates
    updates_per_step: int = 12  # gradient steps per batch env step (32 transitions)
    target_sync: int = 400  # gradient steps between target network copies
    eps_start: float = 1.0
    eps_end: float = 0.03
    eps_decay_frac: float = 0.6
    hidden: tuple = (128, 64)
    grad_clip: float = 5.0
    eval_every: int = 400
    seed: int = 0


class QNet(nn.Module):
    def __init__(self, hidden=(128, 64)):
        super().__init__()
        layers, d = [], OBS_DIM
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        self.trunk = nn.Sequential(*layers)
        self.head_ind = nn.Linear(d, N_IND)
        self.head_maint = nn.Linear(d, N_MAINT)

    def forward(self, x):
        z = self.trunk(x)
        return self.head_ind(z), self.head_maint(z)


def masked_argmax(q: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return q.masked_fill(~mask, -1e9).argmax(1)


class DQNPolicy:
    """Greedy policy from a trained network; uses only env.observe() and env.mask()."""

    def __init__(self, net: QNet, name: str = "DQN"):
        self.net, self.name = net.eval(), name

    @torch.no_grad()
    def act(self, env: DosingEnv):
        obs = torch.as_tensor(env.observe())
        mask = torch.as_tensor(env.mask())
        q_ind, q_maint = self.net(obs)
        q = q_ind if env.decision == 0 else q_maint
        return masked_argmax(q, mask).numpy()


class Replay:
    def __init__(self, size: int, rng: np.random.Generator):
        self.obs = np.zeros((size, OBS_DIM), np.float32)
        self.next_obs = np.zeros((size, OBS_DIM), np.float32)
        self.phase = np.zeros(size, np.int64)  # 0 induction, 1 maintenance
        self.action = np.zeros(size, np.int64)
        self.reward = np.zeros(size, np.float32)
        self.done = np.zeros(size, np.float32)
        self.next_mask = np.zeros((size, N_MAINT), bool)  # the next decision is always a maintenance one
        self.size, self.pos, self.n, self.rng = size, 0, 0, rng

    def add(self, obs, phase, action, reward, next_obs, done, next_mask):
        m = len(obs)
        idx = (self.pos + np.arange(m)) % self.size
        self.obs[idx], self.phase[idx], self.action[idx] = obs, phase, action
        self.reward[idx], self.next_obs[idx], self.done[idx], self.next_mask[idx] = reward, next_obs, done, next_mask
        self.pos = (self.pos + m) % self.size
        self.n = min(self.n + m, self.size)

    def sample(self, k: int):
        i = self.rng.integers(self.n, size=k)
        t = torch.from_numpy
        return (t(self.obs[i]), t(self.phase[i]), t(self.action[i]), t(self.reward[i]), t(self.next_obs[i]),
                t(self.done[i]), t(self.next_mask[i]))


def td_loss(net: QNet, target: QNet, batch, gamma: float) -> torch.Tensor:
    obs, phase, action, reward, next_obs, done, next_mask = batch
    q_ind, q_maint = net(obs)
    # each transition only trains the head its decision used
    a_ind = torch.where(phase == 0, action, torch.zeros_like(action))
    a_mnt = torch.where(phase == 1, action, torch.zeros_like(action))
    q = torch.where(phase == 0, q_ind.gather(1, a_ind[:, None]).squeeze(1),
                    q_maint.gather(1, a_mnt[:, None]).squeeze(1))
    with torch.no_grad():
        _, q_next_online = net(next_obs)
        _, q_next_target = target(next_obs)
        a2 = masked_argmax(q_next_online, next_mask)
        y = reward + gamma * (1 - done) * q_next_target.gather(1, a2[:, None]).squeeze(1)
    return nn.functional.smooth_l1_loss(q, y)


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


def evaluate_greedy(net: QNet, cohort, env_cfg: EnvConfig, reachable: np.ndarray) -> dict:
    env = run_policy(DQNPolicy(net), cohort, env_cfg)
    net.train()
    at = (~env.stopped) & (env.b_true[:, -1] < 200)
    correct = at | (~reachable & env.stopped)
    weeks = np.nanmax(env.visit_day, 1) / 7
    return {"return": float(env.rewards.sum(1).mean()), "pct_at_target": 100 * at.mean(),
            "pct_correct": 100 * correct.mean(), "pct_stopped": 100 * env.stopped.mean(),
            "dose_rate": float(np.mean(np.nansum(env.dose, 1) / weeks))}


def train_dqn(cfg: DQNConfig, env_cfg: EnvConfig, train_pool: ProfilePool, eval_cohort,
              eval_reachable: np.ndarray, verbose: bool = True):
    """Returns (best network, learning curve DataFrame, settings dict)."""
    set_seed(cfg.seed)
    rng = np.random.default_rng(10_000 + cfg.seed)
    net, target = QNet(cfg.hidden), QNet(cfg.hidden)
    target.load_state_dict(net.state_dict())
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    n_rounds = int(np.ceil(cfg.episodes / cfg.parallel))
    gamma_lr = (cfg.lr_final / cfg.lr) ** (1 / max(1, n_rounds))
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma_lr)
    replay = Replay(cfg.buffer_size, rng)

    curve, best, best_state = [], -np.inf, None
    n_updates, episodes, next_eval = 0, 0, cfg.eval_every
    recent = []
    t0 = time.time()
    for _ in range(n_rounds):
        frac = min(1.0, max(0, episodes - cfg.warmup_episodes) / (cfg.eps_decay_frac * cfg.episodes))
        eps = 1.0 if episodes < cfg.warmup_episodes else cfg.eps_start + frac * (cfg.eps_end - cfg.eps_start)
        cohort = sample_cohort(cfg.parallel, train_pool, rng, N_OCC)
        env = DosingEnv(cohort, env_cfg)
        obs, mask = env.observe(), env.mask()
        while env.active.any():
            who = env.active.copy()
            phase = 0 if env.decision == 0 else 1
            with torch.no_grad():
                q = net(torch.as_tensor(obs))[phase]
            greedy = masked_argmax(q, torch.as_tensor(mask)).numpy()
            # random actions are uniform over the allowed ones
            rand = np.array([rng.choice(np.flatnonzero(m)) for m in mask])
            act = np.where(rng.random(env.n) < eps, rand, greedy)
            next_obs, rew, done, next_mask = env.step(act)
            replay.add(obs[who], np.full(who.sum(), phase), act[who], rew[who], next_obs[who],
                       done[who].astype(np.float32), next_mask[who])
            obs, mask = next_obs, next_mask
            if episodes >= cfg.warmup_episodes and replay.n >= cfg.batch_size:
                for _ in range(cfg.updates_per_step):
                    loss = td_loss(net, target, replay.sample(cfg.batch_size), cfg.gamma)
                    opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip)
                    opt.step()
                    n_updates += 1
                    if n_updates % cfg.target_sync == 0:
                        target.load_state_dict(net.state_dict())
        if episodes >= cfg.warmup_episodes:
            sched.step()
        recent.append(env.rewards.sum(1).mean())
        episodes += cfg.parallel

        if episodes >= next_eval or episodes >= cfg.episodes:
            next_eval += cfg.eval_every
            m = evaluate_greedy(net, eval_cohort, env_cfg, eval_reachable)
            row = {"episode": episodes, "epsilon": eps, "train_return": float(np.mean(recent)), **m,
                   "seconds": time.time() - t0}
            curve.append(row)
            recent = []
            if m["return"] > best:
                best, best_state = m["return"], {k: v.clone() for k, v in net.state_dict().items()}
            if verbose:
                print(f"seed {cfg.seed} ep {episodes:6d} eps {eps:.2f} eval return {m['return']:6.3f} "
                      f"at target {m['pct_at_target']:5.1f}% correct {m['pct_correct']:5.1f}% "
                      f"stopped {m['pct_stopped']:4.1f}% rate {m['dose_rate']:.2f} {row['seconds']:5.0f}s",
                      flush=True)
    net.load_state_dict(best_state)
    info = asdict(cfg)
    info["best_eval_return"] = best
    info["best_episode"] = int(pd.DataFrame(curve).loc[lambda d: d["return"].idxmax(), "episode"])
    info["train_seconds"] = time.time() - t0
    return net.eval(), pd.DataFrame(curve), info

