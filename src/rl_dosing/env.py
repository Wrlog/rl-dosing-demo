"""Dosing as a decision process, on a batch of virtual patients at once.

One episode is one patient. There are five decisions:

- day 0: the induction dose (mg/kg), given at days 0, 21 and 42;
- day 98 and the next three visits: a maintenance regimen (dose x interval),
  or stop Drug X and move the patient to another treatment.

The patient is seen at every infusion. At each visit we measure a trough
concentration and the biomarker, both with error, and ADA status. The
biomarker measured at a visit is the one at the end of the interval that
just finished, so it reflects the Cave of that interval.

All patients in a batch make their decisions at the same decision index, but
each carries its own clock because intervals differ.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .model import LLOQ, POP, TARGET, Cohort, PopParams, cv_to_sd, pd_interval, pk_interval
from .model import typical_base, typical_cl, typical_ic50, typical_q, typical_v1, typical_v2

IND_DOSES = (2.5, 5.0, 7.5, 10.0, 12.5, 15.0)  # mg/kg
MAINT_DOSES = (2.5, 5.0, 7.5, 10.0, 12.5, 15.0)  # mg/kg
INTERVALS_WK = (4, 6, 8, 10)
REGIMENS = tuple((d, w) for d in MAINT_DOSES for w in INTERVALS_WK)
STOP = len(REGIMENS)
N_IND = len(IND_DOSES)
N_MAINT = len(REGIMENS) + 1  # regimens plus stop

INDUCTION_TAUS = (21.0, 21.0, 56.0)  # days; maintenance starts at day 98
N_MAINT_DECISIONS = 4
N_DECISIONS = 1 + N_MAINT_DECISIONS
N_OCC = len(INDUCTION_TAUS) + N_MAINT_DECISIONS
N_VISITS = N_OCC + 1
OBS_DIM = 16

REG_DOSE = np.array([d for d, _ in REGIMENS])
REG_TAU = np.array([7.0 * w for _, w in REGIMENS])
REG_RATE = REG_DOSE / (REG_TAU / 7.0)  # mg/kg per week
# Maintenance regimens from cheapest to most intense; ties go to the longer interval.
REG_BY_COST = np.array(sorted(range(len(REGIMENS)), key=lambda a: (REG_RATE[a], -REG_TAU[a])))
STRONGEST = REGIMENS.index((15.0, 4))


def regimen_index(dose: float, weeks: int) -> int:
    return REGIMENS.index((dose, weeks))


@dataclass
class EnvConfig:
    allow_stop: bool = True
    stop_cost: float = 0.6  # per decision left, when the patient moves to another treatment
    drug_cost: float = 0.05  # per mg/kg/week of Drug X
    penalty_cap: float = 3.0  # in doublings above target
    measurement_error: bool = True
    pop: PopParams = field(default_factory=lambda: POP)


def penalty(b_true: np.ndarray, cap: float) -> np.ndarray:
    """0 at or below target, else the number of doublings above target (capped)."""
    return np.clip(np.log2(np.maximum(b_true, 1e-9) / TARGET), 0.0, cap)


class DosingEnv:
    def __init__(self, cohort: Cohort, cfg: EnvConfig | None = None):
        self.cohort = cohort
        self.cfg = cfg or EnvConfig()
        self.n = len(cohort)
        self.reset()

    # -- individual parameters (true, hidden from the agent) -----------------
    def _true_params(self):
        c, p = self.cohort, self.cfg.pop
        self.cl_ind = typical_cl(c.wt, c.alb, c.inf, 0.0, p) * np.exp(p.om_cl * c.z_cl)
        self.v1 = typical_v1(c.wt, p) * np.exp(p.om_v1 * c.z_v1)
        self.q = typical_q(c.wt, p)
        self.v2 = typical_v2(c.wt, p)
        self.base = typical_base(c.inf, p) * np.exp(p.om_base * c.z_base)
        self.ic50 = typical_ic50(c.plt, p) * np.exp(p.om_ic50 * c.z_ic50)

    def reset(self):
        n, c, p = self.n, self.cohort, self.cfg.pop
        self._true_params()
        self.a1 = np.zeros(n)
        self.a2 = np.zeros(n)
        self.b = self.base.copy()  # at steady state before treatment
        self.ada = c.ada0.astype(float).copy()
        self.day = np.zeros(n)
        self.active = np.ones(n, bool)
        self.decision = 0
        self.occ = 0
        self.stopped = np.zeros(n, bool)
        self.stop_decision = np.full(n, -1)
        # records
        self.dose = np.full((n, N_OCC), np.nan)
        self.tau = np.full((n, N_OCC), np.nan)
        self.cave = np.full((n, N_OCC), np.nan)
        self.base_occ = np.full((n, N_OCC), np.nan)
        self.trough_true = np.full((n, N_OCC), np.nan)
        self.visit_day = np.full((n, N_VISITS), np.nan)
        self.b_true = np.full((n, N_VISITS), np.nan)
        self.obs_conc = np.full((n, N_VISITS), np.nan)
        self.blq = np.ones((n, N_VISITS), bool)
        self.obs_bio = np.full((n, N_VISITS), np.nan)
        self.ada_visit = np.full((n, N_VISITS), np.nan)
        self.actions = np.full((n, N_DECISIONS), -1)
        self.rewards = np.zeros((n, N_DECISIONS))
        # visit 0: baseline biomarker, no drug yet
        self.visit_day[:, 0] = 0.0
        self.b_true[:, 0] = self.b
        self.obs_bio[:, 0] = self._measure_bio(self.b, 0)
        self.obs_conc[:, 0] = 0.0
        self.ada_visit[:, 0] = self.ada
        return self.observe(), self.mask()

    def _measure_bio(self, b, v):
        if not self.cfg.measurement_error:
            return b.copy()
        return np.maximum(b * (1.0 + self.cfg.pop.sigma_bio * self.cohort.e_bio[:, v]), 1.0)

    def _measure_conc(self, c, v):
        if not self.cfg.measurement_error:
            return c.copy()
        return np.maximum(c * (1.0 + self.cfg.pop.sigma_conc * self.cohort.e_conc[:, v]), 0.0)

    # -- simulation -----------------------------------------------------------
    def _run_occasion(self, dose_mgkg, tau, who):
        """Give dose_mgkg at the start of occasion self.occ and run tau days for patients in `who`."""
        k, p, c = self.occ, self.cfg.pop, self.cohort
        cl = self.cl_ind * np.exp(cv_to_sd(p.cv_iov_cl) * c.z_iov_cl[:, k]) * p.ada_cl**self.ada
        base_k = self.base * np.exp(cv_to_sd(p.cv_flare) * c.z_flare[:, k])
        a1, a2, cave, trough = pk_interval(self.a1, self.a2, dose_mgkg * c.wt, tau, cl, self.v1, self.q, self.v2)
        b_end = pd_interval(self.b, cave, tau, base_k, self.ic50, p.kout)
        v = k + 1
        w = who
        self.a1[w], self.a2[w], self.b[w] = a1[w], a2[w], b_end[w]
        self.day[w] += tau[w]
        self.dose[w, k], self.tau[w, k], self.cave[w, k] = dose_mgkg[w], tau[w], cave[w]
        self.base_occ[w, k], self.trough_true[w, k] = base_k[w], trough[w]
        # visit at the end of the interval: the biomarker recorded here is B(t_end)
        self.visit_day[w, v] = self.day[w]
        self.b_true[w, v] = b_end[w]
        conc = self._measure_conc(trough, v)
        self.obs_conc[w, v] = conc[w]
        self.blq[w, v] = conc[w] < LLOQ
        self.obs_bio[w, v] = self._measure_bio(b_end, v)[w]
        # ADA can switch on; more likely when the trough is low
        p_on = p.ada_p_max * np.exp(-trough / p.ada_c_ref)
        onset = w & (self.ada == 0) & (c.u_ada[:, k] < p_on)
        self.ada[onset] = 1.0
        self.ada_visit[w, v] = self.ada[w]
        self.occ += 1

    def step(self, actions):
        actions = np.asarray(actions, int)
        cfg = self.cfg
        who = self.active.copy()
        reward = np.zeros(self.n)
        self.actions[who, self.decision] = actions[who]
        if self.decision == 0:
            d = np.asarray(IND_DOSES)[actions]
            for tau in INDUCTION_TAUS:
                self._run_occasion(d, np.full(self.n, tau), who)
            rate = 3 * d / (sum(INDUCTION_TAUS) / 7.0)
            reward = -penalty(self.b, cfg.penalty_cap) - cfg.drug_cost * rate
        else:
            j = self.decision - 1
            stop = who & (actions == STOP)
            if stop.any() and not cfg.allow_stop:
                raise ValueError("stop chosen but not allowed")
            give = who & ~stop
            a = np.where(actions == STOP, 0, actions)
            d, tau = REG_DOSE[a], REG_TAU[a]
            self._run_occasion(d, tau, give)
            reward = -penalty(self.b, cfg.penalty_cap) - cfg.drug_cost * REG_RATE[a]
            reward = np.where(stop, -cfg.stop_cost * (N_MAINT_DECISIONS - j), reward)
            self.stopped |= stop
            self.stop_decision[stop] = self.decision
            self.active &= ~stop
        reward = np.where(who, reward, 0.0)
        self.rewards[who, self.decision] = reward[who]
        self.decision += 1
        if self.decision >= N_DECISIONS:
            self.active[:] = False
        return self.observe(), reward, ~self.active, self.mask()

    # -- what the agent sees --------------------------------------------------
    def current_visit(self) -> int:
        return 0 if self.decision == 0 else len(INDUCTION_TAUS) + self.decision - 1

    def mask(self) -> np.ndarray:
        """Allowed actions for the current decision (per patient, for the current head)."""
        if self.decision == 0:
            return np.ones((self.n, N_IND), bool)
        m = np.ones((self.n, N_MAINT), bool)
        # stopping needs at least one biomarker after induction, which every
        # maintenance decision has; it is also off when the setting disallows it
        seen_post_induction = self.current_visit() >= len(INDUCTION_TAUS)
        m[:, STOP] = self.cfg.allow_stop and seen_post_induction
        return m

    def observe(self) -> np.ndarray:
        """Observation vector built only from measurements, doses and covariates."""
        n, c = self.n, self.cohort
        v = min(self.current_visit(), N_VISITS - 1)
        vp = max(v - 1, 0)
        obs = np.zeros((n, OBS_DIM), np.float32)
        conc = np.where(self.blq[:, v], LLOQ / 2, self.obs_conc[:, v])
        k = v - 1  # last occasion given
        last_dose = self.dose[:, k] if k >= 0 else np.zeros(n)
        last_tau = self.tau[:, k] if k >= 0 else np.zeros(n)
        ind_dose = self.dose[:, 0] if self.occ > 0 else np.zeros(n)
        obs[:, 0] = float(self.decision > 0)
        obs[:, 1] = (N_DECISIONS - min(self.decision, N_DECISIONS)) / N_DECISIONS
        obs[:, 2] = np.log(self.obs_bio[:, 0] / TARGET)
        obs[:, 3] = np.log(self.obs_bio[:, vp] / TARGET)
        obs[:, 4] = np.log(self.obs_bio[:, v] / TARGET)
        obs[:, 5] = np.log(conc / 5.0)
        obs[:, 6] = self.blq[:, v]
        obs[:, 7] = np.nan_to_num(last_dose) / 15.0
        obs[:, 8] = np.nan_to_num(last_tau) / 70.0
        obs[:, 9] = np.nan_to_num(ind_dose) / 15.0
        obs[:, 10] = np.log(c.wt / 55.0)
        obs[:, 11] = (c.alb - 3.9) / 0.4
        obs[:, 12] = np.log(c.inf / 5.0)
        obs[:, 13] = np.log(c.plt / 300.0)
        obs[:, 14] = self.ada_visit[:, v]
        obs[:, 15] = np.nan_to_num(self.visit_day[:, v]) / 364.0
        return np.nan_to_num(obs)


def run_policy(policy, cohort: Cohort, cfg: EnvConfig | None = None) -> DosingEnv:
    """Run a policy on every patient in the cohort and return the finished env (with its records)."""
    env = DosingEnv(cohort, cfg)
    if hasattr(policy, "reset"):
        policy.reset(env)
    while env.active.any():
        a = np.asarray(policy.act(env), int)
        m = env.mask()
        bad = env.active & ~m[np.arange(env.n), a]
        if bad.any():
            raise ValueError(f"{getattr(policy, 'name', policy)} chose a masked action")
        env.step(a)
    return env
