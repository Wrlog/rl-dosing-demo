"""Comparator policies: fixed regimens, a myopic MAP-Bayesian dosing rule and an oracle.

Every policy has `act(env) -> actions`, one action per patient for the
current decision. Only the oracle reads the patient's true parameters.
"""

from __future__ import annotations

import numpy as np

from .env import (IND_DOSES, INDUCTION_TAUS, N_MAINT_DECISIONS, REG_BY_COST, REG_DOSE, REG_TAU, STOP,
                  STRONGEST, DosingEnv, regimen_index)
from .model import TARGET, PopParams, pd_interval, pk_interval
from .model import typical_base, typical_cl, typical_ic50, typical_q, typical_v1, typical_v2

IND_ORDER = np.argsort(IND_DOSES)


class FixedRegimen:
    """The same induction dose and maintenance regimen for everyone; never stops."""

    def __init__(self, ind_dose: float, maint_dose: float, weeks: int, name: str | None = None):
        self.ind = IND_DOSES.index(ind_dose)
        self.maint = regimen_index(maint_dose, weeks)
        self.name = name or f"{ind_dose:g} mg/kg q{weeks}w"

    def act(self, env: DosingEnv):
        return np.full(env.n, self.ind if env.decision == 0 else self.maint)


def standard_regimen():
    return FixedRegimen(5.0, 5.0, 8, "Standard")


def ceiling_regimen():
    return FixedRegimen(max(IND_DOSES), 15.0, 4, "Ceiling")


# ----------------------------------------------------------------------------
# A small batched optimizer: damped Newton steps with finite-difference
# derivatives, one independent problem per patient.
# ----------------------------------------------------------------------------

def batched_minimize(fun, x0: np.ndarray, n_iter: int = 30, h: float = 1e-4, max_step: float = 1.0):
    x = x0.copy()
    n, d = x.shape
    fx = fun(x)
    lam = np.full(n, 1e-2)
    eye = np.eye(d)
    for _ in range(n_iter):
        g = np.zeros((n, d))
        hess = np.zeros((n, d, d))
        fp, fm = [], []
        for i in range(d):
            e = eye[i] * h
            a, b = fun(x + e), fun(x - e)
            g[:, i] = (a - b) / (2 * h)
            hess[:, i, i] = (a - 2 * fx + b) / h**2
            fp.append(a)
            fm.append(b)
        for i in range(d):
            for j in range(i + 1, d):
                e = (eye[i] + eye[j]) * h
                hij = (fun(x + e) - fp[i] - fp[j] + 2 * fx - fm[i] - fm[j] + fun(x - e)) / (2 * h**2)
                hess[:, i, j] = hess[:, j, i] = hij
        w, v = np.linalg.eigh(hess)
        w = np.maximum(np.abs(w), 1e-3) + lam[:, None]
        step = -np.einsum("nij,nj,nkj,nk->ni", v, 1 / w, v, g)
        norm = np.linalg.norm(step, axis=1, keepdims=True)
        step *= np.minimum(1.0, max_step / np.maximum(norm, 1e-12))
        xn = x + step
        fn = fun(xn)
        ok = np.isfinite(fn) & (fn < fx)
        x[ok], fx[ok] = xn[ok], fn[ok]
        lam = np.clip(np.where(ok, lam * 0.3, lam * 10), 1e-6, 1e6)
    return x, fx


def _prop_nll(y, f, sigma, use):
    f = np.maximum(f, 1e-6)
    sd = sigma * f
    return np.sum(np.where(use, ((y - f) / sd) ** 2 + 2 * np.log(sd), 0.0), axis=1)


class MAPBayes:
    """Myopic MAP-Bayesian dosing.

    At each decision: estimate CL and V1 from the troughs (BLQ values dropped),
    then BASE and IC50 from the biomarkers with the PK estimates giving Cave per
    interval (sequential PK then PD). With those estimates, predict the
    biomarker at the next visit under every regimen and take the cheapest one
    predicted to land below `margin * 200`. If none does, give the strongest
    regimen. If even the strongest regimen, held to the last decision, is
    predicted to finish above target, stop (when stopping is allowed).

    The estimation model ignores IOV on CL and flares, and assumes ADA status
    stays as it is now.
    """

    name = "MAP-Bayes"

    def __init__(self, pop: PopParams, margin: float = 0.8, n_iter: int = 25):
        self.pop, self.margin, self.n_iter = pop, margin, n_iter

    # typical values for a subset of patients
    def _typ(self, env, idx):
        c, p = env.cohort, self.pop
        return dict(cl=typical_cl(c.wt[idx], c.alb[idx], c.inf[idx], 0.0, p), v1=typical_v1(c.wt[idx], p),
                    q=typical_q(c.wt[idx], p), v2=typical_v2(c.wt[idx], p),
                    base=typical_base(c.inf[idx], p), ic50=typical_ic50(c.plt[idx], p), wt=c.wt[idx])

    def _pk_path(self, t, eta, doses, taus, ada):
        """Predicted troughs and Cave over past occasions, and the final amounts."""
        p = self.pop
        cl0 = t["cl"] * np.exp(eta[:, 0])
        v1 = t["v1"] * np.exp(eta[:, 1])
        a1 = np.zeros(len(cl0))
        a2 = np.zeros(len(cl0))
        k = doses.shape[1]
        troughs, caves = np.zeros((len(cl0), k)), np.zeros((len(cl0), k))
        for j in range(k):
            cl = cl0 * p.ada_cl ** ada[:, j]
            a1, a2, caves[:, j], troughs[:, j] = pk_interval(a1, a2, doses[:, j] * t["wt"], taus[:, j],
                                                             cl, v1, t["q"], t["v2"])
        return troughs, caves, a1, a2

    def _pd_path(self, t, eta, caves, taus):
        base = t["base"] * np.exp(eta[:, 0])
        ic50 = t["ic50"] * np.exp(eta[:, 1])
        b = base.copy()
        out = [b]
        for j in range(caves.shape[1]):
            b = pd_interval(b, caves[:, j], taus[:, j], base, ic50, self.pop.kout)
            out.append(b)
        return np.stack(out, axis=1), base, ic50

    def estimate(self, env: DosingEnv, idx: np.ndarray):
        """MAP estimates (eta_CL, eta_V1, eta_BASE, eta_IC50) for patients idx, from observed data only."""
        p, k = self.pop, env.occ
        t = self._typ(env, idx)
        doses, taus = env.dose[idx, :k], env.tau[idx, :k]
        ada = env.ada_visit[idx, :k]  # status during each occasion
        eta_pk = np.zeros((len(idx), 2))
        if k > 0:
            y = env.obs_conc[idx, 1:k + 1]
            use = ~env.blq[idx, 1:k + 1]
            om = np.array([p.om_cl, p.om_v1])

            def f_pk(th):
                tr, _, _, _ = self._pk_path(t, th, doses, taus, ada)
                return _prop_nll(y, tr, p.sigma_conc, use) + np.sum((th / om) ** 2, axis=1)

            eta_pk, _ = batched_minimize(f_pk, eta_pk, self.n_iter)
        _, caves, a1, a2 = self._pk_path(t, eta_pk, doses, taus, ada)
        yb = env.obs_bio[idx, :k + 1]
        omb = np.array([p.om_base, p.om_ic50])

        def f_pd(th):
            bp, _, _ = self._pd_path(t, th, caves, taus)
            return _prop_nll(yb, bp, p.sigma_bio, np.ones_like(yb, bool)) + np.sum((th / omb) ** 2, axis=1)

        eta_pd, _ = batched_minimize(f_pd, np.zeros((len(idx), 2)), self.n_iter)
        bpath, base, ic50 = self._pd_path(t, eta_pd, caves, taus)
        cl = t["cl"] * np.exp(eta_pk[:, 0]) * p.ada_cl ** env.ada_visit[idx, k]
        state = dict(a1=a1, a2=a2, b=bpath[:, -1], cl=cl, v1=t["v1"] * np.exp(eta_pk[:, 1]), q=t["q"],
                     v2=t["v2"], base=base, ic50=ic50, wt=t["wt"])
        return np.hstack([eta_pk, eta_pd]), state

    def act(self, env: DosingEnv):
        idx = np.where(env.active)[0]
        out = np.zeros(env.n, int)
        if len(idx) == 0:
            return out
        _, s = self.estimate(env, idx)
        out[idx] = choose_action(env, s, self.margin, self.pop.kout)
        return out


def _predict_induction(s, kout):
    """Biomarker at day 98 for every induction dose, from state s (all patients start untreated)."""
    doses = np.asarray(IND_DOSES)[None, :]
    a1 = np.zeros((len(s["cl"]), 1))
    a2 = np.zeros_like(a1)
    b = s["b"][:, None]
    col = {k: s[k][:, None] for k in ("cl", "v1", "q", "v2", "base", "ic50", "wt")}
    for tau in INDUCTION_TAUS:
        a1, a2, cave, _ = pk_interval(a1, a2, doses * col["wt"], tau, col["cl"], col["v1"], col["q"], col["v2"])
        b = pd_interval(b, cave, tau, col["base"], col["ic50"], kout)
    return b


def _predict_next(s, kout):
    """Biomarker at the next visit for every maintenance regimen (n, n_regimens)."""
    col = {k: s[k][:, None] for k in s}
    a1, a2, cave, _ = pk_interval(col["a1"], col["a2"], REG_DOSE[None, :] * col["wt"], REG_TAU[None, :],
                                  col["cl"], col["v1"], col["q"], col["v2"])
    return pd_interval(col["b"], cave, REG_TAU[None, :], col["base"], col["ic50"], kout)


def _predict_hold_strongest(s, n_left, kout):
    a1, a2, b = s["a1"].copy(), s["a2"].copy(), s["b"].copy()
    d, tau = REG_DOSE[STRONGEST], REG_TAU[STRONGEST]
    for _ in range(n_left):
        a1, a2, cave, _ = pk_interval(a1, a2, d * s["wt"], tau, s["cl"], s["v1"], s["q"], s["v2"])
        b = pd_interval(b, cave, tau, s["base"], s["ic50"], kout)
    return b


def choose_action(env: DosingEnv, s: dict, margin: float, kout: float) -> np.ndarray:
    """Cheapest action predicted to reach margin * target at the next visit (shared by MAP and oracle)."""
    thr = margin * TARGET
    if env.decision == 0:
        pred = _predict_induction(s, kout)
        ok = pred < thr
        first = np.where(ok.any(1), ok.argmax(1), len(IND_DOSES) - 1)  # doses are in increasing order
        return IND_ORDER[first]
    pred = _predict_next(s, kout)[:, REG_BY_COST]
    ok = pred < thr
    act = np.where(ok.any(1), REG_BY_COST[ok.argmax(1)], STRONGEST)
    if env.cfg.allow_stop:
        n_left = N_MAINT_DECISIONS - (env.decision - 1)
        hopeless = _predict_hold_strongest(s, n_left, kout) >= TARGET
        act = np.where(hopeless, STOP, act)
    return act


class Oracle:
    """The same rule as MAPBayes, but with the patient's true parameters and current state.

    It still doesn't know the future IOV, flares or ADA onset draws.
    """

    name = "Oracle"

    def __init__(self, margin: float = 1.0):
        self.margin = margin

    def act(self, env: DosingEnv):
        idx = np.where(env.active)[0]
        out = np.zeros(env.n, int)
        if len(idx) == 0:
            return out
        p = env.cfg.pop
        cl = env.cl_ind[idx] * p.ada_cl ** env.ada[idx]
        s = dict(a1=env.a1[idx], a2=env.a2[idx], b=env.b[idx], cl=cl, v1=env.v1[idx], q=env.q[idx],
                 v2=env.v2[idx], base=env.base[idx], ic50=env.ic50[idx], wt=env.cohort.wt[idx])
        out[idx] = choose_action(env, s, self.margin, p.kout)
        return out

