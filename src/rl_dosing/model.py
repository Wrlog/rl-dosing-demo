"""The simulated Drug X model: two-compartment PK and an indirect-response biomarker.

Every number here is made up for the demo (see the shared model in the README).
All functions work on numpy arrays with one entry per patient, so a whole cohort
moves through one dosing interval in a single call.

Time is in days, doses in mg, volumes in L, concentrations in mg/L and the
biomarker in units.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

T_INF = 2.0 / 24.0  # 2 h infusion
LLOQ = 0.5  # mg/L
TARGET = 200.0  # biomarker units; "at target" means below this


def cv_to_sd(cv: float) -> float:
    """Log-scale SD for a log-normal with the given CV."""
    return float(np.sqrt(np.log1p(cv**2)))


@dataclass(frozen=True)
class PopParams:
    cl: float = 0.30  # L/day for 70 kg, albumin 4, INF 5, ADA negative
    v1: float = 3.2
    q: float = 0.50
    v2: float = 2.0
    ada_cl: float = 1.8  # CL multiplier when ADA positive
    base: float = 600.0  # units at INF 5
    ic50: float = 3.0  # mg/L at PLT 300
    kout: float = 0.04  # /day
    cv_cl: float = 0.30
    cv_v1: float = 0.20
    cv_iov_cl: float = 0.15
    cv_base: float = 0.60
    cv_ic50: float = 0.90
    cv_flare: float = 0.0  # between-occasion variability on BASE ("flares"); off by default
    sigma_conc: float = 0.15
    sigma_bio: float = 0.25
    # ADA onset (my own choice): chance per interval of turning positive,
    # higher when the true trough is low.
    ada_p_max: float = 0.12
    ada_c_ref: float = 2.0

    @property
    def om_cl(self) -> float:
        return cv_to_sd(self.cv_cl)

    @property
    def om_v1(self) -> float:
        return cv_to_sd(self.cv_v1)

    @property
    def om_base(self) -> float:
        return cv_to_sd(self.cv_base)

    @property
    def om_ic50(self) -> float:
        return cv_to_sd(self.cv_ic50)

    def with_(self, **kw) -> "PopParams":
        return replace(self, **kw)


POP = PopParams()


# ----------------------------------------------------------------------------
# Covariate profiles and virtual patients
# ----------------------------------------------------------------------------

@dataclass
class ProfilePool:
    """Simulated baseline covariate profiles (weight, albumin, INF, PLT, ADA)."""

    wt: np.ndarray
    alb: np.ndarray
    inf: np.ndarray
    plt: np.ndarray
    ada: np.ndarray

    def __len__(self) -> int:
        return len(self.wt)

    def subset(self, idx: np.ndarray) -> "ProfilePool":
        return ProfilePool(self.wt[idx], self.alb[idx], self.inf[idx], self.plt[idx], self.ada[idx])


def sample_profile_pool(n: int, rng: np.random.Generator) -> ProfilePool:
    """Draw covariate profiles. INF and albumin share a latent inflammation score."""
    z = rng.standard_normal(n)
    e = rng.standard_normal((4, n))
    inf = np.clip(5.0 * np.exp(0.9 * (0.8 * z + 0.6 * e[0])), 0.5, 60.0)
    alb = np.clip(3.9 - 0.4 * (0.7 * z + 0.714 * e[1]), 2.5, 5.0)
    wt = np.clip(55.0 * np.exp(0.35 * e[2]), 20.0, 100.0)
    plt = np.clip(300.0 * np.exp(0.30 * (0.4 * z + 0.917 * e[3])), 150.0, 600.0)
    ada = (rng.random(n) < 0.08).astype(float)
    return ProfilePool(wt, alb, inf, plt, ada)


def split_pool(pool: ProfilePool, held_out_frac: float, rng: np.random.Generator):
    """Fixed split of the profile pool into training and held-out profiles."""
    idx = rng.permutation(len(pool))
    n_out = int(round(held_out_frac * len(pool)))
    return pool.subset(np.sort(idx[n_out:])), pool.subset(np.sort(idx[:n_out]))


@dataclass
class Cohort:
    """Virtual patients: a covariate profile plus fresh random effects.

    Random effects and all later random draws (IOV, flares, ADA onset,
    measurement error) are stored as standard normals or uniforms, so the same
    patients can be run under different regimens with paired draws, and the
    variances can be rescaled (for the IC50 variability ladder) without
    redrawing.
    """

    wt: np.ndarray
    alb: np.ndarray
    inf: np.ndarray
    plt: np.ndarray
    ada0: np.ndarray
    z_cl: np.ndarray
    z_v1: np.ndarray
    z_base: np.ndarray
    z_ic50: np.ndarray
    z_iov_cl: np.ndarray  # (n, n_occ)
    z_flare: np.ndarray  # (n, n_occ)
    u_ada: np.ndarray  # (n, n_occ)
    e_conc: np.ndarray  # (n, n_visits)
    e_bio: np.ndarray  # (n, n_visits)

    def __len__(self) -> int:
        return len(self.wt)

    def subset(self, idx) -> "Cohort":
        return Cohort(**{k: v[idx] for k, v in self.__dict__.items()})


def sample_cohort(n: int, pool: ProfilePool, rng: np.random.Generator, n_occ: int,
                  cycle_profiles: bool = False) -> Cohort:
    """n virtual patients from a profile pool. Profiles are cycled or drawn at random."""
    if cycle_profiles:
        idx = np.arange(n) % len(pool)
    else:
        idx = rng.integers(len(pool), size=n)
    p = pool.subset(idx)
    n_vis = n_occ + 1
    return Cohort(
        wt=p.wt, alb=p.alb, inf=p.inf, plt=p.plt, ada0=p.ada,
        z_cl=rng.standard_normal(n), z_v1=rng.standard_normal(n),
        z_base=rng.standard_normal(n), z_ic50=rng.standard_normal(n),
        z_iov_cl=rng.standard_normal((n, n_occ)), z_flare=rng.standard_normal((n, n_occ)),
        u_ada=rng.random((n, n_occ)),
        e_conc=rng.standard_normal((n, n_vis)), e_bio=rng.standard_normal((n, n_vis)),
    )


def typical_cl(wt, alb, inf, ada, pop: PopParams = POP):
    return pop.cl * (wt / 70.0) ** 0.75 * (alb / 4.0) ** -1.0 * (inf / 5.0) ** 0.10 * pop.ada_cl**ada


def typical_v1(wt, pop: PopParams = POP):
    return pop.v1 * (wt / 70.0)


def typical_q(wt, pop: PopParams = POP):
    return pop.q * (wt / 70.0) ** 0.75


def typical_v2(wt, pop: PopParams = POP):
    return pop.v2 * (wt / 70.0)


def typical_base(inf, pop: PopParams = POP):
    return pop.base * (inf / 5.0) ** 0.30


def typical_ic50(plt, pop: PopParams = POP):
    return pop.ic50 * (plt / 300.0) ** 0.40


# ----------------------------------------------------------------------------
# PK: exact solution of the two-compartment model over one dosing interval
# ----------------------------------------------------------------------------

def _eig(cl, v1, q, v2):
    k10, k12, k21 = cl / v1, q / v1, q / v2
    s = k10 + k12 + k21
    disc = np.sqrt(s * s - 4.0 * k10 * k21)
    l1 = -(s - disc) / 2.0  # slow (terminal) eigenvalue
    l2 = -(s + disc) / 2.0
    return k10, k12, k21, l1, l2


def _apply(f1, f2, l1, l2, k, x1, x2):
    """F(K) @ x for a 2x2 matrix K with eigenvalues l1, l2 (Sylvester's formula).

    F(K) = [f(l1) (K - l2 I) - f(l2) (K - l1 I)] / (l1 - l2)
    """
    (k11, k12_, k21_, k22) = k
    kx1 = k11 * x1 + k12_ * x2
    kx2 = k21_ * x1 + k22 * x2
    d = l1 - l2
    y1 = (f1 * (kx1 - l2 * x1) - f2 * (kx1 - l1 * x1)) / d
    y2 = (f1 * (kx2 - l2 * x2) - f2 * (kx2 - l1 * x2)) / d
    return y1, y2


def pk_interval(a1, a2, dose_mg, tau, cl, v1, q, v2):
    """Advance the PK over one interval that starts with a 2 h infusion.

    a1, a2 are drug amounts (mg) in the central and peripheral compartment at
    the start. Returns (a1_end, a2_end, cave, trough) where cave is the exact
    average central concentration over the interval and trough the
    concentration at its end (just before the next dose).
    """
    k10, k12, k21, l1, l2 = _eig(cl, v1, q, v2)
    k = (-(k10 + k12), k21, k12, -k21)
    rate = dose_mg / T_INF
    zero = np.zeros_like(a1 * 1.0)

    def e(lam, t):
        return np.exp(lam * t)

    def g(lam, t):  # integral of exp(lam s) ds over [0, t]
        return np.expm1(lam * t) / lam

    def h(lam, t):  # double integral, for the infusion input
        return (g(lam, t) - t) / lam

    t1 = T_INF
    # end of infusion
    x1, x2 = _apply(e(l1, t1), e(l2, t1), l1, l2, k, a1, a2)
    u1, u2 = _apply(g(l1, t1), g(l2, t1), l1, l2, k, rate + zero, zero)
    x1, x2 = x1 + u1, x2 + u2
    auc1_a, _ = _apply(g(l1, t1), g(l2, t1), l1, l2, k, a1, a2)
    auc1_b, _ = _apply(h(l1, t1), h(l2, t1), l1, l2, k, rate + zero, zero)
    # rest of the interval
    t2 = tau - T_INF
    y1, y2 = _apply(e(l1, t2), e(l2, t2), l1, l2, k, x1, x2)
    auc2, _ = _apply(g(l1, t2), g(l2, t2), l1, l2, k, x1, x2)
    cave = (auc1_a + auc1_b + auc2) / v1 / tau
    return y1, y2, cave, y1 / v1


def pd_interval(b0, cave, tau, base, ic50, kout):
    """Exact indirect-response update with Cave held constant over the interval.

    dB/dt = kin (1 - Cave/(IC50 + Cave)) - kout B, kin = BASE kout.
    """
    bss = base * ic50 / (ic50 + cave)
    return bss + (b0 - bss) * np.exp(-kout * tau)


def pd_curve(b0, cave, t, base, ic50, kout):
    """Biomarker at times t (days since the start of the interval)."""
    bss = base * ic50 / (ic50 + cave)
    return bss + (b0 - bss) * np.exp(-kout * t)
