import numpy as np

from rl_dosing.model import (POP, T_INF, pd_interval, pk_interval, sample_cohort, sample_profile_pool, split_pool,
                             typical_cl)


def rk4_pk(a1, a2, dose, tau, cl, v1, q, v2, dt=T_INF / 50):
    """Brute-force reference: RK4 on a fine grid (aligned with the end of the infusion), trapezoid AUC."""
    k10, k12, k21 = cl / v1, q / v1, q / v2
    n = int(round(tau / dt))
    rate = dose / T_INF

    n_inf = int(round(T_INF / dt))

    def f(r, x):
        return np.array([-(k10 + k12) * x[0] + k21 * x[1] + r, k12 * x[0] - k21 * x[1]])

    x, auc, h = np.array([a1, a2], float), 0.0, dt
    for i in range(n):
        r = rate if i < n_inf else 0.0
        k1 = f(r, x)
        k2 = f(r, x + h / 2 * k1)
        k3 = f(r, x + h / 2 * k2)
        k4 = f(r, x + h * k3)
        xn = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        auc += h * (x[0] + xn[0]) / 2
        x = xn
    return x[0], x[1], auc / v1 / tau, x[0] / v1


def test_pk_matches_fine_grid_ode():
    cl, v1, q, v2 = 0.25, 2.6, 0.42, 1.6
    a1, a2 = 0.0, 0.0
    for dose, tau in ((300.0, 21.0), (300.0, 21.0), (450.0, 42.0)):
        ref = rk4_pk(a1, a2, dose, tau, cl, v1, q, v2)
        got = pk_interval(np.array([a1]), np.array([a2]), np.array([dose]), np.array([tau]),
                          np.array([cl]), np.array([v1]), np.array([q]), np.array([v2]))
        for r, g in zip(ref, got):
            assert abs(g[0] - r) / abs(r) < 1e-4
        a1, a2 = got[0][0], got[1][0]


def test_steady_state_cave_is_dose_over_cl_tau():
    cl, v1, q, v2, dose, tau = 0.3, 3.0, 0.5, 2.0, 400.0, 28.0
    a1 = a2 = np.zeros(1)
    for _ in range(80):
        a1, a2, cave, _ = pk_interval(a1, a2, np.array([dose]), np.array([tau]), cl, v1, q, v2)
    assert abs(cave[0] - dose / (cl * tau)) / cave[0] < 1e-6


def test_pd_matches_ode():
    b0, cave, tau, base, ic50, kout = 600.0, 12.0, 56.0, 600.0, 10.0, 0.04
    b, dt = b0, 1e-3
    for _ in range(int(tau / dt)):
        b += dt * (base * kout * (1 - cave / (ic50 + cave)) - kout * b)
    assert abs(pd_interval(b0, cave, tau, base, ic50, kout) - b) < 0.05


def test_covariates_in_range_and_correlated():
    pool = sample_profile_pool(5000, np.random.default_rng(0))
    assert pool.wt.min() >= 20 and pool.wt.max() <= 100
    assert pool.alb.min() >= 2.5 and pool.alb.max() <= 5.0
    assert pool.inf.min() >= 0.5 and pool.inf.max() <= 60
    assert pool.plt.min() >= 150 and pool.plt.max() <= 600
    assert np.corrcoef(np.log(pool.inf), pool.alb)[0, 1] < -0.3  # higher INF, lower albumin


def test_profile_split_is_disjoint_and_complete():
    pool = sample_profile_pool(400, np.random.default_rng(1))
    tr, ho = split_pool(pool, 0.25, np.random.default_rng(2))
    assert len(tr) + len(ho) == 400 and len(ho) == 100
    key = lambda p: set(zip(p.wt.round(9), p.alb.round(9), p.inf.round(9), p.plt.round(9)))
    assert not key(tr) & key(ho)


def test_cohort_is_reproducible():
    pool = sample_profile_pool(50, np.random.default_rng(3))
    a = sample_cohort(20, pool, np.random.default_rng(4), 7)
    b = sample_cohort(20, pool, np.random.default_rng(4), 7)
    assert np.array_equal(a.z_ic50, b.z_ic50) and np.array_equal(a.e_bio, b.e_bio)


def test_typical_cl_covariate_effects():
    base = typical_cl(70.0, 4.0, 5.0, 0.0, POP)
    assert np.isclose(base, POP.cl)
    assert np.isclose(typical_cl(70.0, 4.0, 5.0, 1.0, POP), 1.8 * POP.cl)
    assert typical_cl(70.0, 3.0, 5.0, 0.0, POP) > base  # low albumin, faster clearance
