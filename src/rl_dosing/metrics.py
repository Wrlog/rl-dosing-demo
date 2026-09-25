"""Per-patient endpoints and bootstrap summaries."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .env import DosingEnv, N_VISITS
from .model import TARGET


def patient_table(env: DosingEnv, reachable: np.ndarray) -> pd.DataFrame:
    """One row per patient with every endpoint used in the README."""
    stopped = env.stopped
    final_b = np.where(stopped, np.nan, env.b_true[:, -1])
    at = ~stopped & (env.b_true[:, -1] < TARGET)
    # a correct decision: the patient is at target, or is unreachable and was moved to another treatment
    correct = at | (~reachable & stopped)
    last_day = np.nanmax(env.visit_day, axis=1)
    weeks = last_day / 7.0
    total = np.nansum(env.dose, axis=1)
    # time to sustained target: the first visit from which the true biomarker stays
    # below target through the final visit (never, for stopped patients)
    below = np.nan_to_num(env.b_true[:, 1:], nan=np.inf) < TARGET
    stays = np.flip(np.logical_and.accumulate(np.flip(below, 1), axis=1), 1)
    ever = stays.any(1) & ~stopped
    first = stays.argmax(1) + 1
    ttt = np.where(ever, env.visit_day[np.arange(env.n), first] / 7.0, np.nan)
    stop_visit = np.where(stopped, 3 + env.stop_decision - 1, N_VISITS - 1)
    b_last_seen = env.b_true[np.arange(env.n), stop_visit]
    return pd.DataFrame({
        "patient": np.arange(env.n),
        "reachable": reachable,
        "stopped": stopped,
        "stop_week": np.where(stopped, last_day / 7.0, np.nan),
        "at_target": at,
        "correct": correct,
        "final_b": final_b,
        "b_last_seen": b_last_seen,
        "obs_b_last_seen": env.obs_bio[np.arange(env.n), stop_visit],
        "weeks_on_drug": weeks,
        "total_mgkg": total,
        "induction_mgkg": env.dose[:, 0],
        "dose_rate": total / weeks,
        "cave_last": np.where(stopped, np.nan, env.cave[:, -1]),
        "weeks_to_target": ttt,
        "sustained_target": ever,
        "ret": env.rewards.sum(1),
        "ada_end": env.ada,
    })


STAT_NAMES = ["pct_at_target", "pct_correct", "pct_stopped", "dose_rate", "cave_gcv", "weeks_to_target",
              "mean_return"]


def _stats(d: pd.DataFrame, idx: np.ndarray) -> dict:
    """Statistics for each row of idx (replicates x patients), vectorized over replicates."""
    def col(name):
        return d[name].to_numpy(float)[idx]

    lc = np.log(col("cave_last"))
    with np.errstate(all="ignore"):
        gcv = 100 * np.sqrt(np.expm1(np.nanvar(lc, axis=1, ddof=1)))
        return {
            "pct_at_target": 100 * col("at_target").mean(1),
            "pct_correct": 100 * col("correct").mean(1),
            "pct_stopped": 100 * col("stopped").mean(1),
            "dose_rate": col("dose_rate").mean(1),
            "cave_gcv": gcv,
            "weeks_to_target": np.nanmean(col("weeks_to_target"), axis=1),
            "mean_return": col("ret").mean(1),
        }


def _seed_mean(tables, idx, names):
    per = [_stats(t, idx) for t in tables]
    return {k: np.mean([p[k] for p in per], axis=0) for k in names}


def summarize(tables: list[pd.DataFrame], n_boot: int, rng: np.random.Generator) -> dict:
    """Point estimates and 95% percentile bootstrap CIs over patients.

    `tables` holds one per-patient table per training seed (a single table for
    deterministic policies). Each statistic is averaged over seeds, and each
    bootstrap replicate resamples the same patients in every seed.
    """
    n = len(tables[0])
    point = _seed_mean(tables, np.arange(n)[None, :], STAT_NAMES)
    out = {k: float(point[k][0]) for k in STAT_NAMES}
    if n_boot:
        boot = _seed_mean(tables, rng.integers(n, size=(n_boot, n)), STAT_NAMES)
        for k in STAT_NAMES:
            out[k + "_lo"], out[k + "_hi"] = np.nanpercentile(boot[k], [2.5, 97.5])
    return out


def paired_difference(a: list[pd.DataFrame], b: list[pd.DataFrame], n_boot: int, rng: np.random.Generator,
                      stats=("pct_at_target", "pct_correct", "dose_rate", "mean_return")) -> dict:
    """a minus b on the same patients, with a paired bootstrap CI."""
    n = len(a[0])
    full = np.arange(n)[None, :]
    pa, pb = _seed_mean(a, full, stats), _seed_mean(b, full, stats)
    idx = rng.integers(n, size=(n_boot, n))
    ba, bb = _seed_mean(a, idx, stats), _seed_mean(b, idx, stats)
    out = {}
    for k in stats:
        out[k] = float(pa[k][0] - pb[k][0])
        out[k + "_lo"], out[k + "_hi"] = np.percentile(ba[k] - bb[k], [2.5, 97.5])
    return out
