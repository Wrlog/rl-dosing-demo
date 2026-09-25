"""Draw the figures in figures/ from the CSV files in results/.

    python scripts/make_figures.py [--quick]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
COLORS = {"Standard": "#2a78d6", "Ceiling": "#eb6834", "MAP-Bayes": "#1baf7a", "Oracle": "#eda100",
          "DQN": "#e87ba4", "DQN trained with flares": "#4a3aa7"}
ORDER = ["Standard", "Ceiling", "MAP-Bayes", "Oracle", "DQN"]
TARGET = 200.0

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11,
    "axes.titleweight": "normal", "legend.frameon": False,
})


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def learning_curve(res, figs):
    curves = pd.concat([pd.read_csv(res / f"learning_curve_dqn_seed{s}.csv") for s in (0, 1, 2)])
    refs = pd.read_csv(res / "eval_references.csv").set_index("policy")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharex=True)
    for ax, col, title in ((axes[0], "pct_at_target", "At target at the final visit"),
                           (axes[1], "pct_correct", "Correct decisions")):
        for i, (run, d) in enumerate(curves.groupby("run")):
            ax.plot(d.episode, d[col], color=COLORS["DQN"], lw=1.6, alpha=[1, 0.7, 0.45][i],
                    label="DQN (3 seeds)" if i == 0 else None)
        for name, key, style in (("Ceiling (reachable)", "Ceiling", "-"), ("MAP-Bayes", "MAP-Bayes", "--"),
                                 ("Oracle", "Oracle", ":"), ("Standard", "Standard", "--")):
            y = refs.loc[name, col]
            ax.axhline(y, color=COLORS[key], lw=1.6, ls=style, label=name)
        ax.set_title(title)
        ax.set_xlabel("Training episodes (patients)")
        ax.set_ylabel("% of evaluation cohort")
        ax.set_ylim(30, 100)
    axes[1].legend(loc="center right", fontsize=8.5)
    save(fig, figs / "learning_curve.png")


def policy_bars(res, figs):
    s = pd.read_csv(res / "summary_main.csv").set_index("policy")
    seeds = s.loc[[f"DQN seed {k}" for k in (0, 1, 2)]]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw={"width_ratios": [2, 1]})
    metrics = [("pct_at_target", "At target"), ("pct_correct", "Correct decisions")]
    w = 0.15
    ax = axes[0]
    for j, pol in enumerate(ORDER):
        x = np.arange(len(metrics)) + (j - 2) * (w + 0.02)
        y = [s.loc[pol, m] for m, _ in metrics]
        lo = [s.loc[pol, m] - s.loc[pol, m + "_lo"] for m, _ in metrics]
        hi = [s.loc[pol, m + "_hi"] - s.loc[pol, m] for m, _ in metrics]
        ax.bar(x, y, w, color=COLORS[pol], label=pol, yerr=[lo, hi], ecolor=INK2, capsize=2, error_kw={"lw": 1})
        if pol == "DQN":
            for i, (m, _) in enumerate(metrics):
                ax.scatter([x[i]] * 3, seeds[m], s=12, color=INK, zorder=3, label="DQN seeds" if i == 0 else None)
    ax.set_xticks(np.arange(len(metrics)), [t for _, t in metrics])
    ax.set_ylabel("% of held-out patients")
    ax.set_ylim(0, 105)
    ax.legend(ncol=3, fontsize=8.5, loc="upper left", bbox_to_anchor=(0, 1.18))
    ax = axes[1]
    for j, pol in enumerate(ORDER):
        y = s.loc[pol, "dose_rate"]
        ax.bar(j, y, 0.7, color=COLORS[pol], yerr=[[y - s.loc[pol, "dose_rate_lo"]], [s.loc[pol, "dose_rate_hi"] - y]],
               ecolor=INK2, capsize=2)
    ax.scatter([4] * 3, seeds["dose_rate"], s=12, color=INK, zorder=3)
    ax.set_xticks(range(len(ORDER)), ["Std", "Ceil", "MAP", "Oracle", "DQN"])
    ax.set_ylabel("Dose rate (mg/kg per week)")
    save(fig, figs / "policy_bars.png")


def ceiling_vs_variability(res, figs):
    lad = pd.read_csv(res / "ic50_ladder.csv")
    styles = {"IC50 10 mg/L": ("#2a78d6", "o"), "IC50 10 mg/L, flares": ("#eb6834", "s"),
              "IC50 3 mg/L (shared model)": ("#1baf7a", "^")}
    fig, ax = plt.subplots(figsize=(6.4, 4))
    for setting, (c, mk) in styles.items():
        for reg, ls in (("Ceiling", "-"), ("Standard", "--")):
            d = lad[(lad.setting == setting) & (lad.regimen == reg)].sort_values("ic50_cv")
            ax.fill_between(100 * d.ic50_cv, d.lo, d.hi, color=c, alpha=0.12, lw=0)
            ax.plot(100 * d.ic50_cv, d.pct_at_target, ls=ls, marker=mk, ms=5, color=c, lw=1.8,
                    label=f"{reg}, {setting}")
    ax.invert_xaxis()
    ax.set_xlabel("IC50 between-patient variability (CV %), shrinking to the right")
    ax.set_ylabel("% at target at the final visit")
    ax.legend(fontsize=8, loc="lower left")
    save(fig, figs / "ceiling_vs_ic50_variability.png")


def frontier(res, figs):
    f = pd.read_csv(res / "frontier.csv")
    s = pd.read_csv(res / "summary_main.csv").set_index("policy")
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    fx = f[f.family == "Fixed regimen"]
    ax.scatter(fx.dose_rate, fx.pct_correct, s=26, color="#a3a29c", label="Fixed regimens (dose x interval)")
    for _, r in fx.iterrows():
        offsets = {"5 q8w": (-14, -14), "15 q4w": (4, -10), "10 q8w": (4, -10), "5 q4w": (4, -10)}
        if r.label in offsets:
            ax.annotate(r.label, (r.dose_rate, r.pct_correct), xytext=offsets[r.label], textcoords="offset points",
                        fontsize=8, color=INK2)
    mp = f[f.family == "MAP-Bayes"].sort_values("dose_rate")
    ax.plot(mp.dose_rate, mp.pct_correct, "-o", color=COLORS["MAP-Bayes"], ms=5, lw=1.8,
            label="MAP-Bayes, margins 1.0 to 0.5")
    dq = f[f.family == "DQN"].sort_values("dose_rate")
    ax.plot(dq.dose_rate, dq.pct_correct, "-D", color=COLORS["DQN"], ms=5, lw=1.8,
            label="DQN, drug cost 0 to 0.4")
    for _, r in dq.iterrows():
        ax.annotate(f"{r.drug_cost:g}", (r.dose_rate, r.pct_correct), xytext=(5, 3), textcoords="offset points",
                    fontsize=8, color=INK2)
    ax.scatter([s.loc["Oracle", "dose_rate"]], [s.loc["Oracle", "pct_correct"]], marker="*", s=120,
               color=COLORS["Oracle"], zorder=3, label="Oracle")
    ax.set_xlabel("Dose rate (mg/kg per week)")
    ax.set_ylabel("% correct decisions")
    ax.legend(fontsize=8, loc="lower right")
    save(fig, figs / "frontier.png")


def examples(res, figs):
    v = pd.read_csv(res / "example_visits.csv")
    c = pd.read_csv(res / "example_curves.csv")
    titles = ["Reachable, missed by the standard regimen", "Unreachable (ceiling misses target)",
              "At target on the standard regimen"]
    pols = [("Standard", "Standard"), ("MAP-Bayes", "MAP-Bayes"), ("DQN seed 0", "DQN")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for case, ax in enumerate(axes):
        for pol, key in pols:
            cc = c[(c.case == case) & (c.policy == pol)]
            vv = v[(v.case == case) & (v.policy == pol)]
            ax.plot(cc.day / 7, cc.b_true, color=COLORS[key], lw=1.8, label=key)
            ax.scatter(vv.day / 7, vv.b_obs, color=COLORS[key], s=14, zorder=3)
            st = vv[vv.stopped_here]
            if len(st):
                ax.scatter(st.day / 7, st.b_obs, marker="X", s=90, color=COLORS[key], edgecolor=INK, zorder=4)
        ax.axhline(TARGET, color=INK2, lw=1, ls="--")
        ax.set_yscale("log")
        ax.set_ylim(12, 3000)
        ax.set_title(titles[case], fontsize=10)
        ax.set_xlabel("Weeks")
        # regimen text per policy
        lines = []
        for pol, key in pols:
            vv = v[(v.case == case) & (v.policy == pol)].dropna(subset=["dose_mgkg"])
            maint = " ".join(f"{d:g}/{t / 7:g}w" for d, t in zip(vv.dose_mgkg[3:], vv.tau_days[3:]))
            stop = " then stop" if v[(v.case == case) & (v.policy == pol)].stopped_here.any() else ""
            lines.append(f"{key}: induction {vv.dose_mgkg.iloc[0]:g}; {maint}{stop}")
        ax.text(0.02, 0.02, "\n".join(lines), transform=ax.transAxes, fontsize=6.8, color=INK2, va="bottom")
    axes[0].set_ylabel("Biomarker, units (line true, dots measured)")
    axes[0].legend(fontsize=8, loc="upper right")
    save(fig, figs / "example_trajectories.png")


def misses(res, figs):
    m = pd.read_csv(res / "misses.csv").set_index("policy").loc[["MAP-Bayes", "Oracle", "DQN"]]
    kinds = [k for k in m.columns if k.startswith(("Reachable", "Unreachable"))]
    pal = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
    fig, ax = plt.subplots(figsize=(8.5, 3.2))
    left = np.zeros(len(m))
    n = m["n_patients"].to_numpy()
    for k, col in zip(kinds, pal):
        val = 100 * m[k].to_numpy() / n
        ax.barh(range(len(m)), val, left=left, color=col, label=k, edgecolor=SURFACE, lw=1.5)
        left += val
    ax.set_yticks(range(len(m)), m.index)
    ax.invert_yaxis()
    ax.set_xlabel("% of held-out patients with an incorrect decision")
    ax.legend(fontsize=7.5, loc="center left", bbox_to_anchor=(1.01, 0.5))
    save(fig, figs / "misses.png")


def scenarios(res, figs):
    rows = []
    for scen, label in (("main", "Held-out profiles"), ("seen", "Training profiles"), ("flares", "Flares on"),
                        ("shared_ic50", "IC50 3 mg/L")):
        s = pd.read_csv(res / f"summary_{scen}.csv")
        rows.append(s.assign(scenario=label))
    d = pd.concat(rows)
    pols = ORDER + ["DQN trained with flares"]
    labels = d.scenario.unique()
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    w = 0.13
    for j, pol in enumerate(pols):
        dd = d[d.policy == pol].set_index("scenario").reindex(labels)
        x = np.arange(len(labels)) + (j - 2.5) * (w + 0.01)
        y = dd.pct_correct.to_numpy()
        err = [y - dd.pct_correct_lo.to_numpy(), dd.pct_correct_hi.to_numpy() - y]
        ax.bar(x, y, w, color=COLORS[pol], label=pol, yerr=err, ecolor=INK2, capsize=1.5, error_kw={"lw": 0.8})
    ax.set_xticks(range(len(labels)), labels)
    ax.set_ylabel("% correct decisions")
    ax.set_ylim(40, 102)
    ax.legend(ncol=3, fontsize=8, loc="upper left", bbox_to_anchor=(0, 1.2))
    save(fig, figs / "scenarios.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    res = ROOT / ("results_quick" if args.quick else "results")
    figs = ROOT / ("figures_quick" if args.quick else "figures")
    figs.mkdir(exist_ok=True)
    for f in (learning_curve, policy_bars, ceiling_vs_variability, frontier, examples, misses, scenarios):
        f(res, figs)
    print(f"figures written to {figs}")


if __name__ == "__main__":
    main()
