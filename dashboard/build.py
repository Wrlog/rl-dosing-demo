"""Build the results dashboard from the committed files in results/ and models/.

    python -m dashboard.build              # writes site/index.html
    python dashboard/build.py --out site

Every number on the page is read from a result file, or, for the description
of the decision problem, from the constants in src/rl_dosing. The charts are
plotly figures built here from those tables and drawn in the browser by
plotly.js. Nothing is retrained. The matplotlib PNGs in figures/ are copied
next to the page as a static fallback.

Only numpy, pandas and plotly are needed (the rl_dosing modules imported here
don't use torch).
"""

from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rl_dosing.env import (  # noqa: E402
    IND_DOSES, INDUCTION_TAUS, INTERVALS_WK, MAINT_DOSES, N_DECISIONS, N_MAINT_DECISIONS, OBS_DIM, REGIMENS,
    penalty,
)
from rl_dosing.experiment import MAIN_POP, SHARED_POP, main_env_config  # noqa: E402
from rl_dosing.model import LLOQ, TARGET  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO_URL = "https://github.com/Wrlog/rl-dosing-demo"
PLOTLY_JS = "https://cdn.jsdelivr.net/npm/plotly.js-basic-dist-min@4.1.1/plotly-basic.min.js"
PLOTLY_SRI = "sha384-N2HZsG+IG/3J8CwhGGYz/kmzZ0sprpPWwjhor9ZI4lxuf47i9DXK2/NDGxMaJiEb"

pio.templates.default = "none"

# ---------------------------------------------------------------------------
# Palette. Figures are built with the light values; the page swaps in the
# dark value for each colour string when the viewer is in dark mode.
# ---------------------------------------------------------------------------

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
DARK = {SURFACE: "#1a1a19", INK: "#ffffff", INK2: "#c3c2b7", GRID: "#2c2c2a", AXIS: "#383835"}

COLORS = {  # one colour per policy, the same as the matplotlib figures
    "Standard": ("#2a78d6", "#3987e5"),
    "Ceiling": ("#eb6834", "#d95926"),
    "MAP-Bayes": ("#1baf7a", "#199e70"),
    "Oracle": ("#eda100", "#c98500"),
    "DQN": ("#e87ba4", "#d55181"),
    "DQN trained with flares": ("#4a3aa7", "#9085e9"),
}
for _light, _dark in COLORS.values():
    DARK[_light] = _dark
FIXED = "#b6b4ad"  # fixed regimens on the frontier
DARK[FIXED] = "#5c5b55"
# kinds of miss: reachable patients on a blue ramp, unreachable on a gray one
MISS_COLORS = ["#86b6ef", "#2a78d6", "#104281", "#c3c1b9", "#7d7b74"]
DARK.update({"#86b6ef": "#b7d3f6", "#104281": "#1c5cab", "#c3c1b9": "#8e8c85", "#7d7b74": "#5c5b55"})

FONT = 'system-ui, -apple-system, "Segoe UI", Roboto, sans-serif'
POLICY_ORDER = ["Standard", "Ceiling", "MAP-Bayes", "Oracle", "DQN", "DQN trained with flares"]
SCENARIOS = {
    "main": "Held-out profiles (main)",
    "seen": "Training profiles, new random effects",
    "flares": "Flares on",
    "shared_ic50": f"IC50 {SHARED_POP.ic50:g} mg/L (shared model)",
}
SCENARIO_SHORT = {"main": "Held-out profiles", "seen": "Training profiles", "flares": "Flares on",
                  "shared_ic50": f"IC50 {SHARED_POP.ic50:g} mg/L"}
METRICS = {  # column: (label, unit kind)
    "pct_at_target": ("At target at the final visit", "pct"),
    "pct_correct": ("Correct decisions", "pct"),
    "pct_stopped": ("Moved to another treatment", "pct"),
    "dose_rate": ("Dose rate (mg/kg per week)", "rate"),
    "mean_return": ("Mean return", "ret"),
    "cave_gcv": ("Exposure CV in the last interval", "pct"),
    "weeks_to_target": ("Weeks to sustained target", "wk"),
}


def rgba(hex_color: str, alpha: float) -> str:
    """rgba() string for a palette colour, registering its dark-mode twin."""
    def conv(h):
        h = h.lstrip("#")
        return f"rgba({int(h[0:2], 16)}, {int(h[2:4], 16)}, {int(h[4:6], 16)}, {alpha:g})"
    light = conv(hex_color)
    DARK[light] = conv(DARK.get(hex_color, hex_color))
    return light


def color(policy: str) -> str:
    key = "DQN" if policy.startswith("DQN") and policy not in COLORS else policy
    return COLORS[key][0]


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def num(v: float, digits: int = 1, sign: bool = False) -> str:
    s = f"{v:+.{digits}f}" if sign else f"{v:.{digits}f}"
    return s.replace("-", "−")


def fmt(v: float, kind: str) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    return {"pct": num(v, 1), "rate": num(v, 2), "ret": num(v, 2), "wk": num(v, 1)}[kind]


def ci(v: float, lo: float, hi: float, kind: str) -> str:
    return f"{fmt(v, kind)} ({fmt(lo, kind)} to {fmt(hi, kind)})"


# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------

def axis(title: str = "", **kw) -> dict:
    a = dict(title=dict(text=title, font=dict(size=12, color=INK2), standoff=8), gridcolor=GRID, gridwidth=1,
             zeroline=False, showline=True, linecolor=AXIS, linewidth=1, ticks="",
             tickfont=dict(size=11, color=INK2), automargin=True)
    a.update(kw)
    return a


def base_layout(xtitle: str = "", ytitle: str = "", **kw) -> dict:
    lay = dict(
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        font=dict(family=FONT, size=12, color=INK),
        margin=dict(l=8, r=12, t=8, b=8),
        xaxis=axis(xtitle), yaxis=axis(ytitle),
        hoverlabel=dict(bgcolor=SURFACE, bordercolor=GRID, align="left", font=dict(family=FONT, size=12, color=INK)),
        hovermode="closest",
        legend=dict(orientation="h", x=0, xanchor="left", y=1.02, yanchor="bottom", bgcolor="rgba(0,0,0,0)",
                    font=dict(size=11.5, color=INK2), itemclick="toggle", itemdoubleclick="toggleothers"),
        dragmode=False,
        showlegend=True,
    )
    lay.update(kw)
    return lay


def log_ticks(lo: float, hi: float) -> dict:
    """1-2-5 tick values between lo and hi, for log axes (plotly's default labels every minor digit)."""
    vals = [m * 10.0 ** e for e in range(-1, 6) for m in (1, 2, 5)]
    vals = [v for v in vals if lo <= v <= hi]
    return dict(tickvals=vals, ticktext=[f"{v:g}" for v in vals])


def to_json(fig: go.Figure) -> dict:
    return json.loads(pio.to_json(fig, validate=False))


def err(v, lo, hi) -> dict:
    return dict(type="data", symmetric=False, array=list(np.asarray(hi) - np.asarray(v)),
                arrayminus=list(np.asarray(v) - np.asarray(lo)), color=INK2, thickness=1.2, width=3)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

class Results:
    def __init__(self, res: Path, models: Path):
        self.res = res
        self.summary = {s: pd.read_csv(res / f"summary_{s}.csv") for s in SCENARIOS}
        self.patients = {s: pd.read_csv(res / f"patients_{s}.csv") for s in SCENARIOS}
        self.paired = pd.read_csv(res / "paired_differences.csv")
        self.frontier = pd.read_csv(res / "frontier.csv")
        self.cost = pd.read_csv(res / "cost_tradeoff.csv")
        self.refs = pd.read_csv(res / "eval_references.csv").set_index("policy")
        self.ladder = pd.read_csv(res / "ic50_ladder.csv")
        self.misses = pd.read_csv(res / "misses.csv").set_index("policy")
        self.margins = pd.read_csv(res / "margin_tuning.csv")
        self.visits = pd.read_csv(res / "example_visits.csv")
        self.curves_ex = pd.read_csv(res / "example_curves.csv")
        self.runs = {p.stem: json.loads(p.read_text()) for p in sorted(models.glob("*.json"))}
        self.curves = {name: pd.read_csv(res / f"learning_curve_{name}.csv") for name in self.runs
                       if (res / f"learning_curve_{name}.csv").exists()}

    def patient_rows(self, scen: str, policy: str) -> pd.DataFrame:
        p = self.patients[scen]
        if policy in set(p.policy):
            return p[p.policy == policy]
        if policy == "DQN":
            return p[p.policy.str.startswith("DQN seed")]
        return p.iloc[0:0]


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_reward(cfg) -> dict:
    b = np.geomspace(TARGET / 4, TARGET * 2 ** (cfg.penalty_cap + 1), 300)
    r = -penalty(b, cfg.penalty_cap)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=b, y=r, mode="lines", line=dict(color=INK2, width=2), name="Target part of the reward",
                             hovertemplate="Biomarker %{x:.0f} units<br>reward %{y:.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[b[0], b[-1]], y=[-cfg.stop_cost] * 2, mode="lines",
                             line=dict(color=COLORS["Ceiling"][0], width=2, dash="dash"),
                             name="Stop: cost per decision left",
                             hovertemplate=f"Stopping: {-cfg.stop_cost:g} per decision left<extra></extra>"))
    be = TARGET * 2 ** cfg.stop_cost
    fig.add_vline(x=TARGET, line=dict(color=MUTED, width=1, dash="dot"))
    fig.add_annotation(x=np.log10(TARGET), y=-cfg.penalty_cap, text=f"target {TARGET:g}", showarrow=False,
                       xanchor="right", yanchor="bottom", xshift=-4, font=dict(size=11, color=INK2))
    fig.add_annotation(x=np.log10(be), y=-cfg.stop_cost, text=f"break-even {be:.0f}", showarrow=True, arrowhead=0,
                       ax=40, ay=-26, font=dict(size=11, color=INK2), arrowcolor=MUTED)
    fig.update_layout(**base_layout("True biomarker at the end of the interval (units, log scale)", "Reward",
                                    height=280))
    fig.update_xaxes(type="log", **log_ticks(b[0], b[-1]))
    fig.update_yaxes(range=[-cfg.penalty_cap - 0.2, 0.3])
    return to_json(fig)


def fig_policy(R: Results, scen: str, metric: str) -> dict:
    s = R.summary[scen].set_index("policy")
    pols = [p for p in POLICY_ORDER if p in s.index]
    label, kind = METRICS[metric]
    v, lo, hi = (s.loc[pols, c].to_numpy(float) for c in (metric, metric + "_lo", metric + "_hi"))
    ok = ~np.isnan(v)
    unit = "%" if kind == "pct" else ""
    fig = go.Figure()
    seeds_txt = [f"{int(s.loc[p, 'n_seeds'])} seeds" if s.loc[p, "n_seeds"] > 1 else "1 network" if
                 p.startswith("DQN") else "" for p in pols]
    fig.add_trace(go.Bar(
        x=np.where(ok, v, 0), y=pols, orientation="h", marker=dict(color=[color(p) for p in pols], cornerradius=3),
        error_x=err(np.where(ok, v, 0), np.where(ok, lo, 0), np.where(ok, hi, 0)), width=0.62, showlegend=False,
        customdata=np.c_[lo, hi, seeds_txt],
        hovertemplate=f"%{{y}}<br>{label}: %{{x:.2f}}{unit}<br>95% CI %{{customdata[0]:.2f}} to "
                      f"%{{customdata[1]:.2f}}<br>%{{customdata[2]}}<extra></extra>"))
    seeds = [p for p in s.index if p.startswith("DQN seed")]
    if seeds and "DQN" in pols:
        fig.add_trace(go.Scatter(
            x=s.loc[seeds, metric], y=["DQN"] * len(seeds), mode="markers", name="DQN, single seeds",
            marker=dict(color=INK, size=8, symbol="circle", line=dict(color=SURFACE, width=2)),
            customdata=seeds, hovertemplate=f"%{{customdata}}: %{{x:.2f}}{unit}<extra></extra>"))
    xa = dict(range=[0, 100]) if kind == "pct" and metric != "cave_gcv" else {}
    fig.update_layout(**base_layout(label + (" (%)" if kind == "pct" else ""), "", height=60 + 44 * len(pols),
                                    showlegend=bool(seeds and "DQN" in pols)))
    fig.update_xaxes(**xa)
    fig.update_yaxes(autorange="reversed", showgrid=False, showline=False, ticklabelstandoff=8)
    return to_json(fig)


PAIRED_METRICS = {"pct_at_target": ("At target", "points"), "pct_correct": ("Correct decisions", "points"),
                  "dose_rate": ("Dose rate", "mg/kg per week"), "mean_return": ("Mean return", "")}


def fig_paired(R: Results, scen: str, metric: str) -> dict:
    d = R.paired[R.paired.scenario == scen]
    labels = [f"{a} vs {b}" for a, b in zip(d.a, d.b)]
    name, unit = PAIRED_METRICS[metric]
    v, lo, hi = d[metric].to_numpy(), d[metric + "_lo"].to_numpy(), d[metric + "_hi"].to_numpy()
    fig = go.Figure(go.Scatter(
        x=v, y=labels, mode="markers", marker=dict(color=INK, size=10, line=dict(color=SURFACE, width=2)),
        error_x=err(v, lo, hi) | dict(color=INK2, thickness=2, width=0), showlegend=False,
        customdata=np.c_[lo, hi],
        hovertemplate=f"%{{y}}<br>{name}: %{{x:+.2f}} {unit}<br>95% CI %{{customdata[0]:+.2f}} to "
                      f"%{{customdata[1]:+.2f}}<extra></extra>"))
    fig.add_vline(x=0, line=dict(color=AXIS, width=1.5))
    title = f"First minus second: {name.lower()}" + (f" ({unit})" if unit else "")
    fig.update_layout(**base_layout(title, "", height=70 + 46 * len(labels), showlegend=False))
    fig.update_yaxes(autorange="reversed", showgrid=False, showline=False, ticklabelstandoff=8)
    return to_json(fig)


def regimen_text(label: str) -> str:
    dose, w = label.split(" q")
    return f"{dose} mg/kg every {w.rstrip('w')} weeks"


def fig_frontier(R: Results, metric: str) -> dict:
    f, s = R.frontier, R.summary["main"].set_index("policy")
    label = METRICS[metric][0]
    fig = go.Figure()
    fx = f[f.family == "Fixed regimen"]
    fig.add_trace(go.Scatter(
        x=fx.dose_rate, y=fx[metric], mode="markers", name="Fixed regimens",
        marker=dict(color=FIXED, size=9, line=dict(color=SURFACE, width=1.5)),
        customdata=[regimen_text(x) for x in fx.label],
        hovertemplate=f"%{{customdata}}<br>{label}: %{{y:.1f}}%<br>%{{x:.2f}} mg/kg per week<extra></extra>"))
    mp = f[f.family == "MAP-Bayes"].sort_values("dose_rate")
    fig.add_trace(go.Scatter(
        x=mp.dose_rate, y=mp[metric], mode="lines+markers", name="MAP-Bayes, by margin",
        line=dict(color=COLORS["MAP-Bayes"][0], width=2),
        marker=dict(size=9, line=dict(color=SURFACE, width=1.5)), customdata=mp.label,
        hovertemplate=f"MAP-Bayes, %{{customdata}}<br>{label}: %{{y:.1f}}%<br>%{{x:.2f}} mg/kg per week"
                      "<extra></extra>"))
    dq = f[f.family == "DQN"].sort_values("dose_rate")
    fig.add_trace(go.Scatter(
        x=dq.dose_rate, y=dq[metric], mode="lines+markers+text", name="DQN, by drug cost λ",
        line=dict(color=COLORS["DQN"][0], width=2), marker=dict(size=9, symbol="diamond",
                                                                line=dict(color=SURFACE, width=1.5)),
        text=[f"λ {c:g}" for c in dq.drug_cost], textposition="top right",
        textfont=dict(size=11, color=INK2), customdata=dq.drug_cost,
        hovertemplate=f"DQN, drug cost %{{customdata:g}}<br>{label}: %{{y:.1f}}%<br>%{{x:.2f}} mg/kg per week"
                      "<extra></extra>"))
    for pol, sym in (("Standard", "square"), ("Ceiling", "square"), ("Oracle", "star")):
        fig.add_trace(go.Scatter(
            x=[s.loc[pol, "dose_rate"]], y=[s.loc[pol, metric]], mode="markers", name=pol,
            marker=dict(color=color(pol), size=13 if sym == "star" else 10, symbol=sym,
                        line=dict(color=SURFACE, width=1.5)),
            hovertemplate=f"{pol}<br>{label}: %{{y:.1f}}%<br>%{{x:.2f}} mg/kg per week<extra></extra>"))
    fig.update_layout(**base_layout("Dose rate (mg/kg per week)", f"{label} (%)", height=430, dragmode="zoom"))
    return to_json(fig)


LC_METRICS = {"pct_at_target": ("At target", "pct"), "pct_correct": ("Correct decisions", "pct"),
              "return": ("Mean return", "ret"), "dose_rate": ("Dose rate (mg/kg per week)", "rate"),
              "pct_stopped": ("Moved to another treatment", "pct")}
REF_STYLE = {"Ceiling (reachable)": ("Ceiling", "solid", "Ceiling (share reachable)"),
             "MAP-Bayes": ("MAP-Bayes", "dash", "MAP-Bayes"), "Oracle": ("Oracle", "dot", "Oracle"),
             "Standard": ("Standard", "dashdot", "Standard")}


def run_groups(R: Results) -> dict:
    main = sorted(n for n, j in R.runs.items() if set(j["spec"]) == {"seed"})
    cost = sorted((n for n, j in R.runs.items() if set(j["spec"]) <= {"seed", "drug_cost"} and j["spec"]["seed"] == 0),
                  key=lambda n: R.runs[n]["drug_cost"])
    other = sorted(n for n, j in R.runs.items() if j["spec"].get("flares") or j["spec"].get("shared_ic50"))
    return {"main": main, "cost": cost, "other": other}


def run_label(R: Results, name: str, group: str) -> str:
    j = R.runs[name]
    if group == "main":
        return f"DQN seed {j['seed']}"
    if group == "cost":
        return f"DQN, drug cost {j['drug_cost']:g}"
    if j["spec"].get("flares"):
        return "DQN trained with flares"
    return f"DQN, IC50 {j['ic50_typical']:g} mg/L"


def fig_learning(R: Results, group: str, metric: str) -> dict:
    names = run_groups(R)[group]
    label, kind = LC_METRICS[metric]
    unit = "%" if kind == "pct" else ""
    dashes = ["solid", "dash", "dot", "dashdot"]
    fig = go.Figure()
    xmax = 0
    for i, n in enumerate(names):
        c = R.curves[n]
        xmax = max(xmax, c.episode.max())
        name = run_label(R, n, group)
        col = COLORS["DQN trained with flares"][0] if "flares" in name else COLORS["DQN"][0]
        opacity = 1.0 if group != "cost" else (0.45, 0.65, 0.85, 1.0)[i % 4]
        fig.add_trace(go.Scatter(
            x=c.episode, y=c[metric], mode="lines", name=name, legendgroup=n, opacity=opacity,
            line=dict(color=col, width=2, dash=dashes[i % 4]),
            hovertemplate=f"{name}<br>episode %{{x:,}}<br>{label}: %{{y:.2f}}{unit}<extra></extra>"))
        best = R.runs[n]["best_episode"]
        row = c.loc[(c.episode - best).abs().idxmin()]
        fig.add_trace(go.Scatter(
            x=[row.episode], y=[row[metric]], mode="markers", legendgroup=n, showlegend=False, opacity=opacity,
            marker=dict(color=SURFACE, size=10, line=dict(color=col, width=2)),
            hovertemplate=f"{name}: checkpoint kept<br>episode %{{x:,}}<br>{label}: %{{y:.2f}}{unit}"
                          "<extra></extra>"))
    # the other policies on the same checkpoint cohort (main settings only)
    ref_col = {"pct_at_target": "pct_at_target", "pct_correct": "pct_correct", "return": "mean_return"}.get(metric)
    if ref_col and group != "other" and (metric != "return" or group == "main"):
        for pol, (key, dash, name) in REF_STYLE.items():
            if pol not in R.refs.index or pd.isna(R.refs.loc[pol, ref_col]):
                continue
            y = R.refs.loc[pol, ref_col]
            fig.add_trace(go.Scatter(
                x=[0, xmax], y=[y, y], mode="lines", name=name, line=dict(color=color(key), width=1.6, dash=dash),
                hovertemplate=f"{name}: %{{y:.2f}}{unit}<extra></extra>"))
    fig.update_layout(**base_layout("Training episodes (one patient each)", label + (" (%)" if unit else ""),
                                    height=400, dragmode="zoom"))
    if kind == "pct":
        fig.update_yaxes(range=[0, 101])
    return to_json(fig)


def fig_ladder(R: Results) -> dict:
    lad = R.ladder
    dashes = ["solid", "dash", "dot"]
    symbols = ["circle", "square", "triangle-up"]
    fig = go.Figure()
    for i, setting in enumerate(lad.setting.unique()):
        for reg in ("Ceiling", "Standard"):
            d = lad[(lad.setting == setting) & (lad.regimen == reg)].sort_values("ic50_cv")
            x = 100 * d.ic50_cv
            grp = f"{reg}|{setting}"
            fig.add_trace(go.Scatter(x=x, y=d.lo, mode="lines", line=dict(width=0), legendgroup=grp,
                                     showlegend=False, hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=x, y=d.hi, mode="lines", line=dict(width=0), fill="tonexty",
                                     fillcolor=rgba(color(reg), 0.14), legendgroup=grp, showlegend=False,
                                     hoverinfo="skip"))
            fig.add_trace(go.Scatter(
                x=x, y=d.pct_at_target, mode="lines+markers", name=f"{reg}, {setting}", legendgroup=grp,
                line=dict(color=color(reg), width=2, dash=dashes[i]),
                marker=dict(size=8, symbol=symbols[i], line=dict(color=SURFACE, width=1.5)),
                customdata=np.c_[d.lo, d.hi],
                hovertemplate=f"{reg}, {setting}<br>IC50 CV %{{x:.0f}}%: %{{y:.1f}}% at target"
                              "<br>95% CI %{customdata[0]:.1f} to %{customdata[1]:.1f}<extra></extra>"))
    fig.update_layout(**base_layout("IC50 between-patient variability (CV %), shrinking to the right",
                                    "At target at the final visit (%)", height=440))
    fig.update_xaxes(autorange="reversed")
    fig.update_yaxes(range=[0, 101])
    return to_json(fig)


EXAMPLE_POLICIES = [("Standard", "Standard"), ("MAP-Bayes", "MAP-Bayes"), ("DQN seed 0", "DQN (seed 0)")]


def case_description(R: Results, patient: int) -> str:
    std = R.patients["main"]
    row = std[(std.policy == "Standard") & (std.patient == patient)].iloc[0]
    if not row.reachable:
        return "unreachable, even the ceiling regimen misses target"
    if row.at_target:
        return "at target on the standard regimen"
    return "reachable, missed by the standard regimen"


def fig_example(R: Results, case: int) -> dict:
    v = R.visits[R.visits.case == case]
    c = R.curves_ex[R.curves_ex.case == case]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.07)
    for i, (pol, name) in enumerate(EXAMPLE_POLICIES):
        col = color(pol)
        cc, vv = c[c.policy == pol], v[v.policy == pol]
        fig.add_trace(go.Scatter(
            x=cc.day / 7, y=cc.b_true, mode="lines", name=name, legendgroup=pol, line=dict(color=col, width=2),
            hovertemplate=f"{name}<br>week %{{x:.1f}}: true biomarker %{{y:.0f}}<extra></extra>"), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=vv.day / 7, y=vv.b_obs, mode="markers", name=name, legendgroup=pol, showlegend=False,
            marker=dict(color=col, size=8, line=dict(color=SURFACE, width=1.5)),
            hovertemplate=f"{name}<br>visit at week %{{x:.0f}}: measured %{{y:.0f}}<extra></extra>"), row=1, col=1)
        st = vv[vv.stopped_here]
        if len(st):
            fig.add_trace(go.Scatter(
                x=st.day / 7, y=st.b_obs, mode="markers", name=name, legendgroup=pol, showlegend=False,
                marker=dict(color=col, size=14, symbol="x", line=dict(color=INK, width=1)),
                hovertemplate=f"{name} stops Drug X at week %{{x:.0f}}<extra></extra>"), row=1, col=1)
        dd = vv.dropna(subset=["dose_mgkg"])
        fig.add_trace(go.Bar(
            x=dd.day / 7, y=dd.dose_mgkg, name=name, legendgroup=pol, showlegend=False, offsetgroup=str(i),
            marker=dict(color=col, cornerradius=2), width=1.1,
            customdata=dd.tau_days / 7,
            hovertemplate=f"{name}<br>week %{{x:.0f}}: %{{y:g}} mg/kg, next visit in %{{customdata:g}} weeks"
                          "<extra></extra>"), row=2, col=1)
    fig.add_hline(y=TARGET, line=dict(color=INK2, width=1, dash="dash"), row=1, col=1)
    lay = base_layout(height=470, barmode="group", bargap=0.1)
    lay.pop("xaxis"), lay.pop("yaxis")
    fig.update_layout(**lay)
    fig.update_xaxes(**axis())
    fig.update_xaxes(title=dict(text="Weeks since the first dose", font=dict(size=12, color=INK2)), row=2, col=1)
    fig.update_yaxes(**axis("Biomarker (units, log)"), row=1, col=1)
    lo, hi = min(c.b_true.min(), v.b_obs.min()), max(c.b_true.max(), v.b_obs.max())
    ylo, yhi = min(lo, TARGET) * 0.7, hi * 1.4
    fig.update_yaxes(type="log", range=[np.log10(ylo), np.log10(yhi)], **log_ticks(ylo, yhi), row=1, col=1)
    fig.update_yaxes(**axis("Dose (mg/kg)"), row=2, col=1)
    fig.update_yaxes(range=[0, max(MAINT_DOSES) * 1.1], row=2, col=1)
    fig.add_annotation(x=0, xref="x domain", y=np.log10(TARGET), yref="y", text=f"target {TARGET:g}",
                       showarrow=False, xanchor="left", yanchor="bottom", font=dict(size=11, color=INK2))
    return to_json(fig)


def example_summary(R: Results, case: int) -> str:
    v = R.visits[R.visits.case == case]
    n_ind = len(INDUCTION_TAUS)
    items = []
    for pol, name in EXAMPLE_POLICIES:
        vv = v[v.policy == pol].sort_values("visit")
        doses = vv.dropna(subset=["dose_mgkg"])
        maint = [f"{d:g} mg/kg every {t / 7:g} weeks" for d, t in
                 zip(doses.dose_mgkg.iloc[n_ind:], doses.tau_days.iloc[n_ind:])]
        text = f"induction {doses.dose_mgkg.iloc[0]:g} mg/kg"
        if maint:
            text += ", then " + "; ".join(maint)
        if vv.stopped_here.any():
            text += f"; stopped at week {vv[vv.stopped_here].day.iloc[0] / 7:.0f}"
        last = vv.iloc[-1]
        text += f". Last true biomarker {last.b_true:.0f}."
        items.append(f'<li><span class="swatch" style="background:{color(pol)}"></span>{html.escape(name)}: '
                     f"{html.escape(text)}</li>")
    return '<ul class="regimens">' + "".join(items) + "</ul>"


def miss_kinds(R: Results):
    m = R.misses
    be = m["break_even"].iloc[0]
    kinds = [k for k in m.columns if k.startswith(("Reachable", "Unreachable"))]
    labels = [k.replace("break-even", f"{be:.0f}") for k in kinds]
    return kinds, labels, be


def fig_misses(R: Results, with_standard: bool) -> dict:
    m = R.misses
    kinds, labels, _ = miss_kinds(R)
    pols = [p for p in ("Standard", "MAP-Bayes", "Oracle", "DQN") if p in m.index and (with_standard or
                                                                                     p != "Standard")]
    n = m.loc[pols, "n_patients"].to_numpy(float)
    fig = go.Figure()
    for k, lab, col in zip(kinds, labels, MISS_COLORS):
        cnt = m.loc[pols, k].to_numpy(float)
        fig.add_trace(go.Bar(
            x=100 * cnt / n, y=pols, orientation="h", name=lab, marker=dict(color=col, line=dict(color=SURFACE,
                                                                                              width=2)),
            customdata=cnt, hovertemplate=f"%{{y}}<br>{lab}<br>%{{customdata:.1f}} patients (%{{x:.1f}}%)"
                                          "<extra></extra>"))
    fig.update_layout(**base_layout("Incorrect decisions (% of patients)", "",
                                    height=150 + 46 * len(pols), barmode="stack", bargap=0.35))
    fig.update_yaxes(autorange="reversed", showgrid=False, showline=False, ticklabelstandoff=8)
    fig.update_layout(legend=dict(orientation="h", y=1.02, yanchor="bottom", traceorder="normal"))
    return to_json(fig)


def fig_final_b(R: Results) -> dict:
    p = R.patients["main"]
    _, _, be = miss_kinds(R)
    keep = p[p.reachable & ~p.stopped]
    edges = np.geomspace(keep.final_b.quantile(0.01), keep.final_b.quantile(0.99), 31)
    mids = np.sqrt(edges[:-1] * edges[1:])
    fig = go.Figure()
    for pol in ("Standard", "Ceiling", "MAP-Bayes", "Oracle", "DQN"):
        allp = R.patient_rows("main", pol)
        kp = allp[allp.reachable & ~allp.stopped]
        h, _ = np.histogram(kp.final_b, edges)
        pct = 100 * h / len(allp)
        name = "DQN (3 seeds pooled)" if pol == "DQN" else pol
        fig.add_trace(go.Scatter(
            x=mids, y=pct, mode="lines", name=name, line=dict(color=color(pol), width=2),
            visible="legendonly" if pol == "Ceiling" else True,
            customdata=np.c_[edges[:-1], edges[1:]],
            hovertemplate=f"{name}<br>%{{customdata[0]:.0f}} to %{{customdata[1]:.0f}} units: "
                          "%{y:.1f}% of patients<extra></extra>"))
    fig.add_vline(x=TARGET, line=dict(color=INK2, width=1, dash="dash"))
    fig.add_vline(x=be, line=dict(color=MUTED, width=1, dash="dot"))
    fig.add_annotation(x=np.log10(TARGET), y=1, yref="paper", text=f"target {TARGET:g}", showarrow=False,
                       xanchor="right", yanchor="top", xshift=-4, font=dict(size=11, color=INK2))
    fig.add_annotation(x=np.log10(be), y=1, yref="paper", text=f"break-even {be:.0f}", showarrow=False,
                       xanchor="left", yanchor="top", xshift=4, font=dict(size=11, color=INK2))
    fig.update_layout(**base_layout("Final true biomarker (units, log scale)", "Patients (% of all, per bin)",
                                    height=360, hovermode="x"))
    fig.update_xaxes(type="log", **log_ticks(edges[0], edges[-1]))
    return to_json(fig)


def fig_scenarios(R: Results, metric: str) -> dict:
    label = METRICS[metric][0]
    fig = go.Figure()
    xs = [SCENARIO_SHORT[s] for s in SCENARIOS]
    for pol in POLICY_ORDER:
        rows = [R.summary[s].set_index("policy").reindex([pol]).iloc[0] for s in SCENARIOS]
        v = np.array([r[metric] for r in rows], float)
        if np.isnan(v).all():
            continue
        lo = np.array([r[metric + "_lo"] for r in rows], float)
        hi = np.array([r[metric + "_hi"] for r in rows], float)
        seeds = [f"{int(r['n_seeds'])} seed{'s' if r['n_seeds'] > 1 else ''}" if pol.startswith("DQN") and
                 not np.isnan(r["n_seeds"]) else "" for r in rows]
        fig.add_trace(go.Bar(
            x=xs, y=v, name=pol, marker=dict(color=color(pol), cornerradius=3), error_y=err(v, lo, hi),
            customdata=np.c_[lo, hi, seeds],
            hovertemplate=f"{pol}, %{{x}}<br>{label}: %{{y:.1f}}%<br>95% CI %{{customdata[0]:.1f}} to "
                          "%{customdata[1]:.1f}<br>%{customdata[2]}<extra></extra>"))
    fig.update_layout(**base_layout("", f"{label} (%)", height=400, barmode="group", bargap=0.22,
                                    bargroupgap=0.08))
    fig.update_yaxes(range=[0, 102])
    fig.update_xaxes(showgrid=False)
    return to_json(fig)


# ---------------------------------------------------------------------------
# HTML pieces
# ---------------------------------------------------------------------------

def esc(s) -> str:
    return html.escape(str(s))


def select(group: str, dim: str, label: str, options: list[tuple[str, str]]) -> str:
    opts = "".join(f'<option value="{esc(v)}">{esc(t)}</option>' for v, t in options)
    return (f'<label class="control">{esc(label)}<select data-group="{group}" data-dim="{dim}">{opts}</select>'
            "</label>")


def plot(group: str, chart: str, alt: str, png: str | None = None) -> str:
    fb = (f'<img class="fallback" src="figures/{png}" alt="{esc(alt)}" loading="lazy">' if png else
          '<p class="fallback note">This chart needs JavaScript and plotly.js.</p>')
    return (f'<figure class="fig"><div class="plot" data-group="{group}" data-chart="{chart}" role="img" '
            f'aria-label="{esc(alt)}"></div>{fb}</figure>')


def png_link(png: str) -> str:
    return f'<a class="png" href="figures/{png}" download>Static PNG</a>'


def table(headers: list[str], rows: list[list[tuple[str, float | str]]], sortable: bool = True) -> str:
    head = "".join(f'<th scope="col">{esc(h)}</th>' for h in headers)
    body = []
    for r in rows:
        cells = []
        for j, (text, key) in enumerate(r):
            sk = "" if key is None or (isinstance(key, float) and np.isnan(key)) else key
            tag = "th scope=\"row\"" if j == 0 else "td"
            cells.append(f'<{tag} data-sort="{esc(sk)}">{text}</{tag.split()[0]}>')
        body.append("<tr>" + "".join(cells) + "</tr>")
    cls = ' class="sortable"' if sortable else ""
    return (f'<div class="table-wrap"><table{cls}><thead><tr>{head}</tr></thead><tbody>{"".join(body)}'
            "</tbody></table></div>")


def card(title: str, sub: str, body: str, controls: str = "", extra_cls: str = "", aside: str = "") -> str:
    head = f'<div><h2>{esc(title)}</h2><p class="sub">{sub}</p></div>'
    ctrl = f'<div class="controls">{controls}</div>' if controls else ""
    return (f'<article class="card {extra_cls}"><div class="card-head">{head}{ctrl}</div>{body}'
            f'{f"<div class=card-foot>{aside}</div>" if aside else ""}</article>')


def tile(label: str, value: str, detail: str) -> str:
    return (f'<div class="tile"><div class="label">{esc(label)}</div><div class="value">{value}</div>'
            f'<div class="detail">{detail}</div></div>')


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

def build(res: Path, models: Path, figs: Path, out: Path) -> Path:
    R = Results(res, models)
    cfg = main_env_config()
    gamma = next(iter(R.runs.values()))["gamma"]
    charts: dict[str, dict[str, dict]] = {}
    sm = R.summary["main"].set_index("policy")
    n_main = len(R.patient_rows("main", "Standard"))
    pairs = R.paired.set_index(["scenario", "a", "b"])
    dqn_map = pairs.loc[("main", "DQN", "MAP-Bayes")]
    dqn_std = pairs.loc[("main", "DQN", "Standard")]
    reach_main = 100 * R.patient_rows("main", "Standard").reachable.mean()
    n_seeds = int(sm.loc["DQN", "n_seeds"])
    _, _, break_even = miss_kinds(R)

    # ----- tiles
    tiles = "".join([
        tile("DQN at target", f"{fmt(sm.loc['DQN', 'pct_at_target'], 'pct')}%",
             f"95% CI {fmt(sm.loc['DQN', 'pct_at_target_lo'], 'pct')} to "
             f"{fmt(sm.loc['DQN', 'pct_at_target_hi'], 'pct')}, mean of {n_seeds} seeds"),
        tile("MAP-Bayes at target", f"{fmt(sm.loc['MAP-Bayes', 'pct_at_target'], 'pct')}%",
             f"95% CI {fmt(sm.loc['MAP-Bayes', 'pct_at_target_lo'], 'pct')} to "
             f"{fmt(sm.loc['MAP-Bayes', 'pct_at_target_hi'], 'pct')}"),
        tile("DQN minus MAP-Bayes", f"{num(dqn_map.pct_at_target, 1, True)} pts",
             f"at target, paired 95% CI {num(dqn_map.pct_at_target_lo)} to {num(dqn_map.pct_at_target_hi)}"),
        tile("DQN minus standard regimen", f"{num(dqn_std.pct_at_target, 1, True)} pts",
             f"at target, paired 95% CI {num(dqn_std.pct_at_target_lo, 1, True)} to "
             f"{num(dqn_std.pct_at_target_hi, 1, True)}"),
        tile("Reachable on the ceiling regimen", f"{fmt(reach_main, 'pct')}%",
             f"of the {n_main:,} held-out patients"),
    ])

    # ----- the decision problem
    charts["reward"] = {"": fig_reward(cfg)}
    start_maint = sum(INDUCTION_TAUS)
    ind_days = ", ".join(f"{d:g}" for d in np.cumsum((0,) + INDUCTION_TAUS[:-1]))
    mdp = f"""
    <div class="mdp">
      <div><h3>What it sees</h3><p>{OBS_DIM} numbers at each decision: the measured biomarker at baseline, at the
        previous visit and now, the measured trough concentration (with a flag when it's below the LLOQ of
        {LLOQ:g} mg/L), the last dose and interval, the induction dose, time since the first dose, the latest ADA
        result, and baseline weight, albumin, INF and PLT. It never sees the patient's true clearance, IC50 or
        baseline level, or whether the patient can reach target.</p></div>
      <div><h3>What it can do</h3><p>At day 0 it picks the induction dose, one of {len(IND_DOSES)} doses from
        {min(IND_DOSES):g} to {max(IND_DOSES):g} mg/kg, given at days {ind_days}. From day {start_maint:g} it makes
        {N_MAINT_DECISIONS} maintenance decisions. Each is one of {len(REGIMENS)} regimens ({len(MAINT_DOSES)} doses
        times intervals of {", ".join(f"{w:g}" for w in INTERVALS_WK)} weeks), or stopping Drug X and moving the
        patient to another treatment. Stopping isn't allowed at the induction decision.</p></div>
      <div><h3>The reward</h3><p>After each interval it loses one point for each doubling of the true biomarker
        above {TARGET:g} units (capped at {cfg.penalty_cap:g}) plus {cfg.drug_cost:g} per mg/kg per week of drug.
        Stopping ends the episode and costs {cfg.stop_cost:g} for each maintenance decision left. Discount factor
        {gamma:g}.</p></div>
    </div>"""
    mdp_card = card(
        "The decision problem",
        f"One episode is one virtual patient and {N_DECISIONS} decisions. The chart shows the target part of the reward and "
        "the cost of stopping; it's shallow just above target, which matters for the misses below.",
        mdp + plot("reward", "reward", "Reward against the true biomarker at the end of an interval"),
        aside=f"One change from the shared Drug X model: the typical IC50 here is {MAIN_POP.ic50:g} mg/L instead of "
              f"{SHARED_POP.ic50:g}. The shared value is kept as a sensitivity run.")

    # ----- policy comparison
    pol_metrics = ["pct_at_target", "pct_correct", "dose_rate", "pct_stopped", "mean_return", "cave_gcv",
                   "weeks_to_target"]
    charts["policy"] = {f"{s}|{m}": fig_policy(R, s, m) for s in SCENARIOS for m in pol_metrics}
    policy_card = card(
        "Policies on the same held-out patients",
        f"{n_main:,} virtual patients per scenario. Bars show 95% bootstrap CIs over patients. The DQN bar averages "
        "the training seeds and the dots are the single seeds.",
        plot("policy", "policy", "Horizontal bar chart of the chosen endpoint by policy", "policy_bars.png"),
        controls=select("policy", "scenario", "Scenario", list(SCENARIOS.items()))
        + select("policy", "metric", "Endpoint", [(m, METRICS[m][0]) for m in pol_metrics]),
        aside=png_link("policy_bars.png"))

    charts["paired"] = {f"{s}|{m}": fig_paired(R, s, m) for s in SCENARIOS for m in PAIRED_METRICS}
    paired_card = card(
        "Paired differences",
        "Each row is the first policy minus the second on the same patients, with a paired bootstrap 95% CI. "
        "Left of zero means the first policy scored lower.",
        plot("paired", "paired", "Paired differences between policies with confidence intervals"),
        controls=select("paired", "scenario", "Scenario", list(SCENARIOS.items()))
        + select("paired", "metric", "Endpoint", [(m, v[0]) for m, v in PAIRED_METRICS.items()]))

    # ----- frontier and drug cost
    charts["frontier"] = {m: fig_frontier(R, m) for m in ("pct_correct", "pct_at_target")}
    n_fixed = int((R.frontier.family == "Fixed regimen").sum())
    mp_margins = R.frontier[R.frontier.family == "MAP-Bayes"].label.str.replace("margin ", "")
    cost_rows = []
    for _, r in R.cost.iterrows():
        main_row = np.isclose(r.drug_cost, cfg.drug_cost)
        lab = f"{r.drug_cost:g}" + (f" (main, {n_seeds} seeds)" if main_row else "")
        cost_rows.append([(lab, r.drug_cost), (f"{fmt(r.pct_at_target, 'pct')}%", r.pct_at_target),
                          (f"{fmt(r.pct_correct, 'pct')}%", r.pct_correct),
                          (f"{fmt(r.pct_stopped, 'pct')}%", r.pct_stopped),
                          (fmt(r.dose_rate, "rate"), r.dose_rate)])
    frontier_card = card(
        "Outcome against drug use",
        f"Gray points are the {n_fixed} fixed regimens (the same dose for induction and maintenance, never "
        f"stopping). The MAP-Bayes line runs over its safety margin from {mp_margins.iloc[0]} to "
        f"{mp_margins.iloc[-1]}, and the DQN line over the drug cost weight λ in its reward. Up and to the "
        "left is better. Drag to zoom, double-click to reset.",
        plot("frontier", "frontier", "Scatter of correct decisions against dose rate", "frontier.png")
        + '<h3 class="table-title">DQN by drug cost λ (held-out patients)</h3>'
        + table(["λ", "At target", "Correct", "Stopped", "Dose rate (mg/kg/week)"], cost_rows),
        controls=select("frontier", "metric", "Endpoint",
                        [("pct_correct", "Correct decisions"), ("pct_at_target", "At target")]),
        aside="As λ rises the DQN saves drug mostly by stopping more patients, since the other treatment "
              "carries no drug cost in this reward. " + png_link("frontier.png"))

    # ----- learning curves
    groups = run_groups(R)
    lc_groups = [("main", "Main runs (seeds)"), ("cost", "Drug cost sweep (seed 0)"),
                 ("other", "Flares and shared IC50 (seed 0)")]
    charts["learning"] = {f"{g}|{m}": fig_learning(R, g, m) for g, _ in lc_groups
                          for m in LC_METRICS if groups[g]}
    eval_every = next(iter(R.runs.values()))["eval_every"]
    learn_card = card(
        "Learning curves",
        f"Each run is scored every {eval_every} episodes on a fixed checkpoint cohort of training-profile patients, "
        "separate from the test patients. Straight lines are the other policies on that cohort (main settings "
        "only). Open circles mark the checkpoint kept, the one with the best mean return.",
        plot("learning", "learning", "Line chart of DQN evaluation scores over training", "learning_curve.png"),
        controls=select("learning", "runs", "Runs", [g for g in lc_groups if groups[g[0]]])
        + select("learning", "metric", "Score", [(m, v[0]) for m, v in LC_METRICS.items()]),
        aside="The flare and shared-IC50 runs are scored on their own settings, so the reference lines don't apply "
              "to them. " + png_link("learning_curve.png"))

    # ----- reachability ladder
    charts["ladder"] = {"": fig_ladder(R)}
    n_ladder = int(R.ladder.n.iloc[0])
    ladder_card = card(
        "Reachability as IC50 variability shrinks",
        f"{n_ladder:,} new patients, changing only the IC50 variance and keeping every other random draw fixed. "
        "A patient is reachable if the ceiling regimen gets the true biomarker below target at the final visit, "
        "so the ceiling line is the share reachable. Bands are 95% CIs. Click the legend to hide a line.",
        plot("ladder", "ladder", "Share at target on the ceiling and standard regimens against IC50 variability",
             "ceiling_vs_ic50_variability.png"),
        aside=png_link("ceiling_vs_ic50_variability.png"))

    # ----- example patients
    cases = sorted(R.visits.case.unique())
    charts["example"] = {str(k): fig_example(R, k) for k in cases}
    case_opts = []
    for k in cases:
        pid = int(R.visits[R.visits.case == k].patient.iloc[0])
        case_opts.append((str(k), f"Patient {pid}: {case_description(R, pid)}"))
    panels = "".join(f'<div data-group="example" data-key="{k}"{"" if i == 0 else " hidden"}>'
                     f"{example_summary(R, k)}</div>" for i, k in enumerate(cases))
    example_card = card(
        "Example patients",
        "Lines are the true biomarker, dots the value measured at each visit, and an X marks where a policy stopped "
        "Drug X. The lower panel shows each dose. Click a policy in the legend to hide it.",
        plot("example", "example", "Biomarker and dose over time for one patient under three policies",
             "example_trajectories.png") + panels,
        controls=select("example", "case", "Patient", case_opts),
        aside=png_link("example_trajectories.png"))

    # ----- misses
    charts["misses"] = {"no": fig_misses(R, False), "yes": fig_misses(R, True)}
    kinds, labels, _ = miss_kinds(R)
    miss_pols = [p for p in ("MAP-Bayes", "Oracle", "DQN", "Standard") if p in R.misses.index]
    miss_rows = []
    for k, lab in zip(kinds + ["incorrect"], labels + ["Total"]):
        miss_rows.append([(esc(lab), lab)] + [(num(R.misses.loc[p, k], 1 if p == "DQN" else 0), R.misses.loc[p, k])
                                              for p in miss_pols])
    misses_card = card(
        "Misses",
        f"An incorrect decision is a reachable patient who doesn't finish at target, or an unreachable patient "
        f"kept on Drug X. The break-even level, {break_even:.0f} units, is where stopping and staying on the drug "
        f"cost the same. Out of {int(R.misses.n_patients.iloc[0]):,} held-out patients; the DQN counts are "
        "averaged over seeds.",
        plot("misses", "misses", "Stacked bars of the kinds of incorrect decision by policy", "misses.png")
        + '<h3 class="table-title">Number of patients</h3>'
        + table(["Kind of miss"] + ["DQN" if p == "DQN" else p for p in miss_pols], miss_rows, sortable=False),
        controls=select("misses", "standard", "Standard regimen", [("no", "Hide"), ("yes", "Show")]),
        aside=png_link("misses.png"))
    charts["finalb"] = {"": fig_final_b(R)}
    finalb_card = card(
        "Where reachable patients finish",
        "Final true biomarker for reachable patients kept on Drug X, as a share of all patients on each policy. "
        "Mass to the right of the target line is a miss. The ceiling regimen is hidden; click it in the legend "
        "to show it.",
        plot("finalb", "finalb", "Distribution of the final biomarker for reachable patients by policy"))

    # ----- scenarios
    charts["scenarios"] = {m: fig_scenarios(R, m) for m in ("pct_correct", "pct_at_target")}
    scen_pols = [p for p in POLICY_ORDER if p != "Ceiling"]
    scen_rows = []
    for s, lab in SCENARIOS.items():
        t = R.summary[s].set_index("policy")
        reach = 100 * R.patient_rows(s, "Standard").reachable.mean()
        row = [(esc(lab), lab), (f"{fmt(reach, 'pct')}%", reach)]
        for p in scen_pols:
            if p in t.index:
                row.append((f"{fmt(t.loc[p, 'pct_at_target'], 'pct')} / {fmt(t.loc[p, 'pct_correct'], 'pct')}",
                            t.loc[p, "pct_correct"]))
            else:
                row.append(("", None))
        scen_rows.append(row)
    shared_dqn_seeds = int(R.summary["shared_ic50"].set_index("policy").loc["DQN", "n_seeds"])
    scen_card = card(
        "Scenarios",
        f"Held-out profiles are the main test. Training profiles reuse covariate profiles seen in training with new "
        f"random effects. Flares add between-occasion variability on the baseline biomarker. The shared-IC50 run "
        f"uses a DQN trained in that setting ({shared_dqn_seeds} seed), and the flare-trained DQN is one seed.",
        plot("scenarios", "scenarios", "Grouped bars of the chosen endpoint by scenario and policy", "scenarios.png")
        + '<h3 class="table-title">At target / correct decisions (%)</h3>'
        + table(["Scenario", "Reachable"] + scen_pols, scen_rows),
        controls=select("scenarios", "metric", "Endpoint",
                        [("pct_correct", "Correct decisions"), ("pct_at_target", "At target")]),
        aside=png_link("scenarios.png"))

    # ----- full tables
    panels = []
    for i, (s, lab) in enumerate(SCENARIOS.items()):
        t = R.summary[s]
        rows = []
        for _, r in t.iterrows():
            pr = R.patient_rows(s, r.policy)
            ada = 100 * pr.ada_end.mean() if len(pr) else np.nan
            ind = pr.induction_mgkg.mean() if len(pr) else np.nan
            row = [(esc(r.policy), r.policy), (str(int(r.n_seeds)), r.n_seeds)]
            for m in ("pct_at_target", "pct_correct", "pct_stopped", "dose_rate", "cave_gcv", "weeks_to_target",
                      "mean_return"):
                kind = METRICS[m][1]
                row.append((ci(r[m], r[m + "_lo"], r[m + "_hi"], kind), r[m]))
            row += [(fmt(ada, "pct"), ada), (fmt(ind, "wk"), ind)]
            rows.append(row)
        panels.append(f'<div data-group="summary" data-key="{s}"{"" if i == 0 else " hidden"}>' + table(
            ["Policy", "Seeds", "At target %", "Correct %", "Stopped %", "Dose rate", "Cave CV %",
             "Weeks to target", "Mean return", "ADA+ at end %", "Induction mg/kg"], rows) + "</div>")
    summary_card = card(
        "All endpoints",
        "Point estimates with 95% CIs. Dose rate is mg/kg per week. Cave CV is the between-patient CV of the "
        "average concentration in the last interval. Weeks to sustained target only counts patients who get there. "
        "Click a column header to sort.",
        "".join(panels),
        controls=select("summary", "scenario", "Scenario", list(SCENARIOS.items())))

    # ----- margin tuning
    mt = R.margins
    setting_name = {"main": "main settings", "shared_ic50": f"IC50 {SHARED_POP.ic50:g} mg/L"}
    cols = list(mt.groupby(["policy", "setting"], sort=False).groups)
    best = {c: mt.loc[mt[(mt.policy == c[0]) & (mt.setting == c[1])].mean_return.idxmax(), "margin"] for c in cols}
    mt_rows = []
    for m in sorted(mt.margin.unique(), reverse=True):
        row = [(f"{m:g}", m)]
        for c in cols:
            r = mt[(mt.policy == c[0]) & (mt.setting == c[1]) & (mt.margin == m)].iloc[0]
            used = " (used)" if np.isclose(m, best[c]) else ""
            row.append((f"{num(r.mean_return, 3)}{used}", r.mean_return))
        mt_rows.append(row)
    margin_card = card(
        "Margin tuning for MAP-Bayes and the oracle",
        "Both rules pick the cheapest regimen predicted to land below a margin times the target. Each column is "
        "the mean return on the DQN's checkpoint cohort, so every adaptive policy is tuned on the same patients. "
        "The margin with the best return was used.",
        table(["Margin"] + [f"{pol}, {setting_name.get(st, st)}" for pol, st in cols], mt_rows, sortable=False))

    # ----- assemble
    built = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
    data = {"charts": charts, "dark": DARK}
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    page = PAGE.format(
        plotly_js=PLOTLY_JS, plotly_sri=PLOTLY_SRI, repo=REPO_URL, built=built, tiles=tiles, n_main=f"{n_main:,}",
        mdp=mdp_card, policy=policy_card, paired=paired_card, frontier=frontier_card, learning=learn_card,
        ladder=ladder_card, example=example_card, misses=misses_card, finalb=finalb_card, scenarios=scen_card,
        summary=summary_card, margins=margin_card, payload=payload,
    )
    out.mkdir(parents=True, exist_ok=True)
    (out / "index.html").write_text(page, encoding="utf-8")
    assets = out / "assets"
    assets.mkdir(exist_ok=True)
    for name in ("style.css", "app.js"):
        shutil.copy2(HERE / name, assets / name)
    fig_out = out / "figures"
    fig_out.mkdir(exist_ok=True)
    for png in sorted(figs.glob("*.png")):
        shutil.copy2(png, fig_out / png.name)
    (out / ".nojekyll").write_text("")
    return out / "index.html"


PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>RL Dosing Results</title>
  <meta name="description" content="A DQN that doses a hypothetical drug on a simulated PK/PD model, compared with MAP-Bayesian dosing, the standard regimen and an oracle.">
  <meta name="color-scheme" content="light dark">
  <link rel="icon" href="data:,">
  <link rel="stylesheet" href="assets/style.css">
  <script src="{plotly_js}" integrity="{plotly_sri}" crossorigin="anonymous"></script>
  <script>if (window.Plotly) document.documentElement.classList.add("has-plotly");</script>
</head>
<body>
  <header class="site-header">
    <div class="wrap">
      <div class="topline">
        <p class="eyebrow">Reinforcement learning for dosing</p>
        <button id="theme" class="theme-btn" type="button" aria-pressed="false">Dark mode</button>
      </div>
      <h1>A DQN that doses a hypothetical drug, next to MAP-Bayesian dosing</h1>
      <p class="lede">A Deep Q-Network picks induction and maintenance doses of Drug X for virtual patients on a PK/PD
        simulator, and can stop the drug. I tested it on {n_main} patients built from covariate profiles it never saw
        in training, alongside the standard regimen, a maximum-intensity ceiling, a MAP-Bayesian dosing rule and an
        oracle. Hover over a chart for values and click the legend to hide a series.</p>
      <p class="sim-note">All data here are simulated from a made-up model. Nothing has been validated, and none of it
        is for patient care.</p>
      <nav class="tabs" aria-label="Sections">
        <a href="#overview">Overview</a><a href="#policies">Policies</a><a href="#tradeoff">Drug use</a>
        <a href="#training">Training</a><a href="#reachability">Reachability</a><a href="#patients">Patients</a>
        <a href="#misses">Misses</a><a href="#scenarios">Scenarios</a><a href="#tables">Tables</a>
      </nav>
    </div>
  </header>

  <main class="wrap">
    <section id="overview" class="panel">
      <div class="tiles">{tiles}</div>
      {mdp}
    </section>
    <section id="policies" class="panel">
      {policy}
      {paired}
    </section>
    <section id="tradeoff" class="panel">{frontier}</section>
    <section id="training" class="panel">{learning}</section>
    <section id="reachability" class="panel">{ladder}</section>
    <section id="patients" class="panel">{example}</section>
    <section id="misses" class="panel">
      {misses}
      {finalb}
    </section>
    <section id="scenarios" class="panel">{scenarios}</section>
    <section id="tables" class="panel">
      {summary}
      {margins}
    </section>
  </main>

  <footer class="wrap footer">
    <p>Simulated data only. Not for patient care.</p>
    <p>Built on {built} from the files in <code>results/</code> by <code>dashboard/build.py</code>. Interactive charts
      with plotly; the static PNGs come from matplotlib. Code and data: <a href="{repo}">{repo}</a>.</p>
  </footer>

  <script id="dash-data" type="application/json">{payload}</script>
  <script src="assets/app.js"></script>
</body>
</html>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", type=Path, default=ROOT / "results")
    ap.add_argument("--models", type=Path, default=ROOT / "models")
    ap.add_argument("--figures", type=Path, default=ROOT / "figures")
    ap.add_argument("--out", type=Path, default=ROOT / "site")
    args = ap.parse_args()
    path = build(args.results, args.models, args.figures, args.out)
    print(f"wrote {path} ({path.stat().st_size / 1024:.0f} kB)")


if __name__ == "__main__":
    main()
