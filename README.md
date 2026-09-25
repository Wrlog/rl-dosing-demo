# DQN dose optimization on a simulated PK/PD model

[![tests](https://github.com/Wrlog/rl-dosing-demo/actions/workflows/tests.yml/badge.svg)](https://github.com/Wrlog/rl-dosing-demo/actions/workflows/tests.yml)
[![dashboard](https://github.com/Wrlog/rl-dosing-demo/actions/workflows/pages.yml/badge.svg)](https://wrlog.github.io/rl-dosing-demo/)

A Deep Q-Network that picks doses for a hypothetical drug ("Drug X") on a mechanistic PK/PD simulator.
It makes an induction decision, then four maintenance decisions (dose and interval), and can stop Drug X
and move the patient to another treatment. I compare it on the same held-out virtual patients with the
standard regimen, a maximum-intensity ceiling regimen, a MAP-Bayesian dosing rule and an oracle. This
follows the general approach of my PhD work on model-informed precision dosing; the real data and
results from that work are not public.

Everything here is simulated from made-up parameters. It hasn't been validated against anything and
isn't for patient care. It's for research and teaching only.

Results dashboard: https://wrlog.github.io/rl-dosing-demo/ (interactive charts for every analysis,
built from the files in `results/`).

## Main result

1,500 virtual patients built from covariate profiles the DQN never saw in training, 95% bootstrap CIs.
The DQN row averages three training seeds.

| Policy | At target | Correct decisions | Dose rate (mg/kg/week) |
|---|---:|---:|---:|
| Standard (5 mg/kg, then every 8 weeks) | 49.8% (47.3 to 52.2) | 49.8% | 0.76 |
| MAP-Bayes | 89.7% (88.1 to 91.1) | 97.1% (96.3 to 98.0) | 1.54 |
| DQN, 3 seeds | 81.6% (79.8 to 83.4) | 89.0% (87.5 to 90.3) | 1.72 |

The DQN loses to the MAP-Bayesian comparator: 8.0 percentage points fewer patients at target on the
same patients (95% CI 6.7 to 9.3), while using 0.17 mg/kg/week more drug. It's still far better than
the standard regimen (+31.8 points at target). Most of its extra misses are reachable patients left just
above target, where the reward penalty is small. The dashboard has the learning curves, the drug cost
trade-off, reachability as IC50 variability shrinks, example patients, the miss analysis and the
flare and held-out-profile scenarios.

"Correct" means a reachable patient ends at target, or an unreachable patient is moved to another
treatment. A patient is reachable if the ceiling regimen (15 mg/kg every 4 weeks), run on the same
patient with the same random draws, gets the true biomarker below 200 at the final visit. No policy sees
reachability; it's only used for scoring.

![Grouped bars: at target, correct decisions and dose rate by policy](figures/policy_bars.png)

## Methods

The simulator is the shared Drug X model from my other demo repos: a two-compartment PK model with a 2 h
IV infusion and weight-based dosing, covariates on clearance (weight, albumin, INF, ADA), between-patient
and between-occasion variability, and an indirect-response PD model where Drug X inhibits production of
the biomarker, driven by the average concentration over each interval. The target is a true biomarker
below 200 units. PK and PD are solved exactly over each interval, and the tests check both against a
fine-grid RK4 solution. ADA can turn positive during treatment, more often when troughs are low. Flares
(between-occasion variability on the biomarker baseline) are off in the main runs and on in one
sensitivity run.

One change from the shared model: the typical IC50 is 10 mg/L instead of 3. With 3 mg/L, 98.6% of
patients reach target on the ceiling regimen and 79.7% on the standard one, so there's little to
optimize. I kept the shared value as a sensitivity run.

Virtual patients are a covariate profile plus fresh random effects. 25% of a pool of 1,200 profiles is
held out; training only uses the other 900.

The decision problem:

- Actions: an induction dose at day 0 (6 choices, 2.5 to 15 mg/kg, given at days 0, 21 and 42), then
  four maintenance decisions from day 98, each one of 24 regimens (6 doses times 4, 6, 8 or 10 weeks) or
  stop. Stopping is masked at induction.
- Observation: 16 numbers built only from measurements, doses and baseline covariates. The agent never
  sees the true clearance, IC50, baseline or reachability.
- Reward after each interval: `−min(max(log2(B / 200), 0), 3) − λ × dose rate`, with λ = 0.05. Stopping
  ends the episode at −0.35 per decision left, which pays off once the biomarker is expected to stay
  above about 255 units. Discount 0.97.

The policies:

- Standard: 5 mg/kg at days 0, 21 and 42, then every 8 weeks. Ceiling: 15 mg/kg, then every 4 weeks.
- MAP-Bayes: estimates the patient's parameters from the measurements so far (PK first, then PD),
  predicts the next biomarker under every regimen and takes the cheapest one predicted to land below a
  margin × 200, or stops if even the strongest regimen can't get there. The margin (0.7) was tuned on the
  DQN's checkpoint cohort.
- Oracle: the same rule with the true parameters, margin 0.9. It's myopic, so it isn't the true optimum.
- DQN: a shared trunk (128 and 64 ReLU units) with separate heads for induction and maintenance, double
  DQN targets, Huber loss, uniform replay, epsilon-greedy over allowed actions, 16,000 episodes. The
  checkpoint with the best mean return on a fixed cohort of training-profile patients is kept. Three
  seeds, plus single-seed runs for the drug cost sweep, flares and the shared IC50.

## Limitations

- The DQN only sees the latest two biomarker values, the baseline and one trough. MAP-Bayes uses every
  measurement through a model with the right structure and the true population priors, which is a big
  advantage. A recurrent network or a longer history might close some of the gap. I didn't try that.
- The typical IC50 differs from the shared Drug X model (10 against 3 mg/L), as explained above.
- The reward is my own choice, and the drug cost sweep shows the policy moves a lot when it changes. The
  shallow penalty just above target explains most of the DQN's misses.
- MAP-Bayes and the oracle are myopic: they look one interval ahead, plus the stop check.
- Everything is trained and tested on patients from the same simulator. With real data this would have
  to be offline RL from logged decisions, with safety constraints and off-policy evaluation.
- The flare-trained and shared-IC50 DQNs are one seed each.

## Running it

Needs Python 3.12. On Windows use `.venv\Scripts\activate`.

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install -e .

pytest -q                           # 1 to 2 min
python scripts/run_all.py           # train, evaluate, analyses, figures
python scripts/run_all.py --quick   # small version (2,000 episodes, 300 test patients), about 3 min
python -m dashboard.build           # writes the dashboard to site/ from results/
```

`run_all.py` runs `train.py`, `evaluate.py`, `analyses.py` and `make_figures.py` in turn. The full run
took 8 min 16 s on a 14-thread laptop CPU (Intel Core Ultra 7 165U). Trained networks and all result
CSVs are in the repo, so the evaluation, figures and dashboard also run without retraining. The
dashboard build only needs numpy, pandas and plotly (`pip install -r dashboard/requirements.txt`); to
look at it locally, run `python -m http.server -d site` and open http://localhost:8000. GitHub Actions
rebuilds and publishes it on every push to main. Results were produced with Python 3.12.10, PyTorch
2.14.0 (CPU), NumPy 2.5.3 and pandas 3.0.6 on Windows. Training is deterministic for a given seed on one
machine but may shift slightly across versions or platforms.

## Layout

```
src/rl_dosing/    model (Drug X, patients, exact PK/PD), env, policies, dqn, metrics, experiment setup
scripts/          train.py, evaluate.py, analyses.py, make_figures.py, run_all.py
dashboard/        build.py (plotly figures and the page), app.js, style.css
tests/            PK and PD against RK4, biomarker timing, masks, stop payoff, observation contents,
                  MAP recovery, head separation, reproducible training, dashboard build
models/           trained networks and their settings
results/          learning curves, per-patient tables, summaries, frontier, ladder, misses (CSV)
figures/          static figures drawn with matplotlib from results/
```

## Related reading

Irie K, Tan WR, Mizuno T. Towards reinforcement learning-enabled model-informed precision dosing:
concepts, applications, and implementation. Ther Drug Monit. 2026. In press.

## License

MIT. See [LICENSE](LICENSE).
