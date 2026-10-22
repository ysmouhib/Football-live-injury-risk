# 05: Validation protocol

The live layer must earn its place. Validation happens in four stages; only the last two use
real data (provided by the partner club; never committed).

## 1. Synthetic validation (M2, M6, M10): the machinery works

| Test | Setup | Pass |
| --- | --- | --- |
| Parameter recovery | Generate data from the true model with known β (e.g. [0.30, 0.0, −0.20, 0.0, 0.10, 0.15]), ≈ 400 events; 50 seeds | true β inside the 90 % Laplace interval in 85–95 % of seeds per coefficient |
| Null signal | Same with β = 0 | M2 vs M1 log-loss improvement has bootstrap CI including 0 in ≥ 90 % of seeds; mean |β̂| < 0.1 |
| Calibration | Generated with known Λ, g | O/E within 0.9–1.1 on 2,000 simulated matches |
| Look-ahead guard | Inject a feature that peeks 30 s ahead | replay guard raises; evaluation refuses the run |
| Online = offline | Same match through stream engine and batch computation | identical features to 1e-9 |
| Speed process ergodic | Every band and position jump matrix | irreducible and aperiodic (`markov.is_ergodic`) |
| Zone occupancy | 200 simulated midfielder-bands | TV distance to the semi-Markov law q_i m_i / Σ q_j m_j below 0.03 per band |
| Dwell shape | Bout lengths per zone | KS p > 0.01 against min + Gamma; sprints ≥ 1 s |
| Physics and decline | All ticks; band 6 vs band 0 | speed change within accel/decel limits; HSR 10 to 35 % lower late |
| Stationary start | Minutes 0 to 2, 500 players | occupancy within TV 0.05 of band 0 law (burn-in works) |
| Interval audit | Correct fit vs covariance × 0.25 | passes (k̂ < 0.7, percentiles within 10 %) vs fails |
| Input attribution | p7 too spread (true slope 0.6) | slope interval excludes 1; verdict `input_miscalibrated` |
| Tipping point | Informative censoring on | β intervals move toward 0 as ψ grows; tipping ψ reported |
| Alert operating point | Known RR level matrix | analysis vs simulated alerts within 10 % |

## 2. Retrospective validation on partner data (Phase 2)

- **Data:** ≥ 2–3 league-seasons of minute-level tracking with injury minutes (docs/02_MATH.md §8).
- **Splits:** train on earlier seasons, test on the next; plus leave-one-club-out where possible.
  Norms and pre-match inputs recomputed inside each training fold.
- **Comparators (nested):**
  - M0: pre-match Λ spread evenly over minutes (g ≡ 1, β ≡ 0).
  - M1: M0 × fitted time profile g(t).
  - M2: M1 × live multiplier (main features F1–F6).
  - Benchmarks (from public methods): EWMA acute/chronic load features; k-nearest-neighbour
    "players like him"; threshold rules (staff-set limits).
- **Primary test:** M2 vs M1, spell level (15 min), log loss; player-cluster bootstrap 95 % CI
  must exclude 0.
- **Secondary:** PR-AUC, Brier, O/E per season and club, calibration slope/intercept, decision
  curves over p* ∈ [0.001, 0.02] per 15 min, alert metrics (alerts per team-match, share of
  injuries preceded by an alert within 15 min, precision, lead time distribution).
- **Sensitivity:** ≥ 28-day injuries; muscle only; hamstring only; IPCW for substitutions;
  tipping point for substitutions (docs/02 §4); δ = 120 s; congestion interaction.
- **Diagnostics reported with every run:** input attribution (calibration slope of ln Λ and M1's
  O/E, docs/02 §9); substitution intensity per signal; Aalen-Johansen cumulative incidence of
  non-contact and contact injury against the model's implied incidence; norm drift summary.
- **Exploratory comparators** (research track R1 to R5, docs/02 §14): Holm-adjusted, labelled not
  confirmatory, never adopted on this data. Adoption needs a new pre-registration and a later
  season.
- **Training data:** historical and shadow-mode matches only, or live-mode matches with alerted
  players censored at their first shown alert (docs/02 §6).

### Gate C (pre-registered; all must hold)

1. M2 improves log loss over M1 with CI excluding 0.
2. O/E between 0.8 and 1.25 in every test season.
3. At a budget of ≤ 1.5 alerts per team-match, share of injuries caught > share of spells flagged,
   with CI excluding equality.
4. No subgroup (position, club) with O/E outside 0.6–1.6.

If any fails → `engine.mode: monitor_only`. If criterion 2 fails and input attribution flags the
weekly input, the weekly input is recalibrated and the live layer is judged again on the next
pre-registered season, not on the same data.

## 3. Replay validation (M9): what staff would have seen

Re-run historical matches through the live code path with realistic latency (5–30 s), dropouts
(5 % of samples, bursts of 30 s), identity swaps (video only) and clock drift. Check:
- every displayed value obeys the stale rules;
- features never use data after `t − δ` (guard);
- the replayed risk equals the offline computation for the same inputs;
- alert counts per match stay within budget.

## 4. Shadow season (Phase 4)

Frozen model and config (hashes logged). Runs live in every match; staff do not see it. Scored
once at the end against the Gate C criteria, plus latency and uptime targets (< 30 s for 99 % of
updates; no stale value shown as live).

## 5. Pre-registration template (fill before touching any test data)

```
Model version / commit:
Config hash:
Data: seasons, clubs, injury definition, label source, uncertainty u, delta:
Primary comparison and metric:
Gate C thresholds:
Sensitivity analyses:
Exploratory comparators (Holm-adjusted, not confirmatory):
Training data sources and censoring of alerted players:
What will be reported whatever the outcome:
```

## 6. Model card (generated by M10)

Data, period, populations, injury definition, features, priors, fitted parameters with intervals,
metrics with CIs, calibration plots, subgroup results, known limits, intended use, forbidden uses.
