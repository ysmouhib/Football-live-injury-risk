# 02: The model, exactly

This document is the source of truth for every formula. The reference implementation is
`src/livrisk/riskmath.py`; its tests (`tests/test_riskmath.py`) check the reference values in §12.
Do not re-implement these formulas elsewhere - import them.

## 0. Notation and units

| Symbol | Meaning | Unit |
| --- | --- | --- |
| i | player | - |
| t | match time, minutes since kick-off of the match (first half 0–45+, second half 45–90+) | min |
| period | 1, 2 (regular), 3, 4 (extra time) | - |
| Δ | model interval length | 1 min |
| p₇,ᵢ | calibrated probability of a non-contact injury costing ≥ 7 days in the next 7 days (weekly model output) | probability |
| a | share of weekly hazard that falls in one match | dimensionless |
| Λᵢ | cumulative hazard of a match-injury over 90 regular minutes, at the player's usual live state | dimensionless |
| g(t) | time profile, piecewise constant on 8 bands, mean 1 over the 6 regular bands | dimensionless |
| zᵢ(t) | vector of live features (standardised, player-relative) available at t | SD units |
| ρₖ | measured reliability of feature k (from validation, Study 1) | 0–1 |
| β | live coefficients (log hazard ratio per SD) | - |
| δ | blanking gap: features at t use only data with timestamp ≤ t − δ | s (default 60) |

Event: a **non-contact time-loss injury** (≥ 7 days missed in the main analysis) whose onset is
in the match. Contact injuries are a competing cause (§6).

## 1. From weekly risk to match risk

The weekly model gives p₇. Using match and training incidence (21.1 vs 3.5 per 1,000 h,
Bengtsson 2025) with one 90-minute match and ≈ 5 training hours per week, ≈ 60–65 % of weekly
hazard falls in the match. Hence the prior a₀ = 0.6, re-estimated through α in §4.

```
Λᵢ = a₀ · ( −ln(1 − p₇,ᵢ) )                         # cumulative hazard, 90 regular minutes
```

Weeks with two matches: use a₀ = 0.6 per match initially; a separate α for two-match weeks is
estimated when ≥ 50 events are available in that stratum (else shared).

Implementation: `weekly_to_match_cumhaz(p7, match_share)`; use `log1p`.

## 2. Baseline hazard within the match

Bands (index b):

| b | period | minutes | regular? |
| --- | --- | --- | --- |
| 0 | 1 | [0, 15) | yes |
| 1 | 1 | [15, 30) | yes |
| 2 | 1 | [30, 45) | yes |
| 3 | 1 | ≥ 45 (first-half stoppage) | no |
| 4 | 2 | [45, 60) | yes |
| 5 | 2 | [60, 75) | yes |
| 6 | 2 | [75, 90) | yes |
| 7 | 2 | ≥ 90 (second-half stoppage); also extra time (periods 3, 4) | no |

```
g(t) = exp(θ_b(t))      with  (1/6) Σ_{b regular} exp(θ_b) = 1     (normalisation)
λ₀,ᵢ(t) = (Λᵢ / 90) · g(t)                                           # per minute
```

Optional congestion interaction (config `model.congestion_interaction: true`): for the end-of-half
bands b ∈ {2, 3, 6, 7}, `θ_b → θ_b + η · c_m`, where c_m = 1 if the match is the 2nd of two
matches within 3 days or the 3rd of three with ≤ 4-day gaps (Carling 2016 definitions).

## 3. Live multiplier

Features (exact definitions in §10) are transformed and standardised against the player's own
norm for the same band:

```
x' = log1p(x)                                   # for non-negative, skewed quantities
z  = (x' − m_i,b) / s_i,b                        # player-relative z-score
z  = clip(z, −4, 4)
z* = ρₖ · z                                      # regression-calibration shrink for measurement error
RRᵢ(t) = exp(β · z*ᵢ(t))                         # relative risk vs his usual state
λᵢ(t)  = λ₀,ᵢ(t) · RRᵢ(t)
```

Player norms with recency weighting and empirical-Bayes shrinkage toward the position mean.
Past match j (age a_j in matches, 0 = most recent) gets weight w_j = 2^(−a_j / h), h = 10
(`norms.recency_half_life_matches`). The shrinkage count is the Kish effective number of matches,
never the number of windows:

```
n_eff  = (Σ_j w_j)² / Σ_j w_j²
m_i,b  = (n_eff · x̄^w_i,b + κ · μ_pos,b) / (n_eff + κ)          x̄^w = weighted mean
s²_i,b = (n_eff · s²^w_i,b + κ · s²_pos,b) / (n_eff + κ)          s²^w = weighted variance
```

κ = 5. The weighted variance uses reliability weights (`riskmath.weighted_mean_sd`) over
non-overlapping windows: for a 5-minute feature, 3 disjoint windows per 15-minute band per match
(`norms.spread_windows_per_band`). Overlapping 10-s snapshots are not new evidence and are never
counted.

Norm reset: after ≥ 14 days lost to injury (`days_since_return`) or > 42 days without a match, for
the next 3 matches n_eff is capped at 1, so the norm is mostly the position norm. The output
carries quality reason `norm_reset`.

Drift diagnostic (logged, never gating): after each match, the total-variation distance between
the player's speed-zone occupancy over his last 3 matches and his norm occupancy
(`markov.tv_distance`).

Norms use only matches strictly before the current match date (no look-ahead).

Team-relative variant (candidate, evaluated at Gate C): `z_team = z_i − mean_{j ≠ i, on pitch} z_j`
over the same window.

Safety bound applied in the engine (not in fitting): `RR ∈ [1/RR_max, RR_max]`, RR_max = 4.

Data quality: if a feature's window coverage < `quality.min_coverage` (0.7) or identity confidence
< `quality.min_identity` (0.8), that feature's z* is set to 0 (falls back to usual), and the output
quality is `degraded`. If no live feature is valid, output quality is `prematch_only`.

Mode `monitor_only` (Gate C failed): β ≡ 0; z values are shown as "load vs his norm" only.

## 4. Fitting (offline)

Data: one row per player per minute on the pitch (at risk). y = 1 in the minute of injury onset,
then the player leaves the risk set. Rows end at substitution, red card, match end, or a contact
injury (censoring).

Complementary log-log link - exact for a piecewise-constant hazard:

```
π_ij = 1 − exp(−λ_i(t_j) Δ)
cloglog(π_ij) = ln(−ln(1 − π_ij)) = ln(Λᵢ Δ / 90)  +  α  +  θ_b(j)  +  β · z*_ij
                └──────── offset ────────┘
```

Parameters ψ = (α, θ₀…θ₇, η?, β₁…β_K), K ≤ 6.

Identifiability: Σ_{b regular} θ_b = 0 is imposed in fitting (reparametrise θ₆ = −Σ θ_{0,1,2,4,5});
after fitting, convert to the normalisation of §2 and absorb the level into α.

Priors (independent normal):

| Parameter | Prior | Reason |
| --- | --- | --- |
| α | N(0, 0.5²) | match share within ×e^{±1} of 0.6 at 95 % |
| θ_b | N(0, σ_θ²), σ_θ = ln 3 / 1.96 ≈ 0.561 | a band at most ×3 or ÷3 at 95 % |
| η | N(0, 0.5²) | |
| β_k | N(0, τ²), τ = ln 2 / 1.96 ≈ 0.354 | a 1-SD change at most doubles or halves risk at 95 % |

Estimation: maximum a posteriori by L-BFGS on the negative log posterior with analytic gradient.
With μ = exp(linear predictor) (= λΔ):

```
log-lik_ij = y · ln(1 − e^{−μ}) − (1 − y) · μ
∂/∂η      = y · μ e^{−μ} / (1 − e^{−μ}) − (1 − y) · μ
Fisher weight W = μ² e^{−μ} / (1 − e^{−μ})        (≈ μ for small μ)
```

Uncertainty: Laplace approximation, Σ = (Xᵀ W X + P)⁻¹ with P = diag(prior precisions).

Ends of the at-risk time, and how each is treated:

| End | Treatment | Why |
| --- | --- | --- |
| Contact injury | competing cause; censors the non-contact hazard | cause-specific hazard is the estimand |
| Match end | administrative censoring | deterministic in time, not a random event |
| Substitution, red card | censoring | ignorable when it depends only on modelled signals |

Substitution biases β only when staff react to information the data do not contain (a reported
tightness, a visible limp). No model of substitution removes that bias; it can only be bounded:

- **Substitution intensity** (diagnostic): a discrete-time cloglog model of being substituted in
  minute j given the same signals and band. Substitutions are plentiful, so it is well estimated.
- **IPCW** (sensitivity): weights from that model, stabilised, truncated at the 99th percentile.
- **Tipping point** (sensitivity): impute the substituted players' remaining minutes with hazard
  ψ · λ, ψ ∈ {1, 1.5, 2, 3, 5}, refit, and report for each β_k the smallest ψ at which its 90 %
  interval includes 0.
- **Aalen-Johansen** cumulative incidence of non-contact and contact injury by minute, as a
  descriptive check of the fitted model's implied incidence.

R_rest (§5) is the risk if he stays on. Substitution intensity never enters it: the coach makes
that decision.

## 5. Online outputs (every 60 s)

With z held at its current value (stated on screen):

```
P15ᵢ(t)    = 1 − exp( − Σ_{u=t}^{t+14} λᵢ(u) Δ )
R_restᵢ(t) = 1 − exp( − Σ_{u=t}^{T_end} λᵢ(u) Δ )
R_usualᵢ(t)= same as R_rest with z* = 0
RRᵢ(t)     = exp(β · z*ᵢ(t))
```

T_end: end of second half including announced stoppage (default 90 + 3 if not announced). If the
call happens in the first half, the sum continues through half-time into the full second half.

Substitution: removing player i removes his exposure, so R_restᵢ(t) is the match-injury risk a
substitution would avoid. The substitute's own R_rest (z* = 0, his own Λ) is shown alongside.

Intervals: draw S = 1,000 parameter vectors from N(ψ̂, Σ) (fixed seed per update), compute P15
and R_rest per draw; report the 5th and 95th percentiles (90 % credible interval). When the
weekly model supplies [p7_lo, p7_hi] at level L, each draw also takes
ln Λ ~ N(ln Λ̂, σ²_Λ) with σ_Λ = (ln H(p7_hi) − ln H(p7_lo)) / (2 z_{(1+L)/2}),
H(p) = −ln(1 − p) (`riskmath.log_cumhaz_sd_from_interval`). Norm uncertainty is not included;
the screen help and the model card say so.

### 5.1 Interval audit (once per model version)

Laplace assumes a near-Gaussian posterior. The audit checks it by importance sampling, with no
extra dependency:

1. Draw S = 4,000 vectors from a multivariate Student t (5 degrees of freedom) centred at ψ̂ with
   scale Σ.
2. Log weight = log posterior (unnormalised) − log proposal density.
3. Pareto k̂ on the largest M = min(S/5, 3√S) weights (Zhang and Stephens estimator); effective
   sample size by `riskmath.importance_ess`.
4. On a battery of 200 synthetic (Λ, minute) cases, compare importance-weighted 5th and 95th
   percentiles of P15 and R_rest (`riskmath.weighted_quantile`) with the Laplace ones.

Pass: k̂ < 0.7 and every percentile within 10 % (relative). On failure the version ships the
normalised importance weights of its stored draws, and the engine reports weighted percentiles at
no extra cost per update.

All outputs carry `as_of`, `data_age_s`, `quality ∈ {good, degraded, prematch_only, stale}`.

## 6. Competing causes, labels and training data

- Main outcome: non-contact injuries ≥ 7 days. Secondary: ≥ 28 days; muscle; hamstring.
- Injury-minute label uncertainty u (from double annotation); δ ≥ u.
- Injuries with no identifiable moment are excluded from the live layer (kept in pre-match).
- Feedback loop: once staff act on alerts, flagged players are rested or substituted, and later
  data understate the risk the model saw. Training rows come only from matches where alerts were
  not shown (historical or shadow mode). Live-mode matches may be used with each alerted player
  censored at his first shown alert; the number of censored minutes is reported.

## 7. What alerts can achieve

```
PPV = r · π / x
```
r = share of injuries caught, π = base rate per player per 15-min spell (≈ 0.002625 for muscle
injuries), x = share of spells flagged. Expected alerts per team per match = x × 66
(11 players × six 15-min spells).

### 7.1 Operating point of the alert rules

The rules of docs/06 §3 (persistence, hysteresis, upgrade) turn a player's RR path into a finite
chain over (RR level, consecutive-update counters, alert state). RR levels: < 1.3, 1.3 to 1.5,
1.5 to 2.0, ≥ 2.0 (`alerts.analysis.rr_level_edges`). With a per-update transition matrix of RR
levels (estimated from replayed RR paths: synthetic in development, shadow mode later), a forward
recursion over one match (`markov.propagate`, about 95 updates) gives per player: expected Check
openings, Discuss upgrades and flips between open and cleared. Team figures multiply by the
players on the pitch. The budget effect and P(budget exhausted by minute m) use a Poisson
approximation on team openings and must agree with replay within 10 %.

There is no stationary regime: a match is short and the budget is consumed. The analysis shows
staff what each threshold choice costs in interruptions; the budget itself stays their choice.

Action threshold: an action with cost C and benefit B (if it prevents an injury) is worth taking
when risk > p* = C / (C + B). A physio's look is cheap (p* well below 1 %); a substitution is
expensive, and live risks rarely exceed a few percent - hence the tool never recommends one.

## 8. Sample size

Schoenfeld, for one standardised covariate, two-sided α, power 1 − β:
```
d = (z_{1−α/2} + z_{1−β})² / (ln HR)²         (= 7.849 / (ln HR)² at α = 0.05, power 0.8)
```
HR = 1.22 → 199 events; HR = 1.5 → 48 events. ≈ 130 muscle injuries per 20-team league-season,
so 2–3 league-seasons is the minimum for the live layer.

## 9. Calibration and scoring (evaluation)

- Spell level (15 min): y_spell = 1 if an event occurs in the spell; p_spell = 1 − exp(−Σ λ Δ).
- Observed/expected: O/E = Σ y / Σ p (per season, per club, per risk decile). Target 0.8–1.25.
- Calibration slope/intercept: fit `cloglog(p_true) = c₀ + c₁ · cloglog(p̂)` on test rows; ideal
  c₀ = 0, c₁ = 1.
- Proper scores: log loss and Brier at spell level. Ranking: PR-AUC (spell level).
- Uncertainty: player-cluster bootstrap, 1,000 resamples.
- Calibration of the weekly input may use Platt `σ(a + b·logit p)` or beta calibration
  `1 / (1 + 1 / (e^c · p^a / (1 − p)^b))`.
- Input attribution: refit M1 with ln Λ as a covariate with slope γ (prior N(1, 0.25²)) instead
  of a fixed offset. A γ interval excluding 1, or M1's O/E outside 0.8 to 1.25, attributes a
  calibration failure to the weekly input, which is recalibrated before the live layer is judged.
- Multiplicity: one primary hypothesis (M2 vs M1 log loss). Every exploratory comparison is
  reported with Holm-adjusted p-values (`riskmath.holm_adjust`) and is never adopted on the data
  that suggested it; adoption needs a pre-registered confirmation on a later season.

## 10. Feature definitions (exact)

Speeds from positions sampled at f Hz (10–25): Savitzky–Golay smoothing of x and y (window 1.0 s,
polynomial order 2), speed = ‖dp/dt‖. Speeds > 12 m/s are physically implausible → set missing and
flag. If a source provides speed directly, still apply the same smoothing window to it.

| Symbol | Name | Definition | Window (ending at t − δ) |
| --- | --- | --- | --- |
| HSR | high-speed running distance | distance while speed ≥ 5.5 m/s (19.8 km/h) | - |
| SPR | sprint distance | distance while speed ≥ 7.0 m/s (25.2 km/h) | - |
| effort | effort at threshold v | contiguous run with speed ≥ v lasting ≥ 1.0 s; gaps < 0.5 s merged | - |
| HIE | high-intensity effort | effort at 5.5 m/s | - |
| v_max,i | player's peak speed | max over his past matches of the per-match maximum of the 1-s centred rolling mean speed | past matches only |

Main candidate live features (K ≤ 6, fixed before Gate C):

| k | Feature | Raw value | Transform → z |
| --- | --- | --- | --- |
| F1 | `hsr_5m` | HSR in last 5 min (m) | log1p, z vs his norm for the band |
| F2 | `spr_5m` | SPR in last 5 min (m) | log1p, z vs norm |
| F3 | `recovery_10m` | median time between consecutive HIEs in last 10 min (s); missing if < 2 HIEs | log, z vs norm (sign learned) |
| F4 | `hir_drop_15m` | HSR in last 15 min ÷ HSR in his first 15 min of this match | log ratio, z vs his usual ratio at that band |
| F5 | `nearmax_cum` | cumulative count of efforts reaching ≥ 0.85 · v_max,i this match | log1p, z vs norm (exposure) |
| F6 | `disrupt_5m` | count of cards, goals, injury stoppages in last 5 min | standardised by population SD |

F1 to F4 stay missing until their whole window lies after the player's first sample of the
match (D-029); a missing feature falls back to the usual state.

Secondary (sensitivity analyses only): accelerations 2–4 m/s² count (5 min), team-relative
variants of F1–F4, late-match HSR after 75′. Hard accelerations > 4 m/s² are excluded.

## 11. Possible-injury detection (optional, off by default)

Rule (detection, not prediction): within 10 s after an effort ≥ 0.80 · v_max,i, speed falls below
1.5 m/s and stays below 1.0 m/s for ≥ 8 s while play continues (no stoppage event), or the
tracked player goes to ground (video only). Output: `possible_injury_event` for medical staff.
Thresholds are configurable and must be validated before use.

## 12. Reference values (tested)

| Function | Input | Expected |
| --- | --- | --- |
| `weekly_to_match_cumhaz` | p₇ = 0.0245, a = 0.6 | 0.01488307135 |
| `baseline_hazard_per_min` | Λ above, g = 1 | 1.6536746e−4 |
| `prob_from_cumhaz` | 0.0148830714 | 0.0147728659 |
| `window_probability` | 15 × 1.6536746e−4 per min | 0.0024774 (= 1 − e^{−0.00248051}) |
| `alert_ppv` | r = 0.3, π = 0.002625, x = 0.02 | 0.039375 |
| `alerts_per_match` | x = 0.02 | 1.32 |
| `schoenfeld_events` | HR = 1.22 | 199 |
| `schoenfeld_events` | HR = 1.5 | 48 |
| `prior_sd_for_multiplier` | ×2 at 95 % | 0.3536530 (exact normal quantile 1.959964) |
| `action_threshold` | C = 1, B = 99 | 0.01 |
| `shrink_mean` | n = 3, x̄ = 2.0, κ = 5, μ = 1.0 | 1.375 |
| `beta_calibrate` | a = b = 1, c = 0 | identity |
| `fisher_weight` | μ → 0 | → μ |
| `recency_weights` | ages 0, 10, 20; h = 10 | 1, 0.5, 0.25 |
| `kish_effective_n` | weights 1, 0.5, 0.25 | 2.3333 (= 1.75² / 1.3125) |
| `log_cumhaz_sd_from_interval` | p7 ∈ [0.01, 0.04], L = 0.90 | 0.42606 |
| `holm_adjust` | 0.01, 0.04, 0.03, 0.20 | 0.04, 0.09, 0.09, 0.20 |
| `markov.stationary_distribution` | two states, a = 0.1, b = 0.3 | 0.75, 0.25 |
| `markov.slem` | same | 0.6 |
| `markov.mixing_time` | same, ε = 0.25 | 3 steps |
| `markov.ar1_effective_n` | n = 15, φ = 0.8 | 2.3338 (asymptotic n(1 − φ)/(1 + φ) = 1.67) |

## 13. Markov chain tools

Implemented in `src/livrisk/markov.py` (tested in `tests/test_markov.py`).

- A finite chain with transition matrix P is **ergodic** when it is irreducible (every state
  reaches every other) and aperiodic (period 1, the gcd of its cycle lengths). It then has a
  unique stationary distribution π with πP = π, and P^t(x, ·) → π from every start x.
- **Mixing time** t_mix(ε): the smallest t with max_x TV(P^t(x, ·), π) ≤ ε, computed exactly by
  powering P. The second-largest eigenvalue modulus (SLEM) sets the geometric rate.
- **Semi-Markov occupancy**: with an embedded jump chain Q (zero diagonal) and mean dwell m_i, the
  long-run share of time in state i is q_i m_i / Σ_j q_j m_j, q the stationary law of Q. A Markov
  chain on a fixed step is the special case with geometric dwell of mean 1/(1 − P_ii).
- **Finite-horizon propagation**: x_t = x_{t−1} P_t, for time-inhomogeneous chains (§7.1).
- **AR(1) effective n**: the mean of n consecutive AR(1) values with coefficient φ carries
  n / (1 + 2 Σ_{k=1}^{n−1} (1 − k/n) φ^k) independent values.

Where ergodicity is used, and where it is not:

| Use | Ergodic? | Object |
| --- | --- | --- |
| Synthetic speed zones (M2) | yes, per band and position | π per band; mixing ≈ 13 s ≪ 900 s band |
| Norm drift (§3) | assumed across matches, checked by drift diagnostic | occupancy TV distance |
| Alert rules (§7.1) | no: short horizon, consumable budget | forward recursion |
| Injury within a match | no: injury is absorbing | discrete-time hazard (§4) |

## 14. Research track (never in the main fit)

Exploratory comparators, Holm-adjusted, each needing confirmation on a later season (§9):

| Id | Idea | Definition | Extra parameters |
| --- | --- | --- | --- |
| R1 | Latent fatigue state | 3-state hidden semi-Markov model on z, band-dependent transitions, emissions fitted offline and frozen; filtered state probabilities enter as one candidate signal | 1 to 2 against injuries |
| R2 | Decaying event load | `riskmath.decayed_event_load`, τ = 300 s fixed a priori, in place of F6's 5-minute count | 0 |
| R3 | Movement-pattern shift | KL divergence between the last 15 minutes' zone transitions and his norm matrix (Dirichlet prior κ = 5 matches toward his position matrix) | 1 |
| R4 | Exponentially weighted load | F1 and F2 with time constants 5 and 15 min instead of rectangular windows | 0 |
| R5 | Team-relative signals | F1 to F4 minus the team-mates' mean (§3) | 0 |
