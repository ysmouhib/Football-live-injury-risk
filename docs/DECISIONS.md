# Decisions log

Append one entry per decision. Never edit past entries; supersede them with a new one.

Format:
```
## D-0xx: title (YYYY-MM-DD, milestone)
Context:
Options:
Choice:
Consequences:
```

---

## D-001: Python and package layout (2026-10-22, M0)
Context: local-first tool, scientific stack, easy to test and type-check.
Choice: Python ≥ 3.11, `src/livrisk`, pydantic v2 contracts, pytest + hypothesis, ruff + mypy strict.

## D-002: One-minute discrete-time hazard with cloglog link (2026-10-22, M0)
Context: rare events, piecewise-constant hazard, need exact link between minute probability and
hazard. Choice: 1-minute intervals; cloglog link with pre-match offset (docs/02_MATH.md §4).
Consequence: only K ≤ 6 live coefficients + 8 band levels are estimated from in-match data.

## D-003: Priors centred on "no effect" (2026-10-22, M0)
Choice: β ~ N(0, 0.354²) (a 1-SD change at most doubles risk at 95 %), θ ~ N(0, 0.561²).
Consequence: with weak evidence the tool reports pre-match × time profile.

## D-004: Blanking gap δ = 60 s (2026-10-22, M0)
Context: the injuring sprint (median ≈ 17 m, Aiello 2023) and label uncertainty.
Choice: δ = 60 s default; must be ≥ label uncertainty u; sensitivity at 120 s.

## D-005: Speed thresholds (2026-10-22, M0)
Choice: HSR ≥ 5.5 m/s (19.8 km/h), sprint ≥ 7.0 m/s (25.2 km/h); efforts ≥ 1 s, gaps < 0.5 s
merged. Player-relative near-max at 0.85 · v_max.

## D-006: Coordinate system and clock (2026-10-22, M0)
Choice: metres, origin at centre spot; continuous match clock with period starts 0/2700/5400/6300 s.

## D-007: Alert design (2026-10-22, M0)
Choice: two levels (Check, Discuss), persistence 2/3 updates, hysteresis at RR 1.3, budget 2 per
team per match, cheap agreed actions only. Rationale: alert fatigue evidence (docs/01 §5).

## D-008: Weekly model consumed as output only (2026-10-22, M0)
Context: ysmouhib/injury-ml-research has no licence file.
Choice: read its exported weekly risk CSV (docs/03 §5); do not copy code.

## D-009: Synthetic speed process: band-dependent, semi-Markov, ergodic (2026-10-22, M2)
Context: the full-path generator named a speed Markov model without a matrix, dwell times or a
start rule. Effort and recovery features depend on bout structure, and plain Markov chains force
geometric bout lengths.
Options: hand-tuned Markov chain; semi-Markov chain with explicit dwell times; calibrated from
broadcast tracking.
Choice: six-zone semi-Markov process; jump matrix and gamma dwell times in `configs/default.yaml`;
band factors for within-match decline and position factors for high-speed entries; acceleration
limits 3.5 and 4.5 m/s²; 300 s burn-in. Every band and position matrix must be ergodic. Defaults
sit inside the published sanity ranges; SkillCorner open data informs transition structure only,
because broadcast tracking misses off-camera running.
Consequences: acceptance tests compare simulated occupancy with the closed-form semi-Markov law
per band, and dwell shapes with the configured distributions. The fast path used for model
recovery tests is unaffected.

## D-010: Norms with recency weights, Kish effective n and a reset rule (2026-10-22, M4)
Context: norms assumed past matches exchangeable and gave no answer to returns from injury or
fitness changes.
Options: unweighted norms; drift tests that gate the norm; recency weights with a rule-based reset.
Choice: weights 2^(−age/10) by match; shrinkage against the Kish effective number of matches;
spread from three non-overlapping windows per band and match; reset after ≥ 14 days lost or
> 42 days without a match, capping his own norm at one effective match for three matches. A drift
diagnostic is logged but does not gate.
Rejected: adding window autocorrelation to ρ. ρ measures device and identity error; the window
value is the exposure itself, and its spread is already estimated across matches.
Consequences: new contract fields `days_since_return` and `quality_reasons`.

## D-011: Alert operating point by finite-horizon analysis (2026-10-22, M8)
Context: the budget of 2 per team-match was set without a link to thresholds and hysteresis.
Options: stationary analysis of the alert chain; finite-horizon recursion; simulation only.
Choice: forward recursion over (RR level, counters, alert state) for one match, with a Poisson
approximation for the team budget, checked against replay within 10 %. A stationary analysis does
not apply: a match is about 95 updates and the budget is consumed.
Consequences: the analyst screen shows the expected interruptions for each setting; the budget
stays a staff choice, default 2.

## D-012: Interval audit by importance sampling; pre-match interval propagation (2026-10-22, M6, M7)
Context: Laplace intervals were not checked against the exact posterior, and the displayed
intervals silently ignored uncertainty in p7.
Options: MCMC with a new optional dependency; importance sampling from the Laplace fit.
Choice: Student-t proposal (5 df, 4,000 draws), Pareto k̂ < 0.7, percentiles of P15 and R_rest
within 10 % on 200 synthetic cases; on failure ship importance weights for the stored draws. When
the weekly model supplies [p7_lo, p7_hi], draws vary ln Λ with the implied SD.
Consequences: no new dependency; per-update cost unchanged; the help text states what the
interval includes.

## D-013: Competing events: substitution as diagnostic and tipping point (2026-10-22, M10)
Context: substitution censoring may be informative; a multi-state model was proposed.
Options: censoring with IPCW only; full multi-state model with substitution in the displayed risk;
diagnostics plus tipping-point sensitivity.
Choice: keep the cause-specific hazard; fit a substitution-intensity model as a diagnostic; report
IPCW and a tipping-point analysis; report Aalen-Johansen cumulative incidence. Match end is
administrative censoring. R_rest stays the risk if he stays on.
Rejected: fundamental-matrix and quasi-stationary calculations (they need a long homogeneous
chain); any substitution term in R_rest (the coach decides substitutions).

## D-014: One primary hypothesis; exploratory comparators need confirmation (2026-10-22, M10)
Context: several alternative models were proposed as Gate C comparators, which multiplies the
chances of a false positive.
Choice: the primary test is M2 vs M1 log loss. Research-track models R1 to R5 (docs/02 §14) are
reported with Holm-adjusted p-values, labelled not confirmatory, and adopted only after a new
pre-registration on a later season.

## D-015: Training data free of the tool's own influence (2026-10-22, M10)
Context: once staff act on alerts, flagged players are rested or substituted, and later data
understate the risk the model saw.
Choice: train on historical and shadow-mode matches; live-mode matches only with alerted players
censored at their first shown alert; report the censored minutes.
Consequences: every shown alert and staff action must be logged (already required).

## D-016: Feature engine works on a play clock (2026-10-22, M3)
Context: the match clock `t_s` (docs/03 §1) is ambiguous across a period break: first-half
stoppage runs past 2700 s, where the second half restarts, so a bare `t_s` does not order samples
across periods, and the blanking rule (`data <= t - delta`) needs a strictly real-time ordering.
Options: carry (period, t_s) pairs everywhere; order by arrival time; a derived play clock.
Choice: `features/stream.py` works on the play clock `u` (seconds of play since kick-off, breaks
removed; the synthetic generator's internal convention). `as_of_t_s` and `cutoff_t_s` in
`FeatureSnapshot` carry play-clock seconds; `period` labels the period at `as_of`. Period lengths
come from `period_end` events seen so far; until one arrives the standard length is assumed, which
only affects the cosmetic `period` label during stoppage.
Consequences: the no-look-ahead rule is a plain comparison on `u`; the ambiguity at the break is
documented in docs/03 §7.

## D-017: Reset state reaches the norm table at build time (2026-10-22, M4)
Context: the norm reset rule (docs/02_MATH.md §3) needs per-player inputs the history frame does
not carry: `days_since_return` and `days_since_last_match` come from the weekly input
(PreMatchRisk fields, docs/03 §5), and `matches_since_return` is a count the caller maintains.
Options: derive the reset state inside `build_norms` from the history dates; pass a per-player
mapping to `build_norms`; pass it per query to `NormTable.get` / `z_for`.
Choice: an optional `reset_inputs` mapping of `ResetInputs` on `build_norms`. The built
`NormTable` is then self-contained: `get` and `z_for` need no extra arguments, and the
position-norm fallback of a player in reset still carries `reset=True`, so the quality reason
`norm_reset` reaches the snapshot whatever his history holds.
Consequences: `build_norms` grows one optional parameter; deriving the state from history was
rejected because a windowed history cannot see the absence or matches outside the frame.

