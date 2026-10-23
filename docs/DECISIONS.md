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

## D-018: Risk engine interface: the caller owns the clocks and the data age (2026-10-22, M7)
Context: `FeatureSnapshot` times are play-clock seconds (D-016), `RiskSnapshot` times are match
clock, and `data_age_s` is wall-clock based (docs/04 §4). Someone must bridge the clocks, and
the engine also needs the announced stoppage for its projection (docs/02 §5).
Options: the engine keeps its own play/match clock from the event stream; the caller passes both
clock readings; the caller passes the match clock, the announced stoppage and a per-player data
age.
Choice: `update(snapshots, state, data_age_s)` takes a `MatchState` (match clock plus announced
stoppage, None until announced) and an explicit per-player `data_age_s`. The risk task owns the
clocks, as it already drives the feature engine; the engine never mixes them. The engine
resolves z from the raw features via the norm table (the feature engine stays norm-free) and
returns a `RiskUpdate`: the risk rows plus the consumed snapshots enriched with z and the norm
quality reasons (`norm_reset`, `few_matches`), so the app's quality chip has its reason
(docs/06 §2).
Consequences: replay and live wiring compute the data age where the wall clock exists; the
engine is deterministic in (snapshots, state, ages); `FeatureSnapshot.z` is filled at the 60 s
cadence by the engine, not by the feature engine.

## D-019: Engine quality and reason-code rules (2026-10-22, M7)
Context: docs/02 §3 sets the coverage and identity gates but leaves some edges open: the scope
of the identity gate, the meaning of `few_matches`, the stale fallback and how the wording
table's downside code (LOAD_BELOW) is reached.
Choice: identity confidence gates only the tracking-derived features
(`stream.TRACKING_FEATURES`); event-feed features (disrupt_5m) do not depend on video identity.
A stale row (data age above `quality.stale_hide_s`) zeroes every z* and reports quality `stale`,
so a stale value never looks live. Norm reasons (`norm_reset`, `few_matches`) ride on the
feature snapshot without changing the quality level; `few_matches` fires when the player's own
norm carries fewer effective matches than `norms.min_matches_for_own_norm`. The top reason code
follows docs/06 §5: the largest positive contribution beta_k * z*_k above 0.1 on the log scale
wins; without one, an elevated time profile of the current band (ln g > 0.1) gives
TIME_PROFILE, a dominant negative contribution of a load feature (hsr_5m, spr_5m, nearmax_cum)
gives LOAD_BELOW, else WITHIN_NORMAL; prematch_only and stale rows report DATA_LOST. Posterior
draws are not bounded by rr_max (the bound guards the headline RR only); reported intervals are
widened to contain the point estimate, which the RiskSnapshot contract requires.
Consequences: an identity failure falls back toward the pre-match view for the tracking
features while the event-feed signal survives; the interval always satisfies the contract.


## D-020: Alert rule semantics left open by the spec table (2026-10-22, M8)
Context: docs/06 §3 fixes the alert rules but leaves some edges open: which updates count
toward persistence, whether the R_rest floor and the quality gate apply to a direct Discuss
opening, what the "2 updates" of the clear rule refer to, how ties break when the budget is
short, and how the staff flag "concern" reaches the state machine through the existing
`DecisionRecord` contract (its Action enum has no concern member).
Options: count every update with RR above threshold toward a streak vs only updates with live
quality; gate only Check openings vs any opening; a separate clear-persistence knob; arbitrary
tie-break order; a new Action member for the concern flag.
Choice: a streak counts consecutive updates with RR at or above the threshold AND quality good
or degraded (a stale or prematch_only update reports a fallback RR, docs/02 §3, so it cannot be
evidence and restarts the streak). The R_rest floor and the quality gate apply to any opening,
Check or Discuss; an upgrade from an open Check is not re-gated (it passed at opening) and
consumes no budget. The floor gates the opening update, not the streak. The clear rule uses
`alerts.check.persist_updates` (no separate knob). The budget counts openings per team-match; a
clearing does not refund a slot. Priority among players qualifying in the same update: higher
R_rest first, ties broken by player_id ascending (deterministic). The concern flag is a
`DecisionRecord` with action `other` and the word "concern" in the note on an open Check. In
`alerts.analysis`, `expected_checks` counts openings at either level (each opening is one card
and one budget slot) so the Poisson budget approximation uses the same number.
Consequences: the on-screen behaviour is fully determined by `alerts.*` config values; the
operating-point chain (docs/02 §7.1) mirrors the state machine exactly, which is what makes the
10 % replay agreement meaningful.

## D-021: Data age is the age of the newest sample received (2026-10-23, M9)
Context: docs/04 §4 defined `data_age_s` from the newest sample used. With the blanking gap
δ = 60 s the newest used sample is always at least δ old, so every value would sit above the
60 s grey threshold of docs/06 §4 in normal operation, against the 8 s example of docs/06 §2 and
§8.
Options: age of the newest sample used (always ≥ δ); the same minus δ; age of the newest sample
received.
Choice: `data_age_s` is the wall-clock age of the newest sample received for the player (its
production `wall_time_ns`), or the time since he came on before his first sample. The blanking
gap is a fixed design choice, stated by `cutoff_t_s`, and never counts as age. The screen gets
the age refreshed at every push (every feature tick) and greys or hides values on it.
Consequences: in normal operation the age is about the source latency. A feed failure greys
values 60 s after the last sample and hides them after 180 s, earlier than a used-sample rule
would. Right after half-time, values stay hidden until second-half data has arrived, because the
newest data is from before the break. docs/04 §4 updated.

## D-022: Replay on a simulated clock, guard on buffer reads, log store in part files (2026-10-23, M9)
Context: the replay must drive the live components under latency and dropouts, prove that no
read crosses t − δ, and write everything to an append-only store, without a second
implementation of the live path.
Options: an asyncio replay with a virtual clock; a deterministic tick loop over the same
components. For the guard: checking only the snapshots that come out; checking every buffer read.
For the store: one Parquet file per kind kept open; one new file per flush.
Choice: `replay/simulator.py` runs the live `StreamFeatureEngine`, `RiskEngine` and `AlertLogic`
in a deterministic tick loop on the play clock. The wall clock is simulated from kick-off with a
15-minute half-time break; samples and events arrive after a fixed latency, an event before
samples arriving at the same instant; the match clock given to the risk engine comes only from
delivered events. `StreamFeatureEngine` takes an optional `read_guard` that the sample buffers
call before every read, so a leaky feature stage is caught at the buffer, and the snapshots are
checked as well. `AlertLogic` takes an `id_prefix`; the replay uses the team id, which keeps
alert ids unique within a match. The log store writes
`{root}/{date}/{match_id}/{kind}/part-NNNNN.parquet`, one new exclusively created file per kind
and flush, with flat columns and the config hash on every row: Parquet cannot be appended in
place, and a crash must never leave an unreadable file.
Consequences: replay output equals a direct offline computation on the delivered stream, which
the M9 tests check. Pacing (`speed`) only adds sleeps. docs/03 §9 shows the part-file layout.
Losses at the source and at the ingest queue are counted in every replay result.

## D-023: Spell-level scoring and the operational form of the Gate C criteria (2026-10-23, M10)
Context: docs/05 §2 fixes the criteria but not how spells, flags and intervals are formed, and a
plain sum of at-risk hazards per spell is biased: a player's rows stop at his injury, so the
spell holding an injury would get a lower probability because the injury happened.
Options: score at-risk minutes only; drop spells with any truncation; give the onset spell its
planned exposure.
Choice: spells are 15-minute blocks of a player-match counted from each period's start, with
stoppage minutes as their own spell. A spell's probability sums the observed minutes' hazards
plus, in a spell that ends with the onset, the rest of the planned spell at the onset minute's
hazard; exposure that stops at a substitution or a contact injury stays as observed
(censoring). Criterion 3 flags the top 1.5 / 66 of test spells by the M2 / M1 spell ratio (the
live layer's relative risk) and tests caught share minus flagged share with a paired player
cluster bootstrap. The thresholds (O/E 0.8-1.25, subgroups 0.6-1.6, 1.5 alerts) are constants
in `evaluation/criteria.py`: they are pre-registered, not tunable. `build_design` accepts K = 0
for the M1 comparator.
Consequences: O/E and calibration slope are unbiased on correctly specified synthetic data
(slope 1.00 in the M10 tests). Benchmarks: EWMA acute minus chronic load of F1 and F2
(5 and 15 min), kNN over spell-mean features (200 neighbours, one pseudo-event of smoothing),
and one fitted multiplier above a 2 SD limit. Of the exploratory ids, R4 (`ewma_features`) and
R5 (`team_relative`, needs `team_id`) run on person-minute tables; R1-R3 are reported as not run
until their inputs exist.

## D-024: Conventions of the Gate C diagnostics (2026-10-23, M10)
Context: docs/02 §4 and §9 name the diagnostics but leave their exact form open.
Choice: input attribution enters centred ln(Lambda) as a covariate on top of the offset, so its
coefficient is the slope minus 1 with prior N(0, input_slope_prior_sd^2); the verdict is
`input_miscalibrated` when the 95 % slope interval excludes 1 or M1's O/E leaves 0.8-1.25.
Substitution intensity is fitted on the bands that saw a substitution, without the pre-match
offset and with weak priors (SD 10). IPCW weights are stabilised by the band-only substitution
model and truncated at the 99th percentile. The tipping point imputes the minutes after each
substitution, to the end of regular time, at psi times the fitted hazard of his usual state
(z = 0, the live state after he left being unknown): one seeded imputation per psi, 90 %
intervals. Training-data policy: live-mode rows always go through the censoring at the first
shown alert, even when listed as a training source; rows of unknown sources are excluded and
counted.
Consequences: on synthetic data the substitution model recovers the generator's 0.5 log hazard
ratio on hsr_5m, null effects are erased at psi = 1, and strong effects shrink monotonically
with psi without always being erased by psi = 5 ("not reached" in the report).

## D-025: The bench app serves WebSockets through an optional dependency (2026-10-23, M11)
Context: docs/06 §8 specifies a WebSocket push channel for the live table. uvicorn only offers
the WebSocket protocol when a server-side implementation is installed; the standard library has
none.
Options: vendor a minimal WebSocket implementation; make the page poll the REST snapshot;
add `websockets` as an optional dependency.
Choice: `websockets` under the `serve` extra in pyproject.toml. The app pushes on `/ws/live`
when it is installed and the page falls back to polling the REST snapshot when it is not, so the
core engine and the tests never import it.
Consequences: the dependency list of the core package is unchanged; a bench machine that serves
the app installs the `serve` extra.

## D-026: Ingest adapter conventions (2026-10-23, M12)
Context: vendor tracking and event feeds arrive as CSV or Parquet with vendor column names, and
the SkillCorner open data ships its own layout (match.json plus a tracking JSONL at 10 fps with
no event stream). The adapters must turn both into the contracts of docs/03 without weakening
the fail-safe rule of docs/04 §6.
Options: pass column maps as vendor -> contract or contract -> vendor; coerce out-of-range rows
silently or reject the file; reconstruct SkillCorner events from the metadata or leave the event
list empty.
Choice: column maps are contract field -> vendor column, so an unmapped optional field simply
keeps its contract default (`detected` True, `identity_confidence` 1.0, wall/ingest stamps
missing). A file that violates the ranges of docs/03 §2 raises ValueError; nothing is coerced.
SkillCorner events are reconstructed from match.json: kick-off and period_end from the period
frame windows, a substitution pairs a non-starter's first tracked frame with the same-team
leaver whose last tracked frame immediately precedes it (30-frame tolerance), a red card is
timed at the carded player's last tracked frame when no substitution consumed it, and goals and
untimed cards are omitted because they cannot be placed on the clock. Position groups come from
the role acronym, then the group label, then MF. Clock drift is flagged by `ingest.clock.ClockSync`
at the fixed 2 s tolerance of docs/04 §4 and stays degraded until the next sync.
Consequences: every adapter output validates against the contracts and replays through the M9
simulator unchanged; feeds with more detail (goals, cards) need an official-feed event table
alongside the tracking, which `read_events_table` already accepts.

## D-027: Video identity assignment: windowed votes, hysteresis, fail-safe gaps (2026-10-23, M13)
Context: the video track (docs/01 §4) consumes per-frame tracks from an external pipeline with
noisy jersey reads (a few percent wrong per frame) and tracker identity swaps at player
crossings. Downstream, a FeatureSnapshot takes the minimum sample identity confidence over 15
minutes and gates the tracking features at `quality.min_identity` (D-019), so a single
low-confidence sample poisons a quarter hour and a one-window guess would be expensive.
Options: per-window majority vote with immediate switching; one long-memory vote over the whole
match; per-window Hungarian assignment with confirmation hysteresis and a separate confidence
memory; a probabilistic multi-hypothesis tracker (heavier, research track).
Choice: per-window Hungarian assignment (`scipy.optimize.linear_sum_assignment`) of the visible
tracks to the on-pitch lineup players, with the on-pitch set taken from substitution and
red-card events at their event time. The cost is `1 -` the window's jersey vote share for the
player's shirt (0.5 with no reads) plus half the folded distance `d / (d + 5 m)` to the player's
last seen position. A track's identity changes (or starts being emitted) only after the same
player wins its assignment for 2 consecutive windows, so a swap flag lands at most
`window_s * (confirm_windows + 1)` seconds (3 s at the 1 s default) after the physical swap.
Unconfirmed and contested windows emit nothing: a doubtful stretch is a coverage gap, never a
guessed identity. `identity_confidence` is the share of the last 30 s of reads supporting the
emitted identity, shrunk by a prior of 4 reads at 2 % error and cleared on a confirmed switch;
without any reads it falls back to a continuity score capped at 0.75, below the identity gate,
so a read outage surfaces as `identity_low`. The tunables are parameters of
`assign_identities`, documented in docs/VIDEO_INPUT.md, not config keys: the video path is an
offline post-processing step, not part of the live engine's match-day configuration.
Consequences: on synthetic tracks with 2 % misreads and injected swaps, every emitted sample
carries the right identity and swaps are flagged on both tracks within about 2 s. Team guesses
are hard constraints (a wrong guess leaves the track unassigned rather than misassigned), and
referees or other non-lineup tracks must be filtered by the adapter before the call.
