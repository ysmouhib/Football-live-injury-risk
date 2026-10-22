# 00: Overview and product specification

## 1. One-sentence purpose

During a football match, show bench staff each player's injury risk for the **rest of the match**,
relative to **his own normal**, with the **reason** in plain words and the **age and quality** of
the data behind it - so that a physio can take a look, or staff can discuss options, before a
problem grows.

## 2. Users

| User | Device | Sees | Decides |
| --- | --- | --- | --- |
| Performance analyst (primary operator) | Tablet on the bench | Everything | Nothing alone; relays to coach and doctor |
| Club doctor / physio | Tablet or watch | Everything, incl. medical context | Medical matters |
| Head coach | Verbal relay, optional watch haptic for *Discuss* only | Summary only | Tactical changes and substitutions |

## 3. Three functions, kept separate

1. **Pre-match risk** - a calibrated probability of a non-contact time-loss injury *in this match*,
   derived from a weekly risk model (injury history, playing time, congestion, recent load). The
   weekly model already exists outside this repo; we consume its output (see contracts §5).
2. **Live risk update** - every 60 s, the pre-match risk is shaped by the time point in the match
   and scaled by a small number of live signals (high-speed load, recovery between efforts,
   fatigue signs, disruptive events), each relative to the player's own normal. Outputs: risk in
   the next 15 minutes, risk for the rest of the match, relative risk vs. his normal, intervals.
3. **Possible-injury detection** (optional, off by default) - flags within seconds the movement
   signature of a player pulling up after a sprint so medical staff look. This is *detection* of
   something that may already have happened. It is never presented as prediction.

## 4. Numbers that set expectations (and constrain the design)

| Quantity | Value | Source / calculation |
| --- | --- | --- |
| Elite men's match injury rate (all time-loss) | 21.1 per 1,000 match hours | Bengtsson et al., BMJ Open SEM 2025 |
| Muscle injuries in matches | 10.5 per 1,000 match hours | same |
| Hamstring injuries in matches (LaLiga) | 3.34 per 1,000 match hours | Moreno-Perez et al., Biol Sport 2024 |
| Any time-loss injury in a full match, per player | ≈ 3.2 % | 21.1 × 1.5 / 1000 |
| Muscle injury in a 15-minute spell, per player | ≈ 0.26 % | 10.5 × 0.25 / 1000 |
| Muscle injuries per team per match | ≈ 0.17 (one every ~6 matches) | 10.5 × 11 × 1.5 / 1000 |
| Muscle injuries per 20-team league-season (matches) | ≈ 130 | 12,540 player-hours × 10.5 / 1000 |

Consequence: even a good live model produces mostly false alarms. Flagging 2 % of player-spells
and catching 30 % of muscle injuries gives a precision of about 4 % (docs/02_MATH.md §7). Alerts
are therefore tied only to **cheap actions** (a physio's look at the next stoppage), never to an
automatic substitution.

## 5. What the tool must never do

- Say or imply that a player *will* be injured.
- Recommend a substitution. It shows risk; staff decide.
- Show a risk value without its data age and quality.
- Present low risk as "safe". Most injuries happen to players rated ordinary.
- Use any data from after the decision time (look-ahead).
- Leave the club's network with player data.

## 6. Scope

**In scope (this repository):**
- Data contracts and ingest adapters for tracking data (wearables, stadium optical feed, video
  pipeline output) and match events.
- Stream feature engine, player/team norms, pre-match bridge, live hazard model (offline fit +
  online scoring), alert logic, replay simulator, evaluation harness (Gate C report), bench app.
- Synthetic data generator with known ground truth, for development and testing.

**Out of scope (for now):**
- Training computer-vision models. Video enters through the output of an external tracking
  pipeline (e.g. SoccerNet game-state baselines); this repo only does identity assignment and
  quality control on that output (M13).
- Building the weekly pre-match model (it exists elsewhere; we consume its output).
- Any real player data in the repository.

## 7. Development phases and where this repository fits

| Phase | What happens | This repo's role |
| --- | --- | --- |
| 0 Partners and approvals | Partner club, data agreements, legal | - |
| 1 Data and labels | Tracking archives, minute-level injury labels | Ingest adapters, replay simulator, label tooling |
| 2 Model development | Fit time profile and live layer; pre-registered test | Fitting code, evaluation harness, Gate C report |
| 3 Build and usability | Live system, bench app, latency and failure tests | Engine, alerts, app |
| 4 Shadow season | Runs live, staff do not see it | Shadow mode, logging |
| 5 Live pilot | Alerts in use under an agreed protocol | Same code, protocol config |

**Gate C (the main go/no-go):** the live layer must beat "pre-match risk × time profile" on unseen
seasons and clubs, with calibration (observed/expected) between 0.8 and 1.25 and the alert budget
met. If it fails, the product ships as pre-match risk plus a live display of load vs. the player's
own norm, with **no live risk claims**. The code must support both modes via config
(`engine.mode: risk | monitor_only`).

Gate C tests one primary hypothesis. Alternative models (latent fatigue states, decaying event
load, movement-pattern shift; docs/02 §14) run only as exploratory comparators, corrected for
multiple testing, and need confirmation on a later season before adoption. Shadow-mode seasons
also provide the training data for later model versions, free of the tool's own influence on
staff decisions.

## 8. Non-functional requirements

- Runs on one local machine (laptop or small server, optional single GPU for video). No cloud.
- End-to-end latency from data arrival to screen < 30 s for 99 % of updates (excluding TV feeds).
- Feature update every 10 s; risk update every 60 s.
- Handles 30 tracked players at 10–25 Hz on a laptop CPU.
- All inputs, outputs, alerts and decisions logged append-only (Parquet/DuckDB) for audit,
  replay and retraining.
- Role-based views (coach summary vs. medical detail).
