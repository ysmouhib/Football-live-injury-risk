# 04: Architecture

## 1. Data flow

```
 sources                      ingest/                 features/            model/ + engine/          alerts/          app/
 ───────                      ───────                 ─────────            ────────────────          ───────          ────
 wearables ─┐
 optical  ──┼─► adapters ─► TrackingSample stream ─► StreamFeatureEngine ─► RiskEngine (60 s) ─► AlertLogic ─► WebSocket ─► tablet
 video out ─┘   (M12/M13)   + MatchEvent stream       (10 s, M3)            pre-match bridge (M5)      (M8)        (M11)
 events ──────► adapter ──►                           norms (M4) ─────────► live hazard (M6, fitted offline)
                                                                                     │
                                     every object ─────────────────────────────► append-only LogStore (Parquet/DuckDB)
                                                                                     │
                                                                  replay/ (M9) re-runs logs or synthetic matches
                                                                  evaluation/ (M10) fits, tests, Gate C report
```

All arrows carry the contracts of docs/03_DATA_CONTRACTS.md only.

## 2. Modules and their single responsibility

| Module | Responsibility | Pure? |
| --- | --- | --- |
| `riskmath.py` | Formulas of docs/02_MATH.md | yes |
| `markov.py` | Finite Markov chain tools of docs/02 §13 (ergodicity, stationary law, mixing, occupancy) | yes |
| `contracts.py` | Data shapes and validation | yes |
| `config.py` | Load/validate `configs/*.yaml` into typed objects | yes |
| `synth/generator.py` | Synthetic matches with ground-truth hazard | yes (seeded) |
| `features/stream.py` | Online features with `as_of` and coverage; identical to offline batch computation | stateful, deterministic |
| `features/norms.py` | Player/position/team norms with shrinkage; built from past matches only | yes |
| `model/prematch.py` | p7 → Λ, bands, expected minutes | yes |
| `model/live_hazard.py` | Build person-minute design matrix; MAP + Laplace fit; save/load fitted params | yes |
| `engine/risk_engine.py` | Every 60 s: features + params → `RiskSnapshot` with intervals and quality | stateful |
| `alerts/logic.py` | Alert state machine (persistence, hysteresis, budget) → `Alert` | stateful, deterministic |
| `alerts/analysis.py` | Operating point of the alert rules by forward recursion (docs/02 §7.1) | yes |
| `replay/simulator.py` | Feed recorded or synthetic streams through the live path with latency/dropout injection and a look-ahead guard | - |
| `evaluation/` | Splits, metrics, bootstrap, decision curves, Gate C report, model card; `sensitivity.py`: input attribution, substitution intensity, tipping point, Aalen-Johansen | yes |
| `ingest/` | Source adapters → `TrackingSample`/`MatchEvent`; clock sync | - |
| `video/identity.py` | Assign video tracks to lineup players (Hungarian matching with jersey/team constraints) | yes |
| `app/` | FastAPI server, WebSocket push, static tablet page, role-based views, wording | - |
| `storage/logstore.py` | Append-only Parquet writer; DuckDB views | - |

## 3. Process model (live)

Single process, `asyncio`:
- One task per source adapter, pushing samples into a bounded queue (drop-oldest with a counter
  when full; the counter feeds quality).
- A feature task wakes every `feature_tick_s` on the **match clock** (from events), drains the
  queue, updates per-player state, emits `FeatureSnapshot`s.
- A risk task wakes every `risk_tick_s`, reads the latest snapshots, emits `RiskSnapshot`s and
  alert updates.
- The app pushes the latest state over WebSocket; clients render; every render is logged.
- The log writer batches to Parquet every 10 s.

Replay mode uses the same tasks with a simulated clock (speed factor configurable).

## 4. Clocks

- **Match clock** (`period`, `t_s`) comes from `kickoff`/`period_end` events and the source's
  timestamps. All features and risk use the match clock.
- **Wall clock** measures latency and staleness only: `data_age_s = now_wall − wall_time of the
  newest sample received for that player` (before his first sample: time since he came on). The
  blanking gap δ is part of the design, stated by `cutoff_t_s`, and never counts as age
  (D-021).
- Clock sync: each adapter estimates its offset to the match clock from kick-off; drift > 2 s
  raises a `degraded` flag until resynced.

## 5. Latency budget (99th percentile, excluding TV feeds)

| Stage | Budget |
| --- | --- |
| Source to ingest (wearables, optical, own camera) | ≤ 5 s |
| Feature tick | ≤ 10 s cadence, ≤ 200 ms compute for 30 players |
| Risk tick | ≤ 60 s cadence, ≤ 500 ms compute incl. 1,000 posterior draws for 30 players |
| Push to screen | ≤ 5 s |
| **End to end** | **< 30 s** |

## 6. Failure modes and fallbacks

| Failure | Detection | Behaviour |
| --- | --- | --- |
| Wearable dropout / not worn | coverage per window | affected features z* = 0; quality `degraded`; if none valid, `prematch_only` |
| Identity swap (video) | conflicting jersey reads; low identity confidence | freeze that player's live features until re-identified |
| Player off camera | `detected=False` share | coverage counts detected samples only |
| Clock drift | resync check vs events | `degraded` until resynced |
| Substitution not entered | tracked players vs lineup | prompt analyst; manual event entry in app |
| Network loss to tablet | heartbeat | banner "data stale since mm:ss"; grey > 60 s; hide > 180 s |
| Out-of-range inputs | range checks vs training data | `prematch_only` with note |
| Crash | watchdog/restart | restore state from log store; recompute features from buffered samples |

## 7. Modes

| Mode | Purpose |
| --- | --- |
| `live` | Real match |
| `shadow` | Real match, outputs logged but not displayed to staff (analyst-only debug view) |
| `replay` | Recorded or synthetic match at chosen speed |
| `monitor_only` (engine.mode) | Gate C failed: show load vs norm, no risk claims |

## 8. Security and privacy

- App binds to `127.0.0.1` by default; LAN exposure only by explicit config, with a token.
- Role-based payloads: `coach` receives only names, alert level and one-line reason.
- No outbound network calls. No telemetry.
