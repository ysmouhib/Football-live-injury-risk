# 03: Data contracts

Every module exchanges only the shapes defined here. The pydantic models in
`src/livrisk/contracts.py` implement them (tested). Tabular storage uses the same field names.

## 1. Conventions

| Item | Convention |
| --- | --- |
| Distances | metres (`_m`) |
| Speeds | metres per second (`_mps`); km/h only in the UI (× 3.6) |
| Pitch coordinates | origin at the centre spot; `x_m` along the length in [−L/2, L/2], `y_m` across in [−W/2, W/2]; default L = 105, W = 68. Direction of play does not matter for load features. |
| Match clock | `t_s`: seconds on a continuous match clock. Period 1 starts at 0 and may exceed 2,700 (stoppage). Period 2 starts at 2,700. Period 3 starts at 5,400, period 4 at 6,300. `t_min = t_s / 60` is the `t` of docs/02_MATH.md. |
| Wall clock | `wall_time_ns`: UTC nanoseconds when the sample was produced by the source (if known) and `ingest_time_ns` when received. Used only for latency and staleness. |
| IDs | strings. `player_id` is a stable pseudonymous ID (never a name) inside the engine; names live only in the app's display table. |
| Missing values | `None` in models, null in Parquet. Never 0 as a stand-in for missing. |

## 2. `TrackingSample`: one player, one instant

| Field | Type | Notes |
| --- | --- | --- |
| match_id | str | |
| player_id | str | |
| team_id | str | |
| period | int 1–4 | |
| t_s | float ≥ 0 | match clock |
| x_m, y_m | float or None | None when only speed is available (some wearables) |
| speed_mps | float ≥ 0 or None | if the source provides it; still smoothed by the feature engine |
| source | enum: wearable, optical, video, broadcast, synthetic | |
| detected | bool | False when the position is interpolated or extrapolated by the source (e.g. SkillCorner `is_detected`) |
| identity_confidence | float 0–1 | 1.0 for wearables; from the identity assigner for video |
| wall_time_ns, ingest_time_ns | int or None | |

## 3. `MatchEvent`

| Field | Type | Notes |
| --- | --- | --- |
| match_id | str | |
| period | int 1–4 | |
| t_s | float | |
| type | enum: kickoff, period_end, goal, yellow_card, red_card, substitution, injury_stoppage, stoppage_announced | |
| team_id | str or None | |
| player_id | str or None | player out for substitutions; carded player; injured player |
| player_in_id | str or None | substitutions only |
| stoppage_min | float or None | `stoppage_announced` only |
| source | enum: official_feed, manual, synthetic | |

## 4. `Lineup`

| Field | Type |
| --- | --- |
| match_id, team_id | str |
| kickoff_utc | datetime |
| players | list of `LineupPlayer`: player_id, shirt_number (1–99), position_group (GK, CB, FB, MF, W, FW), starting (bool) |

Goalkeepers are tracked but excluded from risk outputs by default (`engine.include_goalkeepers: false`).

## 5. `PreMatchRisk`: input from the weekly model

| Field | Type | Notes |
| --- | --- | --- |
| player_id | str | |
| as_of_date | date | must be strictly before the match date |
| p7 | float in (0, 1) | calibrated 7-day probability of a non-contact injury ≥ 7 days |
| model_version | str | hash or tag of the weekly model |
| p7_lo, p7_hi | float in (0, 1) or None | optional interval from the weekly model; both or neither; p7_lo ≤ p7 ≤ p7_hi |
| interval_level | float, default 0.90 | coverage of [p7_lo, p7_hi] |
| congested | bool | 2nd of two matches within 3 days, or 3rd of three with ≤ 4-day gaps |
| days_since_last_match | float or None | > 42 triggers a norm reset (docs/02 §3) |
| days_since_return | float or None | days since return from the last time-loss absence; set when it lasted ≥ 14 days |

CSV format accepted by `model/prematch.py`: columns exactly as above, one row per player per
match. The weekly model in `ysmouhib/injury-ml-research` can export its `prospective` log in this
form; consume the file, do not copy its code.

## 6. `InjuryLabel`: training and evaluation only

| Field | Type | Notes |
| --- | --- | --- |
| match_id, player_id | str | |
| period | int | |
| t_s_onset | float | best estimate of onset |
| uncertainty_s | float ≥ 0 | half-width of the annotators' disagreement window |
| contact | bool | |
| tissue | enum: muscle, ligament, tendon, bone, other, unknown | |
| body_area | str | e.g. hamstring, adductor, quadriceps, calf, knee, ankle |
| days_lost | int ≥ 0 or None | |
| source | enum: medical_log, video_annotation, synthetic | |
| annotator_agreement | float 0–1 or None | |

## 7. Outputs

### `FeatureSnapshot` (every 10 s, per player on the pitch)
`match_id, player_id, period, as_of_t_s, cutoff_t_s (= as_of − δ), features: dict[str, float|None]`
(raw values), `z: dict[str, float|None]` (player-relative), `coverage: dict[str, float]`
(share of window with valid samples), `identity_confidence: float`, `quality` (good, degraded,
prematch_only, stale), `quality_reasons: tuple[str, ...]` (codes of `app.wording.QUALITY_REASON_TEXT`:
low_coverage, identity_low, norm_reset, few_matches, clock_drift).

In snapshots, `as_of_t_s` and `cutoff_t_s` carry the **play clock** (seconds of play since
kick-off, period breaks removed), because the raw match clock is ambiguous across a break; the
mapping comes from `period_end` events. See `docs/DECISIONS.md` D-016.

### `RiskSnapshot` (every 60 s, per player on the pitch)
`match_id, player_id, period, as_of_t_s, data_age_s, quality, mode (risk | monitor_only),
p15, p15_lo, p15_hi, r_rest, r_rest_lo, r_rest_hi, r_usual, rr, prematch_band (low | watch | high),
top_reason_code, model_version`.

### `Alert`
`alert_id, match_id, player_id, level (check | discuss | possible_injury), opened_t_s,
cleared_t_s (None while open), reason_code, rr_at_open, r_rest_at_open, budget_used_after`.

### `DecisionRecord`
`alert_id, t_s, actor_role (analyst | physio | doctor | coach), action (none | physio_check |
discussed | role_change | substituted | other), note`.

## 8. Open data for development (no personal data committed)

| Dataset | Use | Where to put it | Notes |
| --- | --- | --- | --- |
| SkillCorner open data (github.com/SkillCorner/opendata) | Real broadcast-tracking movement for the feature engine, quality flags, replay | `data/open/skillcorner/` | 10 A-League 2024/25 matches, 10 fps, `is_detected` flag; repository lists an MIT licence; credit SkillCorner. Tests skip if absent. |
| SoccerMon (datasets.simula.no/soccermon) | Load norms, wellness; CC BY 4.0 | `data/open/soccermon/` | Session-level GPS; not minute-level match injuries |
| SoccerNet game-state reconstruction (github.com/SoccerNet/sn-gamestate) | Output format reference for M13 | `data/open/soccernet/` | Check access terms before downloading videos |
| Synthetic (M2) | Everything else, with known ground truth | generated into `data/synth/` | Default for all tests |

## 9. Storage

- Raw and derived tables: Parquet, partitioned by `match_id`.
- Query layer: DuckDB views over Parquet.
- Log store (append-only): `logs/{date}/{match_id}/` with `tracking.parquet`, `events.parquet`,
  `features.parquet`, `risk.parquet`, `alerts.parquet`, `decisions.parquet`, `display.parquet`
  (what was on screen and when). Never rewritten.
