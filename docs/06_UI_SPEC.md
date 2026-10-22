# 06: Bench interface specification

Designed for a three-second glance by the performance analyst or club doctor on the bench, who
relays what matters to the coach. Two alert levels, each tied to a cheap agreed action, and a
strict cap on alerts per match (clinical alert fatigue: 46–96 % override rates; Poly et al. 2020).

## 1. Screens

| Screen | When | Content |
| --- | --- | --- |
| Pre-match | Before kick-off | Squad list: pre-match band (low / watch / high), r_usual for 90 min, agreed minutes plan per player, data source status |
| Live | During play | Table below; header with match clock, data age, alerts used |
| Half-time | Between halves | First-half load vs norm per player; projected second-half risk for 45 vs 20 more minutes |
| Post-match | After the match | What was shown, alerts, decisions, outcomes; per-player report |
| Shadow debug | Shadow mode, analyst only | Live table plus raw features, coverage, z values |

## 2. Live table (one row per player on the pitch, sorted by `r_rest` descending)

| Column | Source field | Display |
| --- | --- | --- |
| Player | display name + shirt number + position | "#14 Right back" |
| Pre-match | `prematch_band` | chip: Low / Watch / High (text, not colour only) |
| Now vs his normal | `rr` | "×1.9 ▲" / "×0.8 ▼" / "×1.0"; "-" when quality is prematch_only or stale |
| Rest of match | `r_rest` (with `r_rest_lo`–`r_rest_hi` on tap) | "2.1 %" |
| Why | `top_reason_code` → wording table §5 | ≤ 36 characters |
| Data | `quality`, `data_age_s` | Good / Fair (degraded) / Lost (prematch_only) / Stale; icon + text |

Header: `67:12 · second half` · `Live data 8 s old` · `Alerts used: 1 of 2`.

Selected row (tap): detail panel with P15 and its interval, R_rest vs R_usual, the features behind
the multiplier as "his value vs his usual" in plain units, data coverage, and a 15-minute sparkline
of RR. In `monitor_only` mode the panel shows only "load vs his usual" lines. An info button beside
the interval shows `HELP_TEXT["interval"]` (what the range includes and leaves out). A quality
chip other than Good shows its reason from `QUALITY_REASON_TEXT`, e.g. "Compared with position
norm" during a norm reset.

Alert card (bottom): level, player, agreed action, one line of numbers, and the fixed sentence
"Not a prediction of injury; staff decide." Buttons: Acknowledge, Physio checked, Discussed,
Dismiss (each writes a `DecisionRecord`).

Wireframe (illustrative values):

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ 67:12 · second half          ● Live data 8 s old             Alerts used: 1/2 │
├──────────────────────────────────────────────────────────────────────────────┤
│ Player          Pre-match  Now vs his normal  Rest of match  Why          Data │
│▌#14 Right back   [High]       ×1.9 ▲             2.1 %   High-speed m 2× usual Good│
│ #8 Midfielder    [Watch]      ×1.3 ▲             1.2 %   Short recovery        Good│
│ #11 Winger       [Watch]      ×0.8 ▼             0.9 %   Sprint load below     Fair│
│ #9 Striker       [Low]        ×1.0               0.6 %   Within normal range   Good│
│ #3 Centre back   [Low]         -                 0.5 %   Live data lost 2 min  Lost│
├──────────────────────────────────────────────────────────────────────────────┤
│ CHECK · #14 Right back · physio observes at the next stoppage                 │
│ Rest-of-match risk 2.1 % against 1.1 % usual. Not a prediction of injury;     │
│ staff decide.                    [Acknowledge] [Physio checked] [Dismiss]     │
└──────────────────────────────────────────────────────────────────────────────┘
```

## 3. Alert rules (config `alerts.*`)

| Level | Opens when | Clears when | Agreed action |
| --- | --- | --- | --- |
| Check | RR ≥ 1.5 for 2 consecutive risk updates **and** R_rest ≥ floor (0.4 %) **and** quality ∈ {good, degraded} | RR < 1.3 for 2 updates, or player leaves the pitch | Physio observes at the next stoppage |
| Discuss | RR ≥ 2.0 for 3 consecutive updates, or an open Check plus staff flag "concern" | as above | Staff discuss options (role change, substitution window) |
| Possible injury (optional) | detection rule docs/02_MATH.md §11 | acknowledged | Medical staff look immediately |

- Budget: at most `alerts.budget_per_match` (2) Check/Discuss alerts per team per match.
  When exhausted, further qualifying players are shown with a subtle marker in the table but no
  card. Possible-injury events are outside the budget.
- Priority when several qualify in the same update: highest R_rest first.
- An upgrade Check → Discuss does not consume extra budget.
- No alerts in `monitor_only` mode except possible-injury (if enabled).
- Operating point (analyst settings screen, before the match): for the current thresholds, the
  expected Checks per team-match, expected open/clear flips, and the chance the budget is used up
  by 60 and by 90 minutes (docs/02 §7.1). Staff choose the budget knowing what it costs.

## 4. Staleness and quality

| data_age_s | Display |
| --- | --- |
| ≤ 60 | normal |
| 60–180 | greyed, with "x s old" |
| > 180 | values hidden ("-"), banner "data stale since mm:ss" |

## 5. Wording table (reason codes → text)

| Code | Text (≤ 36 chars) |
| --- | --- |
| HSR_HIGH | High-speed metres {k}× his usual |
| SPR_HIGH | Sprint metres {k}× his usual |
| RECOVERY_SHORT | Short recovery between sprints |
| HIR_DROP | Running drop larger than usual |
| NEARMAX_MANY | Many near-top-speed sprints |
| DISRUPT | Just after a card, goal or stoppage |
| TIME_PROFILE | Higher-risk spell of the match |
| WITHIN_NORMAL | Within his normal range |
| LOAD_BELOW | Sprint load below his usual |
| DATA_LOST | Live data lost {m} min ago |

The reason is the feature with the largest positive contribution β_k z*_k (or the time profile if
none exceeds 0.1 on the log scale).

## 6. Forbidden wording (enforced by a test over all UI strings and API payloads)

Never output: "will be injured", "is going to get injured", "injury predicted", "predicts injury",
"safe", "no risk", "substitute now", "take him off", "must come off", "danger", "red zone",
"guaranteed". All user-facing strings live in `app/wording.py`; a test scans it and the rendered
payloads.

## 7. Accessibility and ergonomics

- Readable at arm's length outdoors: minimum 18 px body text on tablet, high contrast, light and
  dark themes; meaning never carried by colour alone (text + symbol).
- One tap to detail; no hover-only information.
- Works offline on the local network; reconnects automatically and shows connection state.
- Role views: `coach` payload contains only names, alert level and one-line reason.

## 8. Example WebSocket payload (live table)

```json
{
  "type": "live_table",
  "match_id": "m-2027-03-14-home",
  "period": 2,
  "clock": "67:12",
  "data_age_s": 8.0,
  "alerts_used": 1,
  "alerts_budget": 2,
  "mode": "risk",
  "rows": [
    {
      "player_id": "P014", "label": "#14 Right back", "prematch_band": "high",
      "rr": 1.9, "r_rest": 0.021, "r_rest_lo": 0.013, "r_rest_hi": 0.031, "r_usual": 0.011,
      "p15": 0.006, "reason_code": "HSR_HIGH", "reason_text": "High-speed metres 2× his usual",
      "quality": "good", "data_age_s": 8.0
    }
  ],
  "alerts": [
    {"alert_id": "A1", "player_id": "P014", "level": "check", "reason_code": "HSR_HIGH"}
  ]
}
```
