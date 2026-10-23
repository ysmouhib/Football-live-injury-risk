# Runbook: operating live-injury-risk

Match-day operation: what to check before, what to watch during, and what to do when something
fails. The tool reports each player's injury risk for the rest of the match relative to his own
normal, with the reason in plain words and the age and quality of the data behind it. It does
not predict injuries and it does not recommend substitutions; staff decide.

## 1. Commands

- `livrisk replay --synthetic --seed 1` replays a synthetic match through the live path and
  serves the bench app (replay mode). Useful flags: `--until-s` stops the replay after that many
  play seconds, `--speed` sets the cursor speed of the served replay, `--no-serve` writes the
  log store and session files without starting the app, and `--latency-s`, `--dropout-rate`,
  `--burst-s` inject the faults of docs/05_VALIDATION.md §3.
- `livrisk serve --match-id <id>` serves the bench app of a finished replay from its log store
  (`--match-date` disambiguates when the id exists under several dates).
- `livrisk fit --input <person-minutes.csv|parquet> --output fitted.json --version <v>` fits the
  live hazard model. `--synthetic --seed <n>` fits a synthetic table instead.
- `livrisk evaluate --output report.md [--card card.md]` writes the synthetic Gate C report.
  With default arguments it is the pre-registered synthetic scenario of the acceptance tests,
  byte-identical between runs.
- `livrisk retention` deletes log partitions older than `storage.retention_days`
  (30 days by default); `--days` overrides.

Every command takes `--config` (default: `LIVRISK_CONFIG`, then `configs/default.yaml`) and is
deterministic: the same arguments produce the same output.

## 2. Before the match (checklist)

- Model version and config frozen. Note both hashes; every log row carries the config hash, and
  the Gate C report of the model version states the engine mode it supports.
- If Gate C failed for this version, run `engine.mode: monitor_only`: load against the player's
  own norm, no live risk figures. Do not run `risk` mode on a version that has not passed.
- The fit comes only from allowed training data: historical and shadow-mode matches, plus
  live-mode matches with alerted players censored at the first shown alert (docs/02 §6).
- Weekly risk input loaded for every squad player, dated before the match day; the pre-match
  bridge rejects look-ahead dates.
- Lineups and shirt numbers entered; wearables charged, fitted and assigned; the video pipeline
  feeding tracks where used.
- Tablet on the local network, app reachable, roles as configured (`app.roles`). The coach view
  shows names, alert levels and one-line reasons only.
- Log store writable with disk headroom (about 1.3 million tracking rows per match at 10 Hz);
  `livrisk retention` run.
- Smoke test on synthetic data: `livrisk replay --synthetic --seed 1 --until-s 600` comes up and
  shows the pre-match screen.

## 3. During the match

- Every value carries its data age. Values older than 60 s are greyed, older than 180 s hidden.
  Treat a greyed value as old information, not as a current reading.
- The quality chip says why a value is degraded (low coverage, identity, norm reset). A player
  on `prematch_only` quality is shown on his pre-match risk alone.
- Alerts are Check or Discuss cards, each with the fixed line "Not a prediction of injury;
  staff decide." The budget is `alerts.budget_per_match` per team per match. Record a decision
  on every alert card; it is written to the log store with the match clock.
- Nothing on the screen is a prediction. A low value is not an all-clear: most injuries happen
  to players rated ordinary.

## 4. Failure handling

The system fails toward the pre-match view and says so; a stale value is never shown as live.
The table of detection and automatic behaviour is docs/04_ARCHITECTURE.md §6; what to do:

### Wearable dropout / not worn

- What happens: the player's window coverage drops; affected features are neutralised (z* = 0)
  and the quality falls to `degraded`, then to `prematch_only` when no feature is valid.
- Do: check the device at the next stoppage and swap in a charged spare. Values recover on their
  own once data flows again; nothing needs restarting.

### Identity swap (video)

- What happens: conflicting jersey reads or low identity confidence freeze the player's live
  features until he is re-identified; his quality line carries the identity reason.
- Do: tell the video operator and check the swap flags in the tracking feed. The freeze is
  intentional: a doubtful identity is a coverage gap, never a guessed one.

### Player off camera

- What happens: only detected samples count toward coverage, so stretches off camera lower
  coverage exactly like a dropout and degrade the quality the same way.
- Do: nothing while it affects one player briefly; if it persists or affects many players, check
  the camera installation and the feed.

### Clock drift

- What happens: a source clock drifting more than 2 s from the match clock flags the values
  `degraded` until the source resyncs.
- Do: resync the source clock (the adapter logs the offset). If it recurs, treat the source as
  unreliable for the day and note it in the match log.

### Substitution not entered

- What happens: the tracked players and the lineup disagree; the analyst is prompted and the
  event can be entered manually in the app.
- Do: enter the substitution as soon as it is noticed. Until then the leaving player keeps live
  rows and the incoming one has none; note the window on any alert decision made in it.

### Network loss to tablet

- What happens: the heartbeat detects the loss; the page shows the banner "Data stale since
  mm:ss", greys values older than 60 s and hides them after 180 s.
- Do: reconnect the tablet to the local network. The app reconnects on its own and shows the
  connection state; no value is lost, the log store holds everything.

### Out-of-range inputs

- What happens: inputs outside the ranges seen in training fall back to `prematch_only` with a
  note; speeds above `features.max_plausible_speed_mps` count as missing and lower coverage.
- Do: check the feed mapping against docs/03_DATA_CONTRACTS.md. A systematically out-of-range
  source is fixed at the adapter, not worked around on match day.

### Crash

- What happens: the watchdog restarts the process; state is restored from the log store and the
  features are recomputed from the buffered samples. A crash loses at most the rows not yet
  flushed (the flush interval), never an already written part file.
- Do: bring the app back up, confirm the match clock and every player's data age, and check the
  log store for the gap. If it crashes again, finish the match in `monitor_only` mode and
  investigate after.

## 5. Shadow mode

Shadow mode (docs/04_ARCHITECTURE.md §7) runs the full live path during real matches with every
output logged and nothing shown to staff; the analyst has a debug view only.

- Freeze the model version and the config for the season; both hashes are logged with every row.
- Run every match. Do not tune on what the shadow logs show. Shadow-mode matches are allowed
  training data for later versions (`evaluation.training_sources`).
- Score once, at the end of the season, against the Gate C criteria plus the latency and uptime
  targets (99 % of updates under 30 s end to end; no stale value shown as live), computed from
  the log store only.
- Leaving shadow mode takes a new pre-registration (docs/05_VALIDATION.md §5); it is not a
  match-day call.

## 6. After the match

- Confirm the match's log partition is complete (tracking, events, features, risk, alerts,
  decisions, display) before the tablets leave the bench.
- Export nothing by default: individual scores stay in the local store (docs/07_RULES_ETHICS.md).
- Run `livrisk retention` on the match machine's schedule. It deletes whole date partitions
  older than the retention window and nothing else, and never touches anything outside the log
  root.
