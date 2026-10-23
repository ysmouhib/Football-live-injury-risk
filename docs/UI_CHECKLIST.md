# UI checklist

Manual check of the bench app against docs/06_UI_SPEC.md and the governance list of
docs/07_RULES_ETHICS.md. Run against a replayed synthetic match (the docs/05 §3 scenario:
30 s latency, 5 % dropout, one 30 s burst per half), served locally, tablet viewport.
Evidence: screenshots in `docs/ui/`, taken from that replay; automated coverage in
`tests/test_m11_app.py` where noted.

Checked 2026-10-23.

## Screens (docs/06 §1)

- [x] Pre-match: squad list with pre-match band and usual risk over 90 min per player,
  starting/bench status. Evidence: `docs/ui/pre_match.png`.
- [x] Live: table with header (match clock, data age, alerts used). Evidence:
  `docs/ui/live_alert.png`.
- [x] Half-time: first-half load vs his norm per player, projected second-half risk for 45 vs
  20 more minutes. Evidence: `docs/ui/half_time.png`.
- [x] Post-match: alerts with level, reason, opened/cleared times and staff decision;
  per-player report with alert count and highest relative value. Evidence:
  `docs/ui/post_match.png`.

## Live table (docs/06 §2)

- [x] One row per on-pitch player, sorted by rest-of-match value descending. Evidence:
  `docs/ui/live_alert.png` and the payload test.
- [x] Columns: pre-match band, now vs his normal with direction symbol, rest of match, why in
  plain words (36 characters or fewer), data quality. Evidence: `docs/ui/live_alert.png`.
- [x] Values that would round to 0.0 % show as "< 0.1 %", never zero. Evidence:
  `docs/ui/pre_match.png`; covered by a payload test.

## Alerts (docs/06 §3)

- [x] Alert card shows level, player, reason, the agreed action and the disclaimer
  ("Not a prediction of injury; staff decide."). Evidence: `docs/ui/live_alert.png`.
- [x] Decision buttons (acknowledge, physio checked, discussed, dismiss) write a decision
  record to the log store. Evidence: `docs/ui/post_match.png` ("Physio checked"); covered by
  an API test.
- [x] Header shows the budget ("Alerts used: 1 of 2"); qualifying players beyond the budget get
  a subtle marker, no card. Evidence: header in `docs/ui/live_grey.png`; marker in
  `docs/img/match/live_discuss.png` (#5 Full back); covered by alert-logic and payload tests.

## Staleness and quality (docs/06 §4)

- [x] Data older than 60 s shows greyed with its age. Evidence: `docs/ui/live_grey.png`.
- [x] Data older than 180 s is hidden ("-") with the banner "Data stale since mm:ss"; the
  banner shows the match time the data stopped (end of the first half in the replay).
  Evidence: `docs/ui/live_stale.png`.
- [x] Quality states Good / Fair / Lost / Stale shown as text, never by colour alone.
  Evidence: the Data column in all live screenshots.

## Wording (docs/06 §5-§6)

- [x] Reason texts match the wording table. Evidence: `docs/ui/live_alert.png`,
  `docs/ui/live_grey.png`.
- [x] No forbidden phrases in any screen or payload (also enforced by a test over all strings
  and replayed payloads). Evidence: all screenshots.

## Accessibility (docs/06 §7)

- [x] Light and dark themes, high contrast, meaning carried by text and symbol together.
  Evidence: `docs/ui/live_alert.png` and `docs/ui/live_dark.png`.
- [x] Connection state shown in the header. Evidence: "Connected" in all screenshots.
- [x] Coach role view contains only labels, alert level and a one-line reason, no numbers.
  Evidence: `docs/ui/coach.png`; payload shape covered by an API test.

## Governance (docs/07)

- [x] Advisory only: every alert needs a human decision; decisions are logged with role, action
  and time. Evidence: `docs/ui/post_match.png`; covered by an API test.
- [x] Role-based views: unknown roles are rejected. Covered by an API test.
- [x] No export or broadcast screen anywhere in the app. Evidence: full screen list above.
