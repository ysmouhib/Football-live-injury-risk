# Video input format for identity assignment

`src/livrisk/video/identity.py` turns the per-frame output of an external tracking pipeline into
`TrackingSample` rows (docs/03 §2) with a `player_id` and an `identity_confidence`. The external
pipeline (detector, tracker, jersey reader) is not part of this repository; this document is the
contract between that pipeline and the identity assigner. The SoccerNet game-state output format
(docs/03 §8) is the reference layout a pipeline adapter should be mapped into first.

## Input: one row per track per frame

| Column | Type | Semantics |
| --- | --- | --- |
| `frame_t_s` | float, seconds | Match clock of the frame (docs/03 §1: period 1 starts at 0, period 2 at 2700) |
| `period` | int 1-4 | Match period of the frame |
| `track_id` | str | Opaque track label from the pipeline; stable while the tracker keeps the same physical player |
| `team_guess` | str | Team of the tracked player; must equal a `team_id` of the given lineups (the assigner raises otherwise, so filter referees and staff in the adapter) |
| `jersey_read` | int 1-99 or empty | Shirt number read on this frame; empty when unreadable |
| `jersey_conf` | float 0-1 or empty | Confidence of the read; weights the vote. Empty counts as zero |
| `x_m`, `y_m` | float, metres | Pitch position, origin at the centre spot (docs/03 §1); required on every row |
| `detected` | bool | False when the pipeline extrapolated the position instead of detecting it |

Rows may arrive in any order; they are sorted internally. Every field except the jersey pair is
required on every row; out-of-range values raise `ValueError` instead of being coerced (D-026).

## Method

- Play-clock windows of `window_s` (default 1 s). Per team and window, the visible tracks are
  assigned to the on-pitch lineup players by Hungarian matching
  (`scipy.optimize.linear_sum_assignment`). The on-pitch set starts from the lineup starters and
  follows substitution and red-card events at their event time.
- Cost per track-player pair: `1 -` share of the window's jersey vote for the player's shirt
  (0.5 when the window has no reads), plus `continuity_weight` times the distance from the
  player's last seen position folded as `d / (d + continuity_scale_m)` (0.5 when the player has
  no fresh position). A pair is refused when reads exist but none support it, or when only a
  stale or distant position anchor supports it.
- Hysteresis: a track is emitted only after the same player wins its assignment for
  `confirm_windows` consecutive windows (default 2). The same rule re-assigns a track after a
  tracker identity swap, and the switch is flagged in the `swaps` output at the end of the
  confirming window, so a flag lands at most `window_s * (confirm_windows + 1)` seconds after
  the physical swap. Unconfirmed or contested windows emit nothing: a doubtful stretch is a
  coverage gap, never a guessed identity.

## Output

`samples`: the TrackingSample columns with `source = video` and no `speed_mps` (the feature
engine derives speed from positions), plus the `track_id` behind each identity for
traceability, in input row order. `identity_confidence` is the share of the last
`read_memory_s` (default 30 s) of reads supporting the emitted identity, shrunk by a prior of
`prior_reads` reads at `prior_misread` error; the read memory is cleared on a confirmed switch.
A window without any reads falls back to a continuity score capped at 0.75, below
`quality.min_identity` (0.8): a jersey-read outage surfaces as the `identity_low` quality
reason in the feature engine, and the risk engine then gates the tracking-derived features
(docs/02 §3).

`swaps`: one row per confirmed identity switch: `track_id`, `period`, `t_s` of the confirming
window's end, `from_player_id`, `to_player_id`.

## Limits

- Players off camera simply have no track in those windows; their coverage dips (docs/04 §6).
- A wrong `team_guess` leaves the track unassignable; the guess is a hard constraint on purpose.
- A tracker that restarts track labels often (e.g. at half-time) loses only
  `confirm_windows - 1` windows per restart.
- Substitution events drive the on-pitch set; a missing event keeps the outgoing player's slot
  alive, but no track can claim it without that player's jersey reads.
