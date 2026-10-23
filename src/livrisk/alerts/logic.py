"""Alert state machine (docs/06_UI_SPEC.md §3).

Deterministic: persistence, hysteresis, budget, priority by r_rest, Check -> Discuss upgrade
without extra budget, clearing on substitution, no Check/Discuss alerts in monitor_only mode.

One instance tracks one team in one match: the budget of ``alerts.budget_per_match`` is per team
per match, so the caller passes only that team's ``RiskSnapshot`` rows. Possible-injury events
(docs/02_MATH.md §11) are raised by the detection rule on the feature stream, not here; they sit
outside the budget and are the only alerts allowed in monitor_only mode.

Rule semantics (D-020):

- A level streak counts consecutive risk updates with RR at or above the level threshold and
  quality good or degraded; a stale or prematch_only update reports a fallback RR, so it cannot
  count as evidence and resets the streak.
- Opening (Check, or Discuss directly) additionally requires R_rest at or above
  ``alerts.check.r_rest_floor``; the floor gates the opening update, not the streak.
- An open alert clears when RR stays below ``alerts.clear_rr_below`` for
  ``alerts.check.persist_updates`` updates (any quality; there is no downgrade from Discuss), or
  immediately when the player leaves the pitch.
- An open Check upgrades to Discuss after ``alerts.discuss.persist_updates`` updates at or above
  ``alerts.discuss.rr_min``, or on a staff "concern" flag; upgrades consume no budget.
- The budget counts openings per team-match; a clearing does not refund a slot. Qualifying
  players denied by the budget are reported by ``muted_qualifiers`` (the table's subtle marker).
- Priority among players qualifying in the same update: higher R_rest first, then player_id.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

from livrisk.config import Config
from livrisk.contracts import Action, Alert, AlertLevel, DecisionRecord, Mode, Quality, RiskSnapshot

logger = logging.getLogger(__name__)

#: Reason code of an opening row without a dominant feature contribution (docs/06 §5).
_FALLBACK_REASON = "WITHIN_NORMAL"


@dataclass
class _Streaks:
    """Per-player consecutive-update counters (capped at their persistence requirements)."""

    check: int = 0
    discuss: int = 0
    clear: int = 0


class AlertLogic:
    """Check/Discuss state machine over a team's risk updates (docs/06 §3).

    Construction fixes the thresholds from ``cfg.alerts`` and the mode from ``cfg.engine.mode``.
    ``update`` is deterministic given its inputs; all times are match-clock seconds (docs/03 §1).
    Alert ids are ``{id_prefix}A1``, ``{id_prefix}A2``, ... in opening order; a per-team prefix
    keeps them unique within a match.
    """

    def __init__(self, cfg: Config, match_id: str, *, id_prefix: str = "") -> None:
        alerts = cfg.alerts
        self._match_id = match_id
        self._id_prefix = id_prefix
        self._mode = Mode(cfg.engine.mode)
        self._rr_check = float(alerts.check.rr_min)
        self._persist_check = int(alerts.check.persist_updates)
        self._r_rest_floor = float(alerts.check.r_rest_floor)
        self._rr_discuss = float(alerts.discuss.rr_min)
        self._persist_discuss = int(alerts.discuss.persist_updates)
        self._clear_rr = float(alerts.clear_rr_below)
        self._budget = int(alerts.budget_per_match)
        self._streaks: dict[str, _Streaks] = {}
        self._open: dict[str, Alert] = {}
        self._budget_used = 0
        self._n_alerts = 0
        self._decisions: list[DecisionRecord] = []
        self._muted: tuple[str, ...] = ()
        self._pending_upgrades: list[Alert] = []
        self._last_t_s = 0.0

    def update(self, risks: Sequence[RiskSnapshot], on_pitch: set[str]) -> list[Alert]:
        """Advance the machine by one risk update; return alerts opened, upgraded, cleared.

        ``risks`` carries one ``RiskSnapshot`` per player with a row at this update; ``on_pitch``
        is the authoritative set of the tracked team's players currently on the pitch. Rows for
        players not in ``on_pitch`` are ignored, and a player who left the pitch has any open
        alert cleared immediately and his streaks dropped. A player on the pitch without a row
        keeps his open alert but his opening streaks restart (no live evidence this update).
        The returned alerts are ordered: opened (by priority), then upgraded, then cleared.
        """
        for row in risks:
            if row.match_id != self._match_id:
                raise ValueError(
                    f"risk row for match {row.match_id!r} in alert logic for {self._match_id!r}"
                )
        t_now = max((r.as_of_t_s for r in risks), default=self._last_t_s)
        self._last_t_s = t_now
        opened: list[Alert] = []
        upgraded: list[Alert] = []
        cleared: list[Alert] = []
        for pid in sorted((set(self._open) | set(self._streaks)) - on_pitch):
            alert = self._open.pop(pid, None)
            if alert is not None:
                cleared.append(alert.model_copy(update={"cleared_t_s": t_now}))
            self._streaks.pop(pid, None)
        rows = sorted((r for r in risks if r.player_id in on_pitch), key=lambda r: r.player_id)
        for row in rows:
            st = self._streaks.setdefault(row.player_id, _Streaks())
            live = row.quality in (Quality.GOOD, Quality.DEGRADED)
            st.check = (
                min(st.check + 1, self._persist_check) if live and row.rr >= self._rr_check else 0
            )
            st.discuss = (
                min(st.discuss + 1, self._persist_discuss)
                if live and row.rr >= self._rr_discuss
                else 0
            )
            st.clear = min(st.clear + 1, self._persist_check) if row.rr < self._clear_rr else 0
        seen = {r.player_id for r in rows}
        for pid, st in self._streaks.items():
            if pid not in seen:
                st.check = 0
                st.discuss = 0
        for pid in sorted(self._open):
            st_open = self._streaks.get(pid)
            if st_open is not None and st_open.clear >= self._persist_check:
                cleared.append(self._open.pop(pid).model_copy(update={"cleared_t_s": t_now}))
        if self._mode != Mode.MONITOR_ONLY:
            self._upgrades(upgraded)
            self._openings(rows, t_now, opened)
        else:
            self._muted = ()
        upgraded.extend(self._pending_upgrades)
        self._pending_upgrades = []
        return opened + upgraded + cleared

    def _upgrades(self, upgraded: list[Alert]) -> None:
        """Upgrade open Checks whose Discuss streak is met (no budget, no floor re-gate)."""
        for pid in sorted(self._open):
            alert = self._open[pid]
            if (
                alert.level == AlertLevel.CHECK
                and self._streaks[pid].discuss >= self._persist_discuss
            ):
                new = alert.model_copy(update={"level": AlertLevel.DISCUSS})
                self._open[pid] = new
                upgraded.append(new)

    def _openings(self, rows: list[RiskSnapshot], t_now: float, opened: list[Alert]) -> None:
        """Open alerts for qualifying players in priority order while the budget lasts."""
        candidates: list[tuple[RiskSnapshot, AlertLevel]] = []
        for row in rows:
            pid = row.player_id
            if pid in self._open or row.r_rest < self._r_rest_floor:
                continue
            st = self._streaks[pid]
            if st.discuss >= self._persist_discuss:
                candidates.append((row, AlertLevel.DISCUSS))
            elif st.check >= self._persist_check:
                candidates.append((row, AlertLevel.CHECK))
        candidates.sort(key=lambda c: (-c[0].r_rest, c[0].player_id))
        muted: list[str] = []
        for row, level in candidates:
            if self._budget_used >= self._budget:
                muted.append(row.player_id)
                continue
            self._budget_used += 1
            self._n_alerts += 1
            alert = Alert(
                alert_id=f"{self._id_prefix}A{self._n_alerts}",
                match_id=self._match_id,
                player_id=row.player_id,
                level=level,
                opened_t_s=t_now,
                reason_code=row.top_reason_code or _FALLBACK_REASON,
                rr_at_open=row.rr,
                r_rest_at_open=row.r_rest,
                budget_used_after=self._budget_used,
            )
            self._open[row.player_id] = alert
            opened.append(alert)
        self._muted = tuple(muted)

    def open_alerts(self) -> list[Alert]:
        """Currently open alerts in opening order."""
        return list(self._open.values())

    @property
    def budget_used(self) -> int:
        """Check/Discuss openings so far this team-match (upgrades do not count)."""
        return self._budget_used

    @property
    def decisions(self) -> tuple[DecisionRecord, ...]:
        """Staff decisions recorded so far, in recording order."""
        return tuple(self._decisions)

    def muted_qualifiers(self) -> tuple[str, ...]:
        """Players qualifying at the latest update but denied a card by the budget (in priority
        order; docs/06 §3 shows them with a subtle marker instead)."""
        return self._muted

    def record_decision(self, record: DecisionRecord) -> None:
        """Store a staff decision; a "concern" flag on an open Check upgrades it to Discuss.

        The concern flag of docs/06 §3 is a ``DecisionRecord`` with action ``other`` and the
        word "concern" in the note (D-020). The upgrade consumes no budget and is returned by
        the next ``update``. Decisions are stored regardless; in monitor_only mode they never
        upgrade.
        """
        self._decisions.append(record)
        if self._mode == Mode.MONITOR_ONLY or record.action != Action.OTHER:
            return
        if "concern" not in record.note.lower():
            return
        for pid, alert in self._open.items():
            if alert.alert_id == record.alert_id and alert.level == AlertLevel.CHECK:
                new = alert.model_copy(update={"level": AlertLevel.DISCUSS})
                self._open[pid] = new
                self._pending_upgrades.append(new)
                return
        logger.warning("concern flag references no open check alert: %s", record.alert_id)
