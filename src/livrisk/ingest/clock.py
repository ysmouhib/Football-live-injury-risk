"""Clock synchronisation of one source clock to the match clock (docs/04_ARCHITECTURE.md §4).

Each adapter estimates its offset to the match clock at kick-off and resyncs at later known
correspondences (period kick-offs). Between sync points the drift is the discrepancy between the
source clock's advance and the match clock's advance; once it exceeds ``DRIFT_TOLERANCE_S`` the
stream stays flagged ``degraded`` (quality reason ``clock_drift``) until the next ``sync``.
"""

from __future__ import annotations

#: Drift tolerance, seconds (docs/04_ARCHITECTURE.md §4).
DRIFT_TOLERANCE_S: float = 2.0


class ClockSync:
    """Offset and drift of one source clock against the match clock, in seconds.

    ``sync`` fixes the offset at a known correspondence (kick-off: the source's own reading
    against match ``t_s`` 0). ``observe`` returns the current drift and latches ``degraded``
    once |drift| exceeds the tolerance; the flag stays set until the next ``sync``.
    """

    def __init__(self, tolerance_s: float = DRIFT_TOLERANCE_S) -> None:
        if tolerance_s <= 0.0:
            raise ValueError(f"tolerance_s must be > 0, got {tolerance_s}")
        self._tolerance = float(tolerance_s)
        self._offset_s: float | None = None
        self._drift_s: float = 0.0
        self._degraded: bool = False

    @property
    def offset_s(self) -> float | None:
        """Source reading minus match reading at the last sync point, seconds (None if unsynced)."""
        return self._offset_s

    @property
    def drift_s(self) -> float:
        """Drift at the last observation, seconds (0.0 before any observation)."""
        return self._drift_s

    @property
    def degraded(self) -> bool:
        """True from the first observation beyond the tolerance until the next ``sync``."""
        return self._degraded

    def sync(self, match_t_s: float, source_t_s: float) -> None:
        """(Re)synchronise at a known correspondence; clears the drift and the degraded flag."""
        self._offset_s = float(source_t_s) - float(match_t_s)
        self._drift_s = 0.0
        self._degraded = False

    def observe(self, match_t_s: float, source_t_s: float) -> float:
        """Drift (seconds) at a new correspondence of the two clocks.

        Raises ``ValueError`` when no sync point has been seen yet: an offset estimate must come
        from kick-off before drift can be judged.
        """
        if self._offset_s is None:
            raise ValueError("observe before sync: no clock offset estimated yet")
        self._drift_s = (float(source_t_s) - self._offset_s) - float(match_t_s)
        if abs(self._drift_s) > self._tolerance:
            self._degraded = True
        return self._drift_s
