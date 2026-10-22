"""Source adapter protocol (docs/03 §2-§3; docs/04 §3-§4). Implementations arrive in M12/M13."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol

from livrisk.contracts import MatchEvent, TrackingSample


class TrackingAdapter(Protocol):
    """Yields TrackingSamples in match-clock order (per player), with source and quality flags."""

    def samples(self) -> AsyncIterator[TrackingSample]: ...


class EventAdapter(Protocol):
    """Yields MatchEvents (kick-off, period end, goals, cards, substitutions, stoppage)."""

    def events(self) -> AsyncIterator[MatchEvent]: ...
