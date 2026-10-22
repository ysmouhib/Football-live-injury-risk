"""Data contracts (docs/03_DATA_CONTRACTS.md). Every module exchanges only these shapes."""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

#: Start of each period on the continuous match clock, seconds.
PERIOD_START_S: dict[int, float] = {1: 0.0, 2: 2700.0, 3: 5400.0, 4: 6300.0}


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Source(StrEnum):
    WEARABLE = "wearable"
    OPTICAL = "optical"
    VIDEO = "video"
    BROADCAST = "broadcast"
    SYNTHETIC = "synthetic"


class EventSource(StrEnum):
    OFFICIAL_FEED = "official_feed"
    MANUAL = "manual"
    SYNTHETIC = "synthetic"


class EventType(StrEnum):
    KICKOFF = "kickoff"
    PERIOD_END = "period_end"
    GOAL = "goal"
    YELLOW_CARD = "yellow_card"
    RED_CARD = "red_card"
    SUBSTITUTION = "substitution"
    INJURY_STOPPAGE = "injury_stoppage"
    STOPPAGE_ANNOUNCED = "stoppage_announced"


class PositionGroup(StrEnum):
    GK = "GK"
    CB = "CB"
    FB = "FB"
    MF = "MF"
    W = "W"
    FW = "FW"


class Quality(StrEnum):
    GOOD = "good"
    DEGRADED = "degraded"
    PREMATCH_ONLY = "prematch_only"
    STALE = "stale"


class Mode(StrEnum):
    RISK = "risk"
    MONITOR_ONLY = "monitor_only"


class PrematchBand(StrEnum):
    LOW = "low"
    WATCH = "watch"
    HIGH = "high"


class AlertLevel(StrEnum):
    CHECK = "check"
    DISCUSS = "discuss"
    POSSIBLE_INJURY = "possible_injury"


class ActorRole(StrEnum):
    ANALYST = "analyst"
    PHYSIO = "physio"
    DOCTOR = "doctor"
    COACH = "coach"


class Action(StrEnum):
    NONE = "none"
    PHYSIO_CHECK = "physio_check"
    DISCUSSED = "discussed"
    ROLE_CHANGE = "role_change"
    SUBSTITUTED = "substituted"
    OTHER = "other"


class Tissue(StrEnum):
    MUSCLE = "muscle"
    LIGAMENT = "ligament"
    TENDON = "tendon"
    BONE = "bone"
    OTHER = "other"
    UNKNOWN = "unknown"


class LabelSource(StrEnum):
    MEDICAL_LOG = "medical_log"
    VIDEO_ANNOTATION = "video_annotation"
    SYNTHETIC = "synthetic"


def _check_period_time(period: int, t_s: float) -> None:
    if period not in PERIOD_START_S:
        raise ValueError(f"period must be 1-4, got {period}")
    if t_s < PERIOD_START_S[period]:
        raise ValueError(f"t_s={t_s} is before the start of period {period}")


class TrackingSample(_Frozen):
    """One player at one instant (§2)."""

    match_id: str
    player_id: str
    team_id: str
    period: int = Field(ge=1, le=4)
    t_s: float = Field(ge=0.0)
    x_m: float | None = None
    y_m: float | None = None
    speed_mps: float | None = Field(default=None, ge=0.0)
    source: Source
    detected: bool = True
    identity_confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    wall_time_ns: int | None = None
    ingest_time_ns: int | None = None

    @model_validator(mode="after")
    def _check(self) -> TrackingSample:
        _check_period_time(self.period, self.t_s)
        if (self.x_m is None) != (self.y_m is None):
            raise ValueError("x_m and y_m must both be set or both be None")
        if self.x_m is None and self.speed_mps is None:
            raise ValueError("a sample needs a position or a speed")
        return self


class MatchEvent(_Frozen):
    """§3."""

    match_id: str
    period: int = Field(ge=1, le=4)
    t_s: float = Field(ge=0.0)
    type: EventType
    team_id: str | None = None
    player_id: str | None = None
    player_in_id: str | None = None
    stoppage_min: float | None = Field(default=None, ge=0.0)
    source: EventSource = EventSource.OFFICIAL_FEED

    @model_validator(mode="after")
    def _check(self) -> MatchEvent:
        _check_period_time(self.period, self.t_s)
        if self.type == EventType.SUBSTITUTION and (
            self.player_id is None or self.player_in_id is None
        ):
            raise ValueError("substitution needs player_id (out) and player_in_id (in)")
        if self.type == EventType.STOPPAGE_ANNOUNCED and self.stoppage_min is None:
            raise ValueError("stoppage_announced needs stoppage_min")
        return self


class LineupPlayer(_Frozen):
    player_id: str
    shirt_number: int = Field(ge=1, le=99)
    position_group: PositionGroup
    starting: bool


class Lineup(_Frozen):
    """§4."""

    match_id: str
    team_id: str
    kickoff_utc: datetime
    players: tuple[LineupPlayer, ...]

    @field_validator("players")
    @classmethod
    def _unique(cls, v: tuple[LineupPlayer, ...]) -> tuple[LineupPlayer, ...]:
        ids = [p.player_id for p in v]
        shirts = [p.shirt_number for p in v]
        if len(set(ids)) != len(ids) or len(set(shirts)) != len(shirts):
            raise ValueError("player_id and shirt_number must be unique within a lineup")
        if sum(p.starting for p in v) > 11:
            raise ValueError("at most 11 starters")
        return v


class PreMatchRisk(_Frozen):
    """§5. Input from the weekly model. The interval is optional; when given, the engine
    propagates it into the displayed intervals (docs/02_MATH.md §5)."""

    player_id: str
    as_of_date: date
    p7: float = Field(gt=0.0, lt=1.0)
    p7_lo: float | None = Field(default=None, gt=0.0, lt=1.0)
    p7_hi: float | None = Field(default=None, gt=0.0, lt=1.0)
    interval_level: float = Field(default=0.90, gt=0.0, lt=1.0)
    model_version: str
    congested: bool = False
    days_since_last_match: float | None = Field(default=None, ge=0.0)
    days_since_return: float | None = Field(default=None, ge=0.0)

    @model_validator(mode="after")
    def _interval(self) -> PreMatchRisk:
        if (self.p7_lo is None) != (self.p7_hi is None):
            raise ValueError("p7_lo and p7_hi must both be set or both be None")
        if (
            self.p7_lo is not None
            and self.p7_hi is not None
            and not self.p7_lo <= self.p7 <= self.p7_hi
        ):
            raise ValueError("p7 must lie within [p7_lo, p7_hi]")
        return self


class InjuryLabel(_Frozen):
    """§6. Training and evaluation only."""

    match_id: str
    player_id: str
    period: int = Field(ge=1, le=4)
    t_s_onset: float = Field(ge=0.0)
    uncertainty_s: float = Field(ge=0.0)
    contact: bool
    tissue: Tissue
    body_area: str
    days_lost: int | None = Field(default=None, ge=0)
    source: LabelSource
    annotator_agreement: float | None = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _check(self) -> InjuryLabel:
        _check_period_time(self.period, self.t_s_onset)
        return self


class FeatureSnapshot(_Frozen):
    """§7. Every 10 s per player on the pitch."""

    match_id: str
    player_id: str
    period: int = Field(ge=1, le=4)
    as_of_t_s: float
    cutoff_t_s: float
    features: dict[str, float | None]
    z: dict[str, float | None]
    coverage: dict[str, float]
    identity_confidence: float = Field(ge=0.0, le=1.0)
    quality: Quality
    #: Codes from ``app.wording.QUALITY_REASON_TEXT`` explaining a non-good quality or a norm reset.
    quality_reasons: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _no_lookahead(self) -> FeatureSnapshot:
        if self.cutoff_t_s > self.as_of_t_s:
            raise ValueError("cutoff_t_s must be <= as_of_t_s (no look-ahead)")
        if any(not 0.0 <= c <= 1.0 for c in self.coverage.values()):
            raise ValueError("coverage values must be in [0, 1]")
        return self


class RiskSnapshot(_Frozen):
    """§7. Every 60 s per player on the pitch."""

    match_id: str
    player_id: str
    period: int = Field(ge=1, le=4)
    as_of_t_s: float
    data_age_s: float = Field(ge=0.0)
    quality: Quality
    mode: Mode
    p15: float = Field(ge=0.0, le=1.0)
    p15_lo: float = Field(ge=0.0, le=1.0)
    p15_hi: float = Field(ge=0.0, le=1.0)
    r_rest: float = Field(ge=0.0, le=1.0)
    r_rest_lo: float = Field(ge=0.0, le=1.0)
    r_rest_hi: float = Field(ge=0.0, le=1.0)
    r_usual: float = Field(ge=0.0, le=1.0)
    rr: float = Field(gt=0.0)
    prematch_band: PrematchBand
    top_reason_code: str | None = None
    model_version: str

    @model_validator(mode="after")
    def _intervals(self) -> RiskSnapshot:
        if not self.p15_lo <= self.p15 <= self.p15_hi:
            raise ValueError("p15 must lie within [p15_lo, p15_hi]")
        if not self.r_rest_lo <= self.r_rest <= self.r_rest_hi:
            raise ValueError("r_rest must lie within [r_rest_lo, r_rest_hi]")
        if self.mode == Mode.MONITOR_ONLY and abs(self.rr - 1.0) > 1e-12:
            raise ValueError("monitor_only mode must report rr == 1")
        return self


class Alert(_Frozen):
    """§7."""

    alert_id: str
    match_id: str
    player_id: str
    level: AlertLevel
    opened_t_s: float
    cleared_t_s: float | None = None
    reason_code: str
    rr_at_open: float = Field(gt=0.0)
    r_rest_at_open: float = Field(ge=0.0, le=1.0)
    budget_used_after: int = Field(ge=0)


class DecisionRecord(_Frozen):
    """§7."""

    alert_id: str
    t_s: float
    actor_role: ActorRole
    action: Action
    note: str = ""
