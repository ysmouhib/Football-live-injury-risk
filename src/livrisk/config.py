"""Configuration loading.

Typed, frozen pydantic models mirroring ``configs/default.yaml`` exactly
(``extra="forbid"``). Units are stated in each field description. ``config_hash``
is logged with every output row so a run can be tied to its exact parameters.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

DEFAULT_CONFIG_PATH = "configs/default.yaml"


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class EngineConfig(_Frozen):
    """Engine cadence and safety bounds."""

    mode: Literal["risk", "monitor_only"] = Field(
        description="risk: live risk outputs; monitor_only: load vs norm only (Gate C failed)"
    )
    include_goalkeepers: bool = Field(description="include GKs in risk outputs")
    feature_tick_s: int = Field(gt=0, description="feature update cadence, seconds")
    risk_tick_s: int = Field(gt=0, description="risk update cadence, seconds")
    blanking_delta_s: int = Field(
        gt=0, description="delta: features at t use data with timestamp <= t - delta, seconds"
    )
    rr_max: float = Field(ge=1.0, description="safety bound on relative risk RR")
    interval_draws: int = Field(gt=0, description="posterior draws per risk update")
    interval_level: float = Field(gt=0.0, lt=1.0, description="credible interval level")
    default_stoppage_1h_min: float = Field(
        ge=0.0, description="assumed first-half stoppage when not announced, minutes"
    )
    default_stoppage_2h_min: float = Field(
        ge=0.0, description="assumed second-half stoppage when not announced, minutes"
    )
    horizon_min: int = Field(gt=0, description="short-window risk horizon (P15), minutes")


class BandCutoffs(_Frozen):
    """Display bands on r_usual at kick-off (docs/06)."""

    watch: float = Field(gt=0.0, description="r_usual above this is at least 'watch'")
    high: float = Field(gt=0.0, description="r_usual above this is 'high'")

    @model_validator(mode="after")
    def _ordered(self) -> BandCutoffs:
        if self.high <= self.watch:
            raise ValueError("prematch.band_cutoffs.high must exceed .watch")
        return self


class PrematchConfig(_Frozen):
    match_share_prior: float = Field(
        gt=0.0, lt=1.0, description="a0: share of weekly hazard falling in one match"
    )
    band_cutoffs: BandCutoffs


class FeaturesConfig(_Frozen):
    smoothing_window_s: float = Field(gt=0.0, description="Savitzky-Golay window, seconds")
    smoothing_polyorder: int = Field(ge=1, description="Savitzky-Golay polynomial order")
    max_plausible_speed_mps: float = Field(
        gt=0.0, description="speeds above this are set missing, m/s"
    )
    hsr_threshold_mps: float = Field(gt=0.0, description="high-speed running threshold, m/s")
    sprint_threshold_mps: float = Field(gt=0.0, description="sprint threshold, m/s")
    effort_min_duration_s: float = Field(gt=0.0, description="min effort duration, seconds")
    effort_merge_gap_s: float = Field(ge=0.0, description="gaps below this are merged, seconds")
    nearmax_fraction: float = Field(
        gt=0.0, lt=1.0, description="fraction of player v_max counting as near-maximal"
    )
    windows_min: list[int] = Field(description="feature window lengths, minutes")
    main_features: list[str] = Field(description="the K <= 6 features of the main fit")
    z_clip: float = Field(gt=0.0, description="z-score clipping bound, SD units")

    @field_validator("windows_min")
    @classmethod
    def _windows_positive(cls, v: list[int]) -> list[int]:
        if not v or any(w <= 0 for w in v):
            raise ValueError("features.windows_min must be non-empty with positive entries")
        return v

    @model_validator(mode="after")
    def _thresholds_ordered(self) -> FeaturesConfig:
        if self.sprint_threshold_mps <= self.hsr_threshold_mps:
            raise ValueError("features.sprint_threshold_mps must exceed hsr_threshold_mps")
        return self


class NormsConfig(_Frozen):
    kappa_matches: int = Field(
        gt=0, description="empirical-Bayes prior weight, in effective matches"
    )
    min_matches_for_own_norm: int = Field(
        ge=1, description="effective matches before the own norm dominates"
    )
    recency_half_life_matches: float = Field(
        gt=0.0, description="half-life of past-match weights, matches"
    )
    spread_windows_per_band: int = Field(
        ge=1, description="non-overlapping windows per 15-min band for the SD"
    )
    reset_after_absence_days: float = Field(
        gt=0.0, description="days lost to injury that trigger a norm reset"
    )
    reset_after_no_match_days: float = Field(
        gt=0.0, description="days without a match that trigger a norm reset"
    )
    reset_matches: int = Field(ge=1, description="matches after return with a capped own norm")
    reset_own_weight_matches: float = Field(
        gt=0.0, description="effective matches allowed for the own norm during a reset"
    )
    log1p_transform: list[str] = Field(description="features transformed with log1p")
    log_transform: list[str] = Field(description="features transformed with log")


class QualityConfig(_Frozen):
    min_coverage: float = Field(
        gt=0.0, le=1.0, description="min share of a window with valid samples"
    )
    min_identity: float = Field(gt=0.0, le=1.0, description="min identity confidence")
    stale_grey_s: float = Field(gt=0.0, description="values older than this are greyed, seconds")
    stale_hide_s: float = Field(gt=0.0, description="values older than this are hidden, seconds")

    @model_validator(mode="after")
    def _hide_after_grey(self) -> QualityConfig:
        if self.stale_hide_s <= self.stale_grey_s:
            raise ValueError("quality.stale_hide_s must exceed stale_grey_s")
        return self


class PriorsConfig(_Frozen):
    """Normal prior SDs of docs/02_MATH.md section 4."""

    alpha_sd: float = Field(gt=0.0, description="prior SD of alpha")
    theta_sd: float = Field(gt=0.0, description="prior SD of band effects theta_b")
    eta_sd: float = Field(gt=0.0, description="prior SD of the congestion interaction eta")
    beta_sd: float = Field(gt=0.0, description="prior SD of live coefficients beta_k")


class IntervalAuditConfig(_Frozen):
    """Once-per-version Laplace audit (docs/02_MATH.md section 5.1)."""

    proposal_df: float = Field(gt=0.0, description="Student-t proposal degrees of freedom")
    draws: int = Field(gt=0, description="importance-sampling draws")
    khat_max: float = Field(gt=0.0, description="Pareto k diagnostic threshold")
    quantile_rel_tol: float = Field(
        gt=0.0, description="relative tolerance, IS vs Laplace percentiles"
    )
    battery_matches: int = Field(gt=0, description="synthetic cases in the audit battery")


class ModelConfig(_Frozen):
    interval_min: float = Field(gt=0.0, description="model interval length Delta, minutes")
    congestion_interaction: bool = Field(description="enable the eta congestion interaction")
    priors: PriorsConfig
    interval_audit: IntervalAuditConfig
    reliability: dict[str, float] = Field(description="rho_k per feature, 0-1; 1.0 until measured")

    @field_validator("reliability")
    @classmethod
    def _reliability_range(cls, v: dict[str, float]) -> dict[str, float]:
        for name, rho in v.items():
            if not 0.0 <= rho <= 1.0:
                raise ValueError(f"model.reliability.{name} must be in [0, 1]")
        return v


class CheckAlertConfig(_Frozen):
    rr_min: float = Field(gt=0.0, description="RR at or above this may open a Check")
    persist_updates: int = Field(ge=1, description="consecutive updates above threshold")
    r_rest_floor: float = Field(
        ge=0.0, description="R_rest below this never alerts (absolute-risk floor)"
    )


class DiscussAlertConfig(_Frozen):
    rr_min: float = Field(gt=0.0, description="RR at or above this may open a Discuss")
    persist_updates: int = Field(ge=1, description="consecutive updates above threshold")


class AlertAnalysisConfig(_Frozen):
    """Operating-point analysis of the alert rules (docs/02_MATH.md section 7.1)."""

    rr_level_edges: list[float] = Field(description="RR level boundaries, ascending")
    updates_per_match: int = Field(gt=0, description="risk updates per match in the recursion")
    players_per_team: int = Field(gt=0, description="players on the pitch per team")
    replay_agreement_rel: float = Field(
        gt=0.0, description="required relative agreement with replay"
    )

    @field_validator("rr_level_edges")
    @classmethod
    def _edges_ascending(cls, v: list[float]) -> list[float]:
        if not v or any(e <= 0 for e in v) or any(b <= a for a, b in itertools.pairwise(v)):
            raise ValueError("alerts.analysis.rr_level_edges must be positive and ascending")
        return v


class PossibleInjuryConfig(_Frozen):
    effort_fraction_of_vmax: float = Field(
        gt=0.0, lt=1.0, description="effort at this fraction of v_max arms the rule"
    )
    drop_below_mps: float = Field(gt=0.0, description="speed must fall below this, m/s")
    stay_below_mps: float = Field(gt=0.0, description="speed must stay below this, m/s")
    stay_s: float = Field(gt=0.0, description="for at least this long, seconds")
    within_s: float = Field(gt=0.0, description="within this long after the effort, seconds")


class AlertsConfig(_Frozen):
    check: CheckAlertConfig
    discuss: DiscussAlertConfig
    clear_rr_below: float = Field(gt=0.0, description="hysteresis: RR below this clears")
    budget_per_match: int = Field(ge=0, description="max alert openings per team per match")
    analysis: AlertAnalysisConfig
    possible_injury_detection: bool = Field(description="master switch, off by default")
    possible_injury: PossibleInjuryConfig


class EvaluationConfig(_Frozen):
    primary: str = Field(description="the one confirmatory hypothesis id")
    multiplicity: Literal["holm"] = Field(description="multiplicity correction for exploratory")
    exploratory: list[str] = Field(description="exploratory comparator ids")
    input_slope_prior_sd: float = Field(
        gt=0.0, description="prior SD of the ln(Lambda) slope gamma"
    )
    tipping_point_max_multiplier: float = Field(
        ge=1.0, description="largest hazard multiplier in the tipping-point sweep"
    )
    training_sources: list[str] = Field(description="allowed training row sources")


class SynthConfig(_Frozen):
    """Synthetic match generator parameters (M2)."""

    burn_in_s: int = Field(ge=0, description="speed-process burn-in, seconds")
    tick_hz: int = Field(gt=0, description="sampling rate, Hz")
    zones_mps: list[float] = Field(description="speed-zone lower edges, m/s")
    jump_matrix: list[list[float]] = Field(
        description="zone jump chain, rows sum to 1, zero diagonal"
    )
    dwell_mean_s: list[float] = Field(description="mean dwell per zone, seconds")
    dwell_min_s: list[float] = Field(description="minimum dwell per zone, seconds")
    dwell_gamma_shape: list[float] = Field(description="Gamma shape per zone")
    band_intensity: list[float] = Field(description="scales entries into run+ per band")
    position_intensity: dict[str, float] = Field(description="scales hsr/sprint per position")
    accel_max_mps2: float = Field(gt=0.0, description="acceleration limit, m/s^2")
    decel_max_mps2: float = Field(gt=0.0, description="deceleration limit, m/s^2")
    vmax_mean_mps: float = Field(gt=0.0, description="player peak-speed mean, m/s")
    vmax_sd_mps: float = Field(ge=0.0, description="player peak-speed SD, m/s")
    vmax_bounds_mps: list[float] = Field(description="player peak-speed bounds, m/s")
    late_hsr_drop_bounds: list[float] = Field(
        description="accepted band-6 vs band-0 HSR drop range"
    )

    @model_validator(mode="after")
    def _consistent(self) -> SynthConfig:
        n = len(self.zones_mps)
        if n < 2 or any(b <= a for a, b in itertools.pairwise(self.zones_mps)):
            raise ValueError("synth.zones_mps must be ascending with at least two zones")
        for name in ("dwell_mean_s", "dwell_min_s", "dwell_gamma_shape"):
            if len(getattr(self, name)) != n:
                raise ValueError(f"synth.{name} must have one entry per zone")
        if any(d <= 0 for d in self.dwell_mean_s) or any(d < 0 for d in self.dwell_min_s):
            raise ValueError("synth dwell times must be positive (min may be zero)")
        if any(s <= 0 for s in self.dwell_gamma_shape):
            raise ValueError("synth.dwell_gamma_shape entries must be positive")
        if len(self.jump_matrix) != n or any(len(row) != n for row in self.jump_matrix):
            raise ValueError("synth.jump_matrix must be square with one row per zone")
        for row in self.jump_matrix:
            if any(p < 0.0 for p in row) or abs(sum(row) - 1.0) > 1e-9:
                raise ValueError("synth.jump_matrix rows must be probabilities")
        if any(self.jump_matrix[k][k] != 0.0 for k in range(n)):
            raise ValueError("synth.jump_matrix diagonal must be zero")
        if len(self.vmax_bounds_mps) != 2 or self.vmax_bounds_mps[0] >= self.vmax_bounds_mps[1]:
            raise ValueError("synth.vmax_bounds_mps must be an ascending pair")
        if len(self.late_hsr_drop_bounds) != 2 or not (
            0.0 <= self.late_hsr_drop_bounds[0] <= self.late_hsr_drop_bounds[1] <= 1.0
        ):
            raise ValueError("synth.late_hsr_drop_bounds must be an ascending pair in [0, 1]")
        return self


class ResearchConfig(_Frozen):
    """Exploratory comparators only; never in the main fit."""

    decayed_disrupt_tau_s: float = Field(gt=0.0, description="decay time constant, seconds")
    ewma_tau_min: list[float] = Field(description="EWMA time constants, minutes")
    movement_pattern_prior_matches: float = Field(
        gt=0.0, description="Dirichlet prior weight toward the position matrix, matches"
    )


class CompetitionConfig(_Frozen):
    live_data_allowed: bool = Field(
        description="whether live-mode matches may be used (censored) in training"
    )


class AppConfig(_Frozen):
    host: str = Field(description="bind address; 127.0.0.1 unless LAN is explicitly enabled")
    port: int = Field(gt=0, le=65535, description="TCP port")
    roles: list[str] = Field(description="known role names for role-based views")


class StorageConfig(_Frozen):
    log_dir: str = Field(description="append-only log store root")
    data_dir: str = Field(description="local data root (never committed)")
    retention_days: int = Field(
        gt=0, description="match log partitions older than this are deleted by the retention job"
    )


class SeedsConfig(_Frozen):
    default: int = Field(description="default seed for every random process")


class Config(_Frozen):
    """Root configuration; mirrors ``configs/default.yaml`` exactly."""

    engine: EngineConfig
    prematch: PrematchConfig
    features: FeaturesConfig
    norms: NormsConfig
    quality: QualityConfig
    model: ModelConfig
    alerts: AlertsConfig
    evaluation: EvaluationConfig
    synth: SynthConfig
    research: ResearchConfig
    competition: CompetitionConfig
    app: AppConfig
    storage: StorageConfig
    seeds: SeedsConfig


def load_config(path: str | Path | None = None) -> Config:
    """Load and validate a YAML config.

    ``path`` defaults to the ``LIVRISK_CONFIG`` environment variable, then to
    ``configs/default.yaml``. Raises ``pydantic.ValidationError`` (a ``ValueError``)
    on unknown keys or out-of-range values.
    """
    if path is None:
        path = Path(os.environ.get("LIVRISK_CONFIG", DEFAULT_CONFIG_PATH))
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"config file did not parse to a mapping: {path}")
    return Config.model_validate(data)


def config_hash(cfg: Config) -> str:
    """Stable SHA-256 of the canonical JSON of ``cfg`` (sorted keys). Logged with every output."""
    payload = json.dumps(cfg.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
