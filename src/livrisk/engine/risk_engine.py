"""Online risk engine (docs/02_MATH.md §3, §5; docs/04_ARCHITECTURE.md §5-§7).

Every ``engine.risk_tick_s`` seconds of match time the risk task calls ``update`` with the
latest feature snapshots and the match state; the engine emits one ``RiskSnapshot`` per on-pitch
player: P15 (next ``engine.horizon_min`` minutes), R_rest (rest of the match), R_usual (the same
with the live multiplier removed), the bounded relative risk RR, interval percentiles from
posterior draws, quality, data age and the top reason code.

Clocks and the as_of rule
-------------------------
``MatchState`` carries the match clock (``period``, ``t_s``; docs/03 §1); the consumed
``FeatureSnapshot`` times are play-clock seconds (breaks removed; docs/DECISIONS.md D-016). The
risk task owns both clocks and the wall clock, and passes each player's ``data_age_s`` (age of
the newest used sample, seconds) explicitly, so the engine never mixes the two clocks. A value
for time ``t`` uses only feature data with timestamps <= ``t - delta``: the feature engine
enforces the blanking when it builds the snapshots; the risk engine never reads samples.

Intervals
---------
Per update and player, ``engine.interval_draws`` parameter vectors are drawn from the Laplace
posterior N(psi_hat, Sigma) with a deterministic seed per (match, player, tick); Sigma is exactly
singular (docs/02 §4), so draws go through ``live_hazard.laplace_factor``, never a Cholesky
factorisation. Per draw, P15 and R_rest are recomputed and the ``engine.interval_level``
percentiles reported. When the weekly model supplied [p7_lo, p7_hi], each draw also jitters
ln Lambda with the SD from ``riskmath.log_cumhaz_sd_from_interval`` (docs/02 §5). When the model
version ships the importance weights of a failed interval audit (docs/02 §5.1), the stored
proposal draws and weighted percentiles (``riskmath.weighted_quantile``) replace the Gaussian
draws. Norm uncertainty is not included (docs/02 §5).

Quality
-------
A fitted feature contributes z* = rho_k * clip(z) only when its raw value exists and transforms,
its window coverage reaches ``quality.min_coverage``, and, for tracking-derived features
(``stream.TRACKING_FEATURES``), the player's identity confidence reaches
``quality.min_identity``; otherwise z* = 0 (falls back to the usual state). Quality is ``stale``
when data_age_s exceeds ``quality.stale_hide_s`` (every z* is zeroed: a stale value is never
shown as live), ``prematch_only`` when the feature snapshot says so or no fitted feature is
valid, ``degraded`` when some fitted feature is invalid or the snapshot is degraded, else
``good``. Norm reasons (``norm_reset``, ``few_matches``) are appended to the enriched feature
snapshot's ``quality_reasons``; they do not change the quality level.
"""

from __future__ import annotations

import hashlib
import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from livrisk import riskmath
from livrisk.config import Config
from livrisk.contracts import (
    PERIOD_START_S,
    FeatureSnapshot,
    Mode,
    PositionGroup,
    PrematchBand,
    PreMatchRisk,
    Quality,
    RiskSnapshot,
)
from livrisk.features.norms import NormTable, z_for
from livrisk.features.stream import TRACKING_FEATURES
from livrisk.model.live_hazard import AuditReport, FittedHazard, laplace_factor
from livrisk.model.prematch import match_cumhaz, prematch_band, usual_risk

FloatArray = NDArray[np.float64]

logger = logging.getLogger(__name__)

#: Contribution threshold on the log-hazard scale for the top reason code (docs/06 §5).
_REASON_LOG_MIN: float = 0.1

#: Feature to reason code when its contribution dominates on the upside (docs/06 §5).
_REASON_HIGH: dict[str, str] = {
    "hsr_5m": "HSR_HIGH",
    "spr_5m": "SPR_HIGH",
    "recovery_10m": "RECOVERY_SHORT",
    "hir_drop_15m": "HIR_DROP",
    "nearmax_cum": "NEARMAX_MANY",
    "disrupt_5m": "DISRUPT",
}

#: Features whose dominant downside contribution reads as "load below his usual" (docs/06 §5).
_LOAD_BELOW_FEATURES: frozenset[str] = frozenset({"hsr_5m", "spr_5m", "nearmax_cum"})

#: Quality reason of a player whose own norm carries fewer effective matches than
#: ``norms.min_matches_for_own_norm`` (a key of app.wording.QUALITY_REASON_TEXT).
_FEW_MATCHES_REASON: str = "few_matches"

#: First beta slot in the parameter vector (alpha, then the 8 band levels, then beta).
_BETA0: int = 1 + riskmath.N_BANDS


@dataclass(frozen=True)
class MatchState:
    """Match state at one risk update: match clock (docs/03 §1) and announced stoppage.

    ``t_s`` is the continuous match clock in seconds (period starts 0 / 2700 / 5400 / 6300).
    ``stoppage_1h_min`` / ``stoppage_2h_min`` are the announced stoppage minutes of each half,
    None until announced; until then the projection plans with ``engine.default_stoppage_*``.
    """

    match_id: str
    period: int
    t_s: float
    stoppage_1h_min: float | None = None
    stoppage_2h_min: float | None = None


@dataclass(frozen=True)
class RiskUpdate:
    """Output of one ``RiskEngine.update``.

    ``risks``: one ``RiskSnapshot`` per on-pitch player (match clock, docs/03 §7). ``features``:
    the consumed snapshots, one per risk row in the same order, enriched with the
    player-relative z values and the norm quality reasons (``norm_reset``, ``few_matches``) that
    only the norm table can supply; the app reads the quality chip's reason from these.
    """

    risks: tuple[RiskSnapshot, ...]
    features: tuple[FeatureSnapshot, ...]


@dataclass(frozen=True)
class _PlayerBaseline:
    """Pre-match inputs of one player, resolved once at construction.

    ``cumhaz`` is Lambda, the cumulative hazard over 90 regular minutes (docs/02 §1);
    ``log_cumhaz_sd`` is the ln-Lambda SD implied by the weekly interval (0 without one);
    ``band`` is the pre-match display band from the 90-minute usual risk.
    """

    cumhaz: float
    log_cumhaz_sd: float
    band: PrematchBand


def _draw_seed(match_id: str, player_id: str, t_s: float, stream: str) -> int:
    """Deterministic 64-bit seed for the posterior draws of one player at one risk update."""
    key = f"{match_id}\0{player_id}\0{t_s:.3f}\0{stream}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "little")


def _band_counts(
    t_min: float, period: int, horizon_min: float | None, s1_min: float, s2_min: float
) -> FloatArray:
    """Minutes per time-profile band in the window from (``t_min``, ``period``).

    The schedule comes from ``riskmath.minute_schedule`` (the same one the point estimates use);
    the counts only vectorise the per-draw computation.
    """
    counts = np.zeros(riskmath.N_BANDS, dtype=np.float64)
    for t, p in riskmath.minute_schedule(t_min, period, horizon_min, s1_min, s2_min):
        counts[riskmath.band_index(t, p)] += 1.0
    return counts


class RiskEngine:
    """Combines pre-match baseline, time profile and live multiplier (docs/02_MATH.md §5).

    Construction resolves everything constant over the match: the Laplace factor of the fitted
    covariance, the per-player pre-match baselines (Lambda, its interval SD, the display band)
    and, when the model version ships them, the proposal draws and importance weights of a failed
    interval audit. ``update`` is deterministic given its inputs.
    """

    def __init__(
        self,
        cfg: Config,
        fitted: FittedHazard,
        norms: NormTable,
        prematch: Mapping[str, PreMatchRisk],
        positions: Mapping[str, PositionGroup],
        *,
        audit: AuditReport | None = None,
    ) -> None:
        n_par = len(fitted.param_names)
        if fitted.point.shape != (n_par,) or fitted.cov.shape != (n_par, n_par):
            raise ValueError(
                f"fitted hazard has {n_par} parameter names but point {fitted.point.shape} "
                f"and cov {fitted.cov.shape}"
            )
        if n_par != _BETA0 + len(fitted.feature_names):
            raise ValueError("param_names must be alpha, the 8 band levels, then the betas")
        self._cfg = cfg
        self._fitted = fitted
        self._norms = norms
        self._positions = dict(positions)
        engine = cfg.engine
        self._mode = Mode(engine.mode)
        self._rr_max = float(engine.rr_max)
        self._horizon_min = float(engine.horizon_min)
        self._n_draws = int(engine.interval_draws)
        self._s1_default = float(engine.default_stoppage_1h_min)
        self._s2_default = float(engine.default_stoppage_2h_min)
        tail = (1.0 - float(engine.interval_level)) / 2.0
        self._q_lo = tail
        self._q_hi = 1.0 - tail
        self._z_clip = float(cfg.features.z_clip)
        self._min_coverage = float(cfg.quality.min_coverage)
        self._min_identity = float(cfg.quality.min_identity)
        self._stale_hide_s = float(cfg.quality.stale_hide_s)
        self._few_matches_below = float(cfg.norms.min_matches_for_own_norm)
        rho: list[float] = []
        for name in fitted.feature_names:
            if name not in cfg.model.reliability:
                logger.warning("no model.reliability entry for feature %r; using 1.0", name)
            rho.append(float(cfg.model.reliability.get(name, 1.0)))
        self._reliability = np.asarray(rho, dtype=np.float64)
        self._point = np.asarray(fitted.point, dtype=np.float64)
        self._alpha = fitted.alpha
        self._beta = np.asarray(fitted.beta, dtype=np.float64)
        self._g_levels = fitted.g_levels()
        self._factor = laplace_factor(np.asarray(fitted.cov, dtype=np.float64))
        self._baseline: dict[str, _PlayerBaseline] = {}
        for pid, risk in prematch.items():
            cumhaz = match_cumhaz(risk, float(cfg.prematch.match_share_prior))
            log_sd = 0.0
            if risk.p7_lo is not None and risk.p7_hi is not None:
                log_sd = riskmath.log_cumhaz_sd_from_interval(
                    risk.p7_lo, risk.p7_hi, risk.interval_level
                )
            self._baseline[pid] = _PlayerBaseline(
                cumhaz=cumhaz,
                log_cumhaz_sd=log_sd,
                band=prematch_band(
                    usual_risk(cumhaz),
                    float(cfg.prematch.band_cutoffs.watch),
                    float(cfg.prematch.band_cutoffs.high),
                ),
            )
        self._audit_draws: FloatArray | None = None
        self._audit_weights: FloatArray | None = None
        if audit is not None and not audit.passed:
            draws = np.asarray(audit.proposal_draws, dtype=np.float64)
            weights = np.asarray(audit.weights, dtype=np.float64)
            if draws.ndim != 2 or draws.shape[1] != n_par:
                raise ValueError(
                    f"audit proposal draws have shape {draws.shape}, expected (*, {n_par})"
                )
            if weights.shape != (draws.shape[0],) or bool(np.any(weights < 0.0)):
                raise ValueError("audit weights must be non-negative, one per proposal draw")
            total = float(weights.sum())
            if total <= 0.0:
                raise ValueError("audit weights must have a positive sum")
            self._audit_draws = draws
            self._audit_weights = weights / total
        self._warned: set[str] = set()

    def update(
        self,
        snapshots: Sequence[FeatureSnapshot],
        state: MatchState,
        data_age_s: Mapping[str, float],
    ) -> RiskUpdate:
        """One risk update: one ``RiskSnapshot`` per on-pitch player with a pre-match row.

        ``snapshots`` are the latest feature snapshots (play-clock times, D-016); ``state`` is
        the match clock of this update; ``data_age_s`` carries the age of the newest used sample
        per player (seconds) and must cover every snapshotted player. Players without a
        pre-match row or a known position are skipped (a warning is logged once per player);
        goalkeepers are skipped unless ``engine.include_goalkeepers``. A value for time ``t``
        uses only data with timestamps <= ``t - delta`` (the feature engine enforces the
        blanking; the risk engine never reads samples). Deterministic given the inputs.
        """
        if state.period not in PERIOD_START_S or state.t_s < PERIOD_START_S[state.period]:
            raise ValueError(f"invalid match state: period {state.period}, t_s {state.t_s}")
        t_min = state.t_s / 60.0
        band_now = riskmath.band_index(t_min, state.period)
        s1 = self._s1_default if state.stoppage_1h_min is None else float(state.stoppage_1h_min)
        s2 = self._s2_default if state.stoppage_2h_min is None else float(state.stoppage_2h_min)
        counts_p15 = _band_counts(t_min, state.period, self._horizon_min, s1, s2)
        counts_rest = _band_counts(t_min, state.period, None, s1, s2)
        missing_age = sorted({s.player_id for s in snapshots}.difference(data_age_s))
        if missing_age:
            raise ValueError(f"data_age_s is missing players: {missing_age}")
        risks: list[RiskSnapshot] = []
        enriched: list[FeatureSnapshot] = []
        for snap in sorted(snapshots, key=lambda s: s.player_id):
            if snap.match_id != state.match_id:
                raise ValueError(
                    f"snapshot for match {snap.match_id!r} in update for {state.match_id!r}"
                )
            pid = snap.player_id
            position = self._positions.get(pid)
            baseline = self._baseline.get(pid)
            if position is None or baseline is None:
                self._warn_once(pid, "no pre-match row or position; skipped")
                continue
            if position == PositionGroup.GK and not self._cfg.engine.include_goalkeepers:
                continue
            row, full = self._player_update(
                snap,
                baseline,
                position,
                band_now,
                float(data_age_s[pid]),
                state,
                t_min,
                s1,
                s2,
                counts_p15,
                counts_rest,
            )
            risks.append(row)
            enriched.append(full)
        return RiskUpdate(risks=tuple(risks), features=tuple(enriched))

    def _player_update(
        self,
        snap: FeatureSnapshot,
        baseline: _PlayerBaseline,
        position: PositionGroup,
        band_now: int,
        data_age_s: float,
        state: MatchState,
        t_min: float,
        s1_min: float,
        s2_min: float,
        counts_p15: FloatArray,
        counts_rest: FloatArray,
    ) -> tuple[RiskSnapshot, FeatureSnapshot]:
        """The risk row and the enriched feature snapshot of one player at one update."""
        stale = data_age_s > self._stale_hide_s or snap.quality == Quality.STALE
        names = self._fitted.feature_names
        z_shown: dict[str, float | None] = {}
        z_gated: list[float] = []
        reasons = list(snap.quality_reasons)
        n_valid = 0
        for name in names:
            raw = snap.features.get(name)
            z: float | None = None
            if raw is not None and math.isfinite(raw):
                z = z_for(self._norms, snap.player_id, position, band_now, name, raw, self._cfg)
            z_shown[name] = z
            valid = (
                not stale
                and z is not None
                and snap.coverage.get(name, 0.0) >= self._min_coverage
                and (
                    name not in TRACKING_FEATURES or snap.identity_confidence >= self._min_identity
                )
            )
            if valid and z is not None:
                n_valid += 1
                z_gated.append(z)
            else:
                z_gated.append(math.nan)
            self._norm_reasons(snap.player_id, position, band_now, name, reasons)
        z_star = riskmath.shrink_feature(
            np.asarray(z_gated, dtype=np.float64), self._reliability, self._z_clip
        )
        quality = self._quality(snap.quality, stale, n_valid)
        rr = (
            1.0
            if self._mode == Mode.MONITOR_ONLY
            else (riskmath.relative_risk(self._beta, z_star, self._rr_max))
        )
        point_cumhaz = baseline.cumhaz * math.exp(self._alpha)
        p15 = riskmath.projected_probability(
            point_cumhaz, self._g_levels, rr, t_min, state.period, self._horizon_min, s1_min, s2_min
        )
        r_rest = riskmath.projected_probability(
            point_cumhaz, self._g_levels, rr, t_min, state.period, None, s1_min, s2_min
        )
        r_usual = riskmath.projected_probability(
            point_cumhaz, self._g_levels, 1.0, t_min, state.period, None, s1_min, s2_min
        )
        p15_lo, p15_hi, rest_lo, rest_hi = self._intervals(
            snap.player_id, baseline, z_star, state, counts_p15, counts_rest
        )
        row = RiskSnapshot(
            match_id=state.match_id,
            player_id=snap.player_id,
            period=state.period,
            as_of_t_s=float(state.t_s),
            data_age_s=data_age_s,
            quality=quality,
            mode=self._mode,
            p15=p15,
            p15_lo=min(p15_lo, p15),
            p15_hi=max(p15_hi, p15),
            r_rest=r_rest,
            r_rest_lo=min(rest_lo, r_rest),
            r_rest_hi=max(rest_hi, r_rest),
            r_usual=r_usual,
            rr=rr,
            prematch_band=baseline.band,
            top_reason_code=self._reason_code(quality, z_star, band_now),
            model_version=self._fitted.version,
        )
        full = snap.model_copy(update={"z": z_shown, "quality_reasons": tuple(reasons)})
        return row, full

    def _quality(self, snap_quality: Quality, stale: bool, n_valid: int) -> Quality:
        """Output quality from staleness, the snapshot's own quality and the valid count."""
        if stale:
            return Quality.STALE
        if n_valid == 0 or snap_quality == Quality.PREMATCH_ONLY:
            return Quality.PREMATCH_ONLY
        if n_valid < len(self._fitted.feature_names) or snap_quality == Quality.DEGRADED:
            return Quality.DEGRADED
        return Quality.GOOD

    def _norm_reasons(
        self,
        player_id: str,
        position: PositionGroup,
        band: int,
        feature: str,
        reasons: list[str],
    ) -> None:
        """Append the norm quality reasons (norm_reset, few_matches) of one feature lookup."""
        try:
            entry = self._norms.get(player_id, position, band, feature)
        except KeyError:
            return
        for reason in entry.quality_reasons:
            if reason not in reasons:
                reasons.append(reason)
        if entry.n_effective < self._few_matches_below and _FEW_MATCHES_REASON not in reasons:
            reasons.append(_FEW_MATCHES_REASON)

    def _draws_for(
        self, match_id: str, player_id: str, t_s: float
    ) -> tuple[FloatArray, FloatArray | None]:
        """Posterior parameter draws (S x P) and, for the audit fallback, importance weights.

        Gaussian draws come from the Laplace factor with a seed per (match, player, tick); a
        model version that failed its interval audit ships fixed proposal draws and weights.
        """
        if self._audit_draws is not None:
            return self._audit_draws, self._audit_weights
        rng = np.random.default_rng(_draw_seed(match_id, player_id, t_s, "params"))
        z = rng.standard_normal((self._n_draws, self._factor.shape[1]))
        return np.asarray(self._point[None, :] + z @ self._factor.T, dtype=np.float64), None

    def _intervals(
        self,
        player_id: str,
        baseline: _PlayerBaseline,
        z_star: FloatArray,
        state: MatchState,
        counts_p15: FloatArray,
        counts_rest: FloatArray,
    ) -> tuple[float, float, float, float]:
        """Interval percentiles (P15 lo/hi, R_rest lo/hi) from the posterior draws.

        Both windows reuse the same parameter draws, so the two intervals are consistent. The
        pre-match interval enters as an independent ln-Lambda jitter per draw; the rr_max bound
        guards the headline RR only, not the draws. Vectorised form of
        ``riskmath.projected_probability``: cumhaz = Lambda * exp(alpha) * (counts . g) / 90 * RR
        per draw, probabilities via ``riskmath.prob_from_cumhaz`` (as in the audit battery).
        """
        draws, weights = self._draws_for(state.match_id, player_id, state.t_s)
        if self._mode == Mode.MONITOR_ONLY:
            draws = np.array(draws, copy=True)
            draws[:, _BETA0:] = 0.0
        rr_draws = np.exp(draws[:, _BETA0:] @ z_star)
        log_lam = math.log(baseline.cumhaz) + draws[:, 0]
        if baseline.log_cumhaz_sd > 0.0:
            rng = np.random.default_rng(_draw_seed(state.match_id, player_id, state.t_s, "cumhaz"))
            log_lam = log_lam + rng.standard_normal(draws.shape[0]) * baseline.log_cumhaz_sd
        lam = np.exp(log_lam)
        g = np.exp(draws[:, 1:_BETA0])
        out: list[float] = []
        for counts in (counts_p15, counts_rest):
            cumhaz = lam * (g @ counts) / riskmath.REGULAR_MINUTES * rr_draws
            probs = -np.expm1(-cumhaz)
            if weights is None:
                qs = np.quantile(probs, [self._q_lo, self._q_hi])
                out.extend((float(qs[0]), float(qs[1])))
            else:
                out.extend(
                    (
                        riskmath.weighted_quantile(probs, weights, self._q_lo),
                        riskmath.weighted_quantile(probs, weights, self._q_hi),
                    )
                )
        return out[0], out[1], out[2], out[3]

    def _reason_code(self, quality: Quality, z_star: FloatArray, band_now: int) -> str:
        """Top reason code (docs/06 §5): the dominant contribution on the log-hazard scale.

        The contribution of feature k is beta_k * z*_k (z*_k alone in monitor_only, where no
        risk claim is made). The largest positive contribution above ``_REASON_LOG_MIN`` wins;
        without one, an elevated time profile of the current band explains the row, then a
        clearly low load feature reads as LOAD_BELOW, else WITHIN_NORMAL. Prematch-only and
        stale rows report DATA_LOST.
        """
        if quality in (Quality.PREMATCH_ONLY, Quality.STALE):
            return "DATA_LOST"
        signal = z_star if self._mode == Mode.MONITOR_ONLY else self._beta * z_star
        best = _REASON_LOG_MIN
        best_code: str | None = None
        worst = -_REASON_LOG_MIN
        worst_name: str | None = None
        for j, name in enumerate(self._fitted.feature_names):
            code = _REASON_HIGH.get(name)
            if code is None:
                continue
            contrib = float(signal[j])
            if contrib > best:
                best, best_code = contrib, code
            if contrib < worst:
                worst, worst_name = contrib, name
        if best_code is not None:
            return best_code
        if math.log(float(self._g_levels[band_now])) > _REASON_LOG_MIN:
            return "TIME_PROFILE"
        if worst_name is not None and worst_name in _LOAD_BELOW_FEATURES:
            return "LOAD_BELOW"
        return "WITHIN_NORMAL"

    def _warn_once(self, player_id: str, message: str) -> None:
        """Log a warning the first time it applies to a player (match-long state)."""
        if player_id not in self._warned:
            self._warned.add(player_id)
            logger.warning("player %s: %s", player_id, message)
