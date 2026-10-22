"""Player, position and team norms (docs/02_MATH.md §3).

Norms are built only from matches strictly before ``as_of_date``. Past matches are weighted by
``riskmath.recency_weights``; shrinkage (``riskmath.shrink_mean``, ``riskmath.shrink_sd``) uses
the Kish effective number of matches, never the number of windows. The spread uses
non-overlapping windows (``norms.spread_windows_per_band`` per band and match).

Reset state
-----------
The reset rule (docs/02_MATH.md §3) needs per-player inputs the history frame does not carry:
``days_since_return`` and ``days_since_last_match`` come from the weekly input (the
``PreMatchRisk`` fields, docs/03 §5), and ``matches_since_return`` is maintained by the caller
(0 on the first match back). They reach the table as a ``reset_inputs`` mapping on
``build_norms`` (docs/DECISIONS.md D-017), so ``get`` and ``z_for`` need no extra arguments.
While a reset is active the player's own ``n_effective`` is capped at
``norms.reset_own_weight_matches`` (the norm is mostly the position norm) and entries carry
``reset=True``, surfaced downstream as the quality reason ``norm_reset``
(``NormEntry.quality_reasons``); a position-norm fallback for a player in reset carries it too.

When a position has no usable spread (``s_pos`` zero or unknown), entries carry ``sd = NaN``
and ``z_for`` returns None: the feature falls back to the usual state downstream.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from livrisk import markov, riskmath
from livrisk.config import Config
from livrisk.contracts import PositionGroup

FloatArray = NDArray[np.float64]

logger = logging.getLogger(__name__)

#: Quality reason code of an active norm reset (a key of app.wording.QUALITY_REASON_TEXT).
NORM_RESET_REASON: str = "norm_reset"


@dataclass(frozen=True)
class NormEntry:
    """Shrunk norm for one player x band x feature, on the transformed scale.

    ``mean``/``sd`` live on the transformed scale of the feature (log1p, log or identity);
    ``sd`` is NaN when no spread could be estimated. ``n_matches`` counts the player's past
    matches behind the entry (0 for a pure position fallback).
    """

    mean: float
    sd: float
    n_matches: int
    #: Kish effective number of matches after recency weighting (and any reset cap).
    n_effective: float
    #: True while a norm reset is active (docs/02_MATH.md §3); quality reason ``norm_reset``.
    reset: bool = False

    @property
    def quality_reasons(self) -> tuple[str, ...]:
        """Quality reason codes of outputs using this entry: ``norm_reset`` during a reset."""
        return (NORM_RESET_REASON,) if self.reset else ()


@dataclass(frozen=True)
class ResetInputs:
    """Per-player inputs of the norm reset rule (docs/02_MATH.md §3; docs/03 §5).

    ``days_since_return`` is set when the last time-loss absence lasted at least
    ``norms.reset_after_absence_days`` days; ``days_since_last_match`` needs no absence.
    ``matches_since_return`` counts the matches played since the return (0 on the first match
    back); the caller maintains it, e.g. from its own match log.
    """

    days_since_return: float | None = None
    days_since_last_match: float | None = None
    matches_since_return: int = 0


@dataclass(frozen=True)
class _MatchAgg:
    """One past match's contribution to a player x band x feature norm (transformed scale)."""

    player_id: str
    position: PositionGroup
    band: int
    feature: str
    weight: float
    mean: float
    chunk_means: tuple[float, ...]


class NormTable:
    """Lookup of shrunk norms; falls back to the position norm when the player has no history.

    ``get`` of a player without history for the band and feature returns the pure position norm
    (``n_matches`` and ``n_effective`` 0; ``reset`` still set when the player is in a reset).
    Raises ``KeyError`` when not even the position norm exists (no history for that position,
    band and feature). Built by ``build_norms``; treated as immutable.
    """

    def __init__(
        self,
        entries: Mapping[tuple[str, int, str], NormEntry],
        position_entries: Mapping[tuple[PositionGroup, int, str], NormEntry],
        reset_players: frozenset[str] = frozenset(),
    ) -> None:
        self._entries = dict(entries)
        self._position = dict(position_entries)
        self._reset = frozenset(reset_players)

    def get(self, player_id: str, position: PositionGroup, band: int, feature: str) -> NormEntry:
        """The player's shrunk norm for (band, feature), or the pure position norm."""
        entry = self._entries.get((player_id, band, feature))
        if entry is not None:
            return entry
        pos = self._position.get((position, band, feature))
        if pos is None:
            raise KeyError(f"no norm for position {position}, band {band}, feature {feature!r}")
        return NormEntry(
            mean=pos.mean,
            sd=pos.sd,
            n_matches=0,
            n_effective=0.0,
            reset=player_id in self._reset,
        )


def _transform(feature: str, value: float, cfg: Config) -> float | None:
    """The configured per-feature transform (``norms.log1p_transform`` / ``norms.log_transform``,
    else identity). None when the result is not finite (e.g. log of a non-positive value)."""
    nc = cfg.norms
    if feature in nc.log1p_transform:
        out = math.log1p(value) if value > -1.0 else math.nan
    elif feature in nc.log_transform:
        out = math.log(value) if value > 0.0 else math.nan
    else:
        out = value
    return out if math.isfinite(out) else None


def _aggregate(
    df: pd.DataFrame, positions: dict[str, PositionGroup], cfg: Config
) -> list[_MatchAgg]:
    """Per-match aggregation of a date-filtered, transformed history frame.

    ``df`` carries the transformed values in the ``_tval`` column (finite only). One record per
    player x band x feature x match date: the per-match mean of the transformed values and the
    means of ``norms.spread_windows_per_band`` contiguous, near-equal chunks of the match's rows
    in that band (frame order; empty chunks skipped). Windows never count as matches: each
    record carries the single recency weight of its match, where the match age (0 = most
    recent) is the player's rank of the date among his past match dates.
    """
    pid = df["player_id"].astype(str).to_numpy()
    feat = df["feature"].astype(str).to_numpy()
    band_f = pd.to_numeric(df["band"]).to_numpy(dtype=np.float64, na_value=np.nan)
    if not bool(np.all(np.isfinite(band_f))) or not bool(np.all(band_f == np.round(band_f))):
        raise ValueError("history.band must contain integer band indices")
    band = band_f.astype(np.int64)
    dates = pd.to_datetime(df["match_date"]).dt.date.to_numpy()
    tval = df["_tval"].to_numpy(dtype=np.float64)

    unknown = sorted({str(p) for p in pid} - set(positions))
    for p in unknown:
        logger.warning("norms: player %s is not in the positions map; his rows are ignored", p)

    dates_by_player: dict[str, set[date]] = {}
    for p, d in zip(pid, dates, strict=True):
        dates_by_player.setdefault(str(p), set()).add(d)
    age_of: dict[tuple[str, date], int] = {}
    for p, dset in dates_by_player.items():
        for age, d in enumerate(sorted(dset, reverse=True)):
            age_of[(p, d)] = age

    groups: dict[tuple[str, int, str, date], list[float]] = {}
    for i in range(len(df)):
        key = (str(pid[i]), int(band[i]), str(feat[i]), dates[i])
        bucket = groups.get(key)
        if bucket is None:
            groups[key] = [float(tval[i])]
        else:
            bucket.append(float(tval[i]))

    half_life = float(cfg.norms.recency_half_life_matches)
    n_windows = int(cfg.norms.spread_windows_per_band)
    out: list[_MatchAgg] = []
    for (p, b, f, d), vals in groups.items():
        position = positions.get(p)
        if position is None:
            continue
        arr = np.asarray(vals, dtype=np.float64)
        weight = float(riskmath.recency_weights([float(age_of[(p, d)])], half_life)[0])
        chunks = [c for c in np.array_split(arr, n_windows) if c.size > 0]
        out.append(
            _MatchAgg(
                player_id=p,
                position=position,
                band=b,
                feature=f,
                weight=weight,
                mean=float(arr.mean()),
                chunk_means=tuple(float(c.mean()) for c in chunks),
            )
        )
    return out


def _pooled(
    group: list[_MatchAgg],
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Weighted per-match means and per-window values of a group of match records.

    Returns (match means, match weights, window means, window weights); a window carries the
    recency weight of its match.
    """
    means = np.asarray([a.mean for a in group], dtype=np.float64)
    weights = np.asarray([a.weight for a in group], dtype=np.float64)
    chunks = np.asarray([c for a in group for c in a.chunk_means], dtype=np.float64)
    chunk_weights = np.asarray([a.weight for a in group for _ in a.chunk_means], dtype=np.float64)
    return means, weights, chunks, chunk_weights


def build_norms(
    history: pd.DataFrame,
    positions: dict[str, PositionGroup],
    as_of_date: date,
    cfg: Config,
    reset_inputs: Mapping[str, ResetInputs] | None = None,
) -> NormTable:
    """``history`` columns: player_id, match_date, band, feature, value (raw).

    Rows with ``match_date >= as_of_date`` are ignored (no look-ahead), as are rows whose
    transformed value is not finite and rows of players missing from ``positions`` (a warning
    is logged for those). Raw values are transformed per config (``norms.log1p_transform``,
    ``norms.log_transform``; else identity) before any aggregation. ``reset_inputs`` carries
    the per-player reset state (see the module docstring); players absent from it are not in
    reset.
    """
    nc = cfg.norms
    required = {"player_id", "match_date", "band", "feature", "value"}
    missing = required.difference(history.columns)
    if missing:
        raise ValueError(f"history is missing columns: {sorted(missing)}")

    match_ts = pd.to_datetime(history["match_date"])
    df = history.loc[(match_ts < pd.Timestamp(as_of_date)).to_numpy()].copy()
    if df.shape[0] == 0:
        return NormTable({}, {})

    raw = pd.to_numeric(df["value"], errors="coerce").to_numpy(dtype=np.float64, na_value=np.nan)
    tval = raw.copy()
    with np.errstate(invalid="ignore", divide="ignore"):
        is_log1p = df["feature"].isin(list(nc.log1p_transform)).to_numpy()
        tval[is_log1p] = np.log1p(raw[is_log1p])
        is_log = df["feature"].isin(list(nc.log_transform)).to_numpy()
        tval[is_log] = np.log(raw[is_log])
    df["_tval"] = tval
    df = df.loc[np.isfinite(tval)]
    if df.shape[0] == 0:
        return NormTable({}, {})

    aggs = _aggregate(df, positions, cfg)

    player_groups: dict[tuple[str, int, str], list[_MatchAgg]] = {}
    pos_groups: dict[tuple[PositionGroup, int, str], list[_MatchAgg]] = {}
    for agg in aggs:
        player_groups.setdefault((agg.player_id, agg.band, agg.feature), []).append(agg)
        pos_groups.setdefault((agg.position, agg.band, agg.feature), []).append(agg)

    pos_entries: dict[tuple[PositionGroup, int, str], NormEntry] = {}
    for key, group in pos_groups.items():
        means, weights, chunks, chunk_weights = _pooled(group)
        mu, _ = riskmath.weighted_mean_sd(means, weights)
        _, s_pos = riskmath.weighted_mean_sd(chunks, chunk_weights)
        pos_entries[key] = NormEntry(
            mean=mu, sd=s_pos, n_matches=len(group), n_effective=riskmath.kish_effective_n(weights)
        )

    reset_players = frozenset(
        p
        for p, ri in (reset_inputs or {}).items()
        if reset_active(
            ri.days_since_return, ri.days_since_last_match, ri.matches_since_return, cfg
        )
    )

    kappa = float(nc.kappa_matches)
    entries: dict[tuple[str, int, str], NormEntry] = {}
    for (p, b, f), group in player_groups.items():
        pos_entry = pos_entries[(positions[p], b, f)]
        means, weights, chunks, chunk_weights = _pooled(group)
        xw, _ = riskmath.weighted_mean_sd(means, weights)
        _, sw = riskmath.weighted_mean_sd(chunks, chunk_weights)
        n_eff = riskmath.kish_effective_n(weights)
        reset = p in reset_players
        if reset:
            n_eff = min(n_eff, float(nc.reset_own_weight_matches))
        entries[(p, b, f)] = NormEntry(
            mean=riskmath.shrink_mean(n_eff, xw, kappa, pos_entry.mean),
            sd=riskmath.shrink_sd(n_eff, sw, kappa, pos_entry.sd)
            if pos_entry.sd > 0.0
            else math.nan,
            n_matches=len(group),
            n_effective=n_eff,
            reset=reset,
        )
    return NormTable(entries, pos_entries, reset_players)


def reset_active(
    days_since_return: float | None,
    days_since_last_match: float | None,
    matches_since_return: int,
    cfg: Config,
) -> bool:
    """Norm reset rule: after >= ``reset_after_absence_days`` lost to injury, or more than
    ``reset_after_no_match_days`` without a match, for the next ``reset_matches`` matches."""
    nc = cfg.norms
    if matches_since_return >= nc.reset_matches:
        return False
    if days_since_return is not None and days_since_return >= nc.reset_after_absence_days:
        return True
    return (
        days_since_last_match is not None and days_since_last_match > nc.reset_after_no_match_days
    )


def occupancy_drift(recent: Sequence[float], norm: Sequence[float]) -> float:
    """Total-variation distance between recent and usual speed-zone occupancy (diagnostic only;
    use ``markov.tv_distance``). ``recent`` and ``norm`` are occupancy shares summing to 1
    (e.g. the player's last 3 matches vs his norm); logged downstream, never gating."""
    return markov.tv_distance(
        np.asarray(recent, dtype=np.float64), np.asarray(norm, dtype=np.float64)
    )


def z_for(
    table: NormTable,
    player_id: str,
    position: PositionGroup,
    band: int,
    feature: str,
    raw_value: float | None,
    cfg: Config,
) -> float | None:
    """Transform per config (log1p / log / none), then z against the shrunk norm.

    None when the value is missing or non-finite, when its transform is undefined, or when no
    usable norm exists (no spread, or no position norm at all); the feature then falls back to
    the usual state downstream. The entry's ``quality_reasons`` (e.g. ``norm_reset``) are the
    caller's to surface.
    """
    if raw_value is None or not math.isfinite(raw_value):
        return None
    try:
        entry = table.get(player_id, position, band, feature)
    except KeyError:
        return None
    x = _transform(feature, raw_value, cfg)
    if x is None:
        return None
    z = riskmath.zscore(x, entry.mean, entry.sd)
    return None if math.isnan(z) else z


def team_relative(z_by_player: dict[str, float | None]) -> dict[str, float | None]:
    """z_i minus the mean z of the other on-pitch team-mates (docs/02_MATH.md §3).

    Missing values (None) stay None and never enter the mean. When no team-mate has a value the
    team reference cannot be formed and the player keeps his own z.
    """
    valued = {p: z for p, z in z_by_player.items() if z is not None}
    total = math.fsum(valued.values())
    n = len(valued)
    out: dict[str, float | None] = {}
    for pid, z in z_by_player.items():
        if z is None:
            out[pid] = None
        elif n <= 1:
            out[pid] = z
        else:
            out[pid] = z - (total - z) / (n - 1)
    return out
