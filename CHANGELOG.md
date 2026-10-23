# Changelog

## Unreleased

- Typed, validated configuration loading with a stable config hash (M1).
- Synthetic data generator: person-minute fast path with known hazard, AR(1) live state and
  informative substitution censoring; full-match path with a band-dependent semi-Markov speed
  process, tracking, events, lineups, pre-match risks and injury labels (M2).
- Stream feature engine with quality flags (M3).
- Player, position and team norms with recency weighting, shrinkage and a reset rule (M4).
- Pre-match bridge: weekly risk CSV to a match baseline hazard, usual-risk bands (M5).
- Live hazard model: MAP fit with Laplace covariance and importance-sampling interval audit (M6).
- Online risk engine: posterior-draw intervals, quality rules and monitor-only mode (M7).
- Alert state machine with budget and operating-point analysis (M8).
- Replay simulator with latency, dropout, burst and drift injection, a look-ahead guard on
  feature reads, and an append-only Parquet log store (M9).
- Evaluation harness and Gate C report: time and club splits, spell-level metrics with
  player-cluster bootstrap, benchmarks, decision curves, diagnostics, Holm-adjusted
  exploratory comparators and a model card (M10).

## 0.1.0 - 2026-10-22

- Specification (docs 00-07) and design decisions D-001 to D-008.
- Maths core (`riskmath`), data contracts and user-facing wording, with tests.
- Typed module skeleton and pending acceptance tests for M1-M14.
- Markov chain tools (`markov`): ergodicity, stationary law, mixing time, semi-Markov occupancy.
- Specification of the semi-Markov synthetic speed process, recency-weighted norms with a reset
  rule, the interval audit, substitution diagnostics, the alert operating point, the multiplicity
  rule and the training-data policy (D-009 to D-015).
- Optional p7 interval and quality reasons in the data contracts.
