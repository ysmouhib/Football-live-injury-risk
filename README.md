# live-injury-risk

Local, offline-first decision support for football bench staff: each player's injury risk for the
rest of the match, relative to his own normal, with the reason in plain words and the age and
quality of the data behind it.

**Intended purpose.** Research and decision support for qualified staff. It does not diagnose,
does not predict that any player will be injured, and does not recommend substitutions.

## Status

Version 0.1.0: specification complete, core tested.

| Part | Module | State |
|---|---|---|
| Maths core | `livrisk.riskmath` | done, tested |
| Markov chain tools | `livrisk.markov` | done, tested |
| Data contracts | `livrisk.contracts` | done, tested |
| User-facing wording | `livrisk.app.wording` | done, tested |
| M1 configuration | `livrisk.config` | done |
| M2 synthetic data | `livrisk.synth` | done |
| M3 stream features | `livrisk.features.stream` | done |
| M4 player and team norms | `livrisk.features.norms` | done |
| M5 pre-match bridge | `livrisk.model.prematch` | done |
| M6 live hazard model | `livrisk.model.live_hazard` | done |
| M7 online risk engine | `livrisk.engine` | done |
| M8 alerts and operating point | `livrisk.alerts` | done |
| M9 replay and log store | `livrisk.replay`, `livrisk.storage` | done |
| M10 evaluation, Gate C, sensitivity | `livrisk.evaluation` | done |
| M11 bench app | `livrisk.app` | done |
| M12 ingest adapters | `livrisk.ingest` | done |
| M13 video identity | `livrisk.video` | planned |
| M14 packaging | | planned |

Acceptance tests for planned parts already exist in `tests/` and carry the `pending` marker until
the part is built.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
make check
```

`make check` runs ruff, the format check, mypy (strict) and the tests that are not pending.

## Documentation

1. `docs/00_OVERVIEW.md` - product, users, scope, phases
2. `docs/01_EVIDENCE.md` - literature behind each design choice
3. `docs/02_MATH.md` - the model, every formula and reference value
4. `docs/03_DATA_CONTRACTS.md` - data shapes, units, coordinates, clocks
5. `docs/04_ARCHITECTURE.md` - modules, data flow, latency, failure modes
6. `docs/05_VALIDATION.md` - synthetic tests, retrospective protocol, Gate C
7. `docs/06_UI_SPEC.md` - bench screen, alerts, wording rules
8. `docs/07_RULES_ETHICS.md` - Law 4, GDPR, MDR, AI Act
9. `docs/DECISIONS.md` - design decisions

## Data

No personal data is stored in this repository. See `data/README.md`.
