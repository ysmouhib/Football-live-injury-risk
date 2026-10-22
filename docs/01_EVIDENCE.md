# 01: Evidence behind the design

Read this before changing any modelling choice. Each design decision below traces to a source.
Effects from small case-control studies are probably overstated (see §2), so the live layer is
estimated with shrinkage toward "no effect" and must earn its place at Gate C.

## 1. What the literature supports

| Input | Evidence | Strength | Design consequence |
| --- | --- | --- | --- |
| Pre-match risk (injury history, age, playing time, congestion) | Injury history is the strongest known predictor (Green et al., BJSM 2020: recent hamstring injury RR 4.8). A league-scale public-data model ranks risk with ROC AUC ≈ 0.66 on an unseen season. | Strong | Baseline hazard per player (docs/02_MATH.md §2) |
| Time point in the match | Highest risk in first and last 15 min (EPL, BJSM 2002); last 15 min of first half (World Cups, Ryynänen et al.; LaLiga hamstrings, Moreno-Perez et al. 2024) | Moderate, consistent | Piecewise-constant time profile g(t), estimated |
| Congestion × match time | Last 15 min of 2nd match in a 2-match cycle: IRR 3.1; final match of a 3-match cycle: IRR 2.0 (Carling et al., BJSM 2016) | Moderate (one club) | Allow g(t) to differ in congested periods |
| High-speed running in last 5 min | 51 vs 29 m above 21 km/h before 44 hamstring injuries; > 30 m gave OR 7.1; no difference over 15 min (Moreno-Perez et al. 2024) | Weak–moderate; window contains the injuring sprint | Candidate live feature, measured strictly before t − δ |
| Sprint distance in last minute | +11 m in 1 min → OR 1.22 (Gregson et al., JSAMS 2019, 29 players) | Weak, exploratory | Candidate |
| Recovery between high-intensity efforts | 36 s vs 99 s before injuring effort (Carling et al. 2010, 10 injuries) | Weak | Candidate |
| Fatigue: drop in high-intensity running | Normal: −35 to −45 % last vs first 15 min; −12 % in 5 min after the most intense spell (Mohr et al. 2003). Link to injury unknown | Pattern established; injury link unknown | Candidate: player's drop vs his usual drop, and vs team drop |
| Near-max sprints | Injuring sprints at 85 % vs 81 % of season max (Miralles-Iborra et al., preprint 2026); 88 % (Aiello et al. 2023) | Describes the injuring action | Count as exposure; never an alert trigger on its own |
| Disruptive events (cards, goals, injuries) | 76.7 vs 54.0 injuries per 1,000 h in the 5 min after (Ryynänen et al.) | Moderate | Candidate, from match-event feed |

## 2. The leakage trap that shapes the maths

Aiello et al. (2023) found that the run in which a hamstring injury happens covers a median 17 m
at about 29 km/h. That run sits inside every "5 minutes before injury" window, so it explains much
of the published 22 m gap in high-speed running. Therefore:
- Features for time t use data only up to t − δ (δ ≥ uncertainty of the injury-minute label).
- Horizons are ≥ 5 minutes; the main horizon is 15 minutes.
- Evaluation always compares against a model *without* live features on the same rows.

## 3. What prediction can achieve (calibrate expectations)

| Study | Validation | Result |
| --- | --- | --- |
| Huth et al. 2025 (FC Barcelona women; medRxiv, published 2026) | Held-out season, weekly retraining | ROC AUC 0.74; average precision 0.04 (6.9× chance); +20 match-weighted availability days at 50 % certainty |
| Hecksteden et al. 2023 (88 players, 51 injuries) | Holdout | ROC AUC 0.62 |
| League-scale public-data model (repo ysmouhib/injury-ml-research) | Pre-registered one-shot holdout | 28-day PR-AUC 0.158 at 9.1 % prevalence; ROC AUC 0.66 |
| Leckey et al., BJSM 2025 (38 studies) | Review | Trees best in 60 %; logistic regression won 4/12 head-to-heads |
| Zone7 vendor study (11 teams, 423 injuries) | Vendor-written | 72 % flagged 1–7 days ahead; no precision reported |
| Rossi et al. 2018, PLoS ONE (one Italian professional club, one season, training injuries) | Internal, no external validation | Recall 0.80, precision 0.50; an unreplicated result, not a ceiling to expect |

Expert consensus (Bahr 2016; FSI Talk 2025 with C. Carling and C. Barnes; Tondut et al. 2023):
individual injuries are not predictable; risk can be ranked and used for decision support.

## 4. Reading the match live

| Source | Evidence | Consequence |
| --- | --- | --- |
| Stadium optical tracking | TRACAB Gen5 data deliverable within 150 ms (SVG 2020); even optical tracking agrees poorly with GPS on accelerations > 4 m/s² (ICC 0.57; Sensors 2025) | Exclude hard accelerations from main features |
| Broadcast video | Players detected 36–64 % of the time on broadcast feeds vs 89–96 % on tactical feed (Mills et al. 2025); TV shows < 50 % of player activity (Sci Med Football 2025); sprint distance error ~31 % in simulation (Mortensen & Bornn 2020) | Broadcast is fallback only; every feature carries a visible-share quality flag |
| Video pipelines | SoccerNet game-state winner: GS-HOTA 63.8 on 30 s clips at 5 m tolerance, up to 80 fps on a laptop GPU, models 0.3–35.5 M parameters (Golovkin et al. 2025) | Small local models suffice; full-match identity is the open problem |
| Wearables live | Vendor apps stream live and support per-player threshold alerts (STATSports Sonra) | Primary live source for own players where competition allows |
| Stream delay | TV over the air ≈ 11.5 s; streaming 26–78 s behind live (SVG 2025) | Not usable as primary live source |

## 5. Human factors

Clinical decision-support alerts are overridden 46–96 % of the time when too frequent or
irrelevant (Poly et al., JMIR Med Inform 2020). Hence: two alert levels only, persistence and
hysteresis, an alert budget per match, a reason in words for every alert, and periodic review.

## 6. Ideas adopted from public patents (methods only, no code copied)

- Team-relative live features: compare a player's drop with the team's drop in the same minutes
  (IBM, US 10,953,280).
- Short video clips attached to alerts (same).
- Pre-match load advice as an optimisation: smallest cut keeping risk under threshold (Inplay,
  US 2021/0295184).
- "Players like him" nearest-neighbour baseline (Intel, US 11,049,031 - listed as expired).

## 7. Markov methods: what the evidence supports

| Study | What it shows | Used for |
| --- | --- | --- |
| Tabben et al., Biol Sport 2026 (1,258 players, 4,700 injuries, Qatar Stars League) | A basic Markov model of index to subsequent injury: 34 % of injuries were subsequent; hamstring recurrence within a season 7.5 %; groin then hamstring 2.9 % | Norm reset after absence; season-level availability (backlog) |
| Rojas-Salazar et al., arXiv 2010.10739 (soccer referee, 1 Hz wearable) | A hidden semi-Markov model with explicit, covariate-dependent state durations; plain HMMs force geometric durations | Semi-Markov dwell in the generator; R1 design |
| Watson et al. 2024 (rugby league, GPS, accelerometer, heart rate) | HMM activity states; transitions shift with score difference and elapsed time | Why R1 states may track tactics, not fatigue |
| Mohr et al. 2003 | High-intensity running falls in the last 15 minutes | Band-dependent intensity in the generator |
| Vehtari et al., JMLR 2024 | Pareto k̂ diagnostic for importance sampling; k̂ < 0.7 reliable | Interval audit (docs/02 §5.1) |
| Perdomo et al., ICML 2020 | Predictions that change decisions change the data they are later judged on | Training-data policy (docs/02 §6) |
| Holm 1979 | Step-down family-wise error control | Exploratory comparators (docs/02 §9) |

## 8. Sources

- Bengtsson H et al. (2025) BMJ Open Sport Exerc Med 11:e002772. doi:10.1136/bmjsem-2025-002772
- Moreno-Perez V et al. (2024) Biol Sport 41(1):227–233
- Gregson W et al. (2019) J Sci Med Sport - sprinting and muscle injury in match-play, Qatar Stars League
- Carling C et al. (2010) Int J Sports Med 31(3):180–185
- Carling C, McCall A, Le Gall F, Dupont G (2016) Br J Sports Med 50(12):764–768
- Mohr M, Krustrup P, Bangsbo J (2003) J Sports Sci 21:519–528
- Ryynänen J et al. - Br J Sports Med 47(15):970
- Aiello F et al. (2023) J Sci Med Sport 26:465–470. doi:10.1016/j.jsams.2023.07.007
- Miralles-Iborra A et al. (2026) Research Square preprint rs-11139698
- Green B et al. (2020) Br J Sports Med 54:1081–1088
- Huth M et al. (2025) medRxiv 10.1101/2025.04.23.25326218
- Hecksteden A et al. (2023) Sci Med Football 7(3):214–228
- Leckey C et al. (2025) Br J Sports Med 59:491–500
- Bahr R (2016) Br J Sports Med - Why screening tests to predict injury do not work
- Golovkin V et al. (2025) arXiv:2504.06357
- Mills et al. (2025) arXiv:2508.19477
- Mortensen & Bornn (2020) arXiv:2001.07692
- Poly TN et al. (2020) JMIR Med Inform 8(7):e15653
- IFAB Laws of the Game, Law 4
- Zone7 validation study (2022), zone7.ai
- Rossi A et al. (2018) PLoS ONE 13(7):e0201264
- Tabben M, Chamari K, Alkhelaifi K, Alam T, Almulla J (2026) Biol Sport 43:489–498
- Rojas-Salazar S, Schliep EM, Wikle CK, Hawkey M. arXiv:2010.10739
- Watson N, Hendricks S, Weaving D et al. (2024) doi:10.1080/02701367.2024.2362253
- Vehtari A, Simpson D, Gelman A, Yao Y, Gabry J (2024) J Mach Learn Res 25(72):1–58
- Perdomo J, Zrnic T, Mendler-Dünner C, Hardt M (2020) Performative prediction. ICML
- Holm S (1979) Scand J Stat 6(2):65–70
- Zumeta-Olaskoaga L (2024) PhD thesis, University of the Basque Country, hdl:10810/68336
