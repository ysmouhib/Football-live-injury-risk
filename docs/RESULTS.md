# Results on held-out synthetic seasons

[Back to the README](../README.md)


Before any club data is used, the method has to show three things: it finds an effect when one
is there, it gets the totals right, and its alerts land where the injuries are. We test this on
synthetic seasons where we set the truth ourselves.

**The test.** 9,000 synthetic player-matches across 10 clubs and 3 seasons. In this data the live
running numbers really do change the injury risk, by amounts we chose (for example, one standard
deviation of extra high-speed running raises the risk by 35 %). The model is fitted on earlier
seasons and scored on the next season, which it has never seen: season 2, then season 3. That
gives 30,991 fifteen-minute stretches of play and 325 non-contact injuries to score against.
Every number in this section comes from one deterministic run of `livrisk evaluate` with its
default settings.

> The synthetic injury rate is higher than in real football: 1.05 % of stretches contain the
> start of an injury, about four times the real rate for muscle injuries (0.26 % per player per
> 15 minutes). It is raised so the tests have enough injuries to measure. Compare the lines in
> each chart with each other, not with real-world rates.

## The four pre-registered tests (Gate C)

The pass marks were fixed before the run. If any test fails, the tool runs in monitor-only mode:
pre-match risk plus live load against each player's normal, without live risk percentages.

| Test | Pass mark | Result |
|---|---|---|
| 1. The live numbers improve on "pre-match risk shaped by the phase of the match" | 95 % interval of the improvement above zero | **Pass.** Log-loss improvement 0.000531 per stretch, 95 % interval 0.000175 to 0.000905 |
| 2. The totals are right in every test season | Observed ÷ expected between 0.8 and 1.25 | **Pass.** Season 2: 1.00. Season 3: 1.06 |
| 3. At 1.5 alerts per team per match, flagged stretches hold a larger share of the injuries than of the playing time | 95 % interval of the difference above zero | **Pass.** 8.6 % of injuries in 2.3 % of stretches; difference 3.4 to 9.5 points |
| 4. No club is badly off | Observed ÷ expected between 0.6 and 1.6 for every club | **Pass.** Clubs range from 0.81 to 1.28 |

## The totals are right

Across both test seasons the model expected 316.4 injuries and 325 happened (observed ÷
expected 1.03). Sorting the stretches into ten equal groups by predicted risk, the riskiest tenth
held 83 injuries against 74.9 expected, and the lowest tenth 6 against 6.8. The second tenth is
the furthest off: 21 against 13.2.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="img/calibration-dark.png">
  <img src="img/calibration.png" alt="Bars of expected injuries and dots of observed injuries for ten groups of stretches ordered by predicted risk; the dots follow the bars, rising from 6 to 83" width="760">
</picture>

The same holds minute by minute. By the 90th minute, 6.48 % of players had a non-contact injury
against 6.16 % expected.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="img/over_the_match-dark.png">
  <img src="img/over_the_match.png" alt="Cumulative share of players injured by match minute, observed and expected; the two lines stay close from kick-off to the 90th minute" width="760">
</picture>

## Where the alerts land

We ranked all 30,991 stretches by how much the live numbers raised the risk and flagged the top
2.3 %, which matches the alert budget of 1.5 per team per match.

- **28 of the 325 injuries (8.6 %) began in a flagged stretch**, 3.8 times the share that random
  flags would catch.
- **28 of the 704 flags (4.0 %) were followed by an injury in that stretch.** The other 676 were
  not.
- **297 of the 325 injuries began in a stretch that was not flagged.**
- In the 28 caught cases, the injury began a median of 5 minutes into the flagged stretch.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="img/flags_and_catches-dark.png">
  <img src="img/flags_and_catches.png" alt="Left: 2.3 % of stretches flagged against 8.6 % of injuries beginning in a flagged stretch. Right: 4 of every 100 flags followed by an injury" width="760">
</picture>

## What these numbers mean for a real club

**They are not a forecast of how the tool will do with real players.** They depend on two
things we chose when we made the synthetic data: how often injuries happen, and how strongly
the running numbers are tied to them. With a club's real data either could turn out differently,
and the tool could do better or worse than this.

- **How strongly the running numbers are tied to injury.** In the synthetic data we set
  deliberately modest effects (35 % more risk per standard deviation of extra high-speed
  running). If real fatigue signals are stronger, more injuries would fall in flagged stretches;
  if they are weaker, fewer would. Only a club's history can tell.
- **How rare injuries are.** This part does not depend on the model. A muscle injury starts in
  about 0.26 % of 15-minute stretches per player in real elite football, and two cards per team
  per match means flagging about 2.3 % of stretches. Even a model that caught every single
  injury could therefore have at most about 1 card in 9 followed by an injury (0.26 ÷ 2.3).
  A model that caught 30 % of them would have about 1 in 30.

So the useful questions for a club are how many injuries a card catches and how much earlier,
not whether most cards are "right". Most cards will not be followed by an injury whatever the
model; that is why each card asks only for a cheap action.

## Three times it worked

These three player-matches are among the 28 caught cases, picked by us because the pattern is
easy to see. They are not typical of all 28. The blue line is how much the live numbers raised
his risk each minute, the shaded band is the flagged 15-minute stretch, and the red line is the
start of the injury in the synthetic ground truth.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="img/examples-dark.png">
  <img src="img/examples.png" alt="Three player-matches: in each, the live figure climbs above twice the player's normal shortly before or during a flagged stretch that contains the start of the injury" width="820">
</picture>

- **sm003177 (season 3).** For the first 14 readings (0:00 to 13:00) his figure moved between
  ×0.94 and ×1.63. It jumped to ×2.47 at 14:00 and peaked at ×3.42 at 16:00. The 15:00 to 30:00
  stretch was flagged at ×2.74; the injury began in the 19th minute.
- **sm001242 (season 2).** Below his normal from 63:00 to 69:00 (×0.45 to ×0.70), then above
  ×1.5 from 71:00 and up to ×2.51 at 80:00. The 75:00 to 90:00 stretch was flagged at ×1.94;
  the injury began in the 83rd minute.
- **sm001648 (season 3).** Near his normal for most of the match, then ×1.94 at 70:00 and a peak
  of ×3.91 at 79:00. The 75:00 to 90:00 stretch was flagged at ×2.49; the injury began in the
  90th minute.

## Against simpler methods

Three simpler methods built from public approaches (an acute:chronic load ratio, "players like
him", a fixed threshold rule) did not improve on pre-match risk shaped by the phase of the match:
all three 95 % intervals include zero. Do not read too much into this. The synthetic data were
generated from the same kind of model the tool fits, so this comparison checks that the machinery
works, not which method is best on real data.
