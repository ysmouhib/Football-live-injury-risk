# What it would take at a club

[Back to the README](../README.md)


Everything in this repository is synthetic for one reason: the tool needs data that only a club holds. Most
professional clubs already collect that data. With access to it, the tool can be put through
the same tests on real players, and only then can anyone say whether it works.

## What the tool needs, and where a club already has it

| The tool needs | Why | Where it usually lives at a club |
|---|---|---|
| Live tracking of every player during the match, several times a second | The live running numbers (high-speed metres, sprints, recovery between efforts) | GPS vests worn in matches, or the stadium's optical tracking system |
| Each player's past matches, tracked the same way | To learn "his normal" at each stage of a match | The archive of the same vest or stadium system, usually several seasons |
| A weekly pre-match risk for each player | The starting point before kick-off: injury history, minutes played, fixture congestion, recent load | The medical and sports-science department's records and load monitoring |
| Injury records with the match minute of each injury | To check, after the fact, whether the figures and cards were right | The medical department's injury log, matched to video for the minute |

None of this leaves the club. The tool runs on one machine at the club or the ground, and the
data stays on the club's own network.

## Why public data or a TV broadcast is not enough

Free public tracking data exists. SkillCorner, for example, publishes tracking of ten real
A-League 2024/25 matches taken from the TV pictures, and the tool can already read it. It shows
the problem well:

- **Too little history.** Ten matches across a whole league give most players zero, one or two
  past matches, so "his normal" falls back on the average for his position. A club's archive has
  dozens of matches per player.
- **No pre-match risk.** There is no injury history, training load or fixture list behind the
  players, so every player would start from the same league-average figure.
- **No injury records.** Without them nobody can check whether a card was right, so the four
  pre-registered tests ([RESULTS.md](RESULTS.md)) cannot be run at all.
- **Players off the screen.** A TV camera follows the ball. Players out of the picture have their
  positions estimated, which is far less reliable than a GPS vest for counting sprints.

Public data like this is still useful for one thing: checking that the system copes with real
running patterns and real data gaps. It cannot tell anyone whether the risk figures mean
anything.

## The steps with a club

1. **Load the history.** Past seasons of tracking and the injury log are read in on the club's
   machine. "His normal" is then built from each player's own matches.
2. **Run the four tests on past seasons.** The model is fitted on earlier seasons and tested on
   a later one it has never seen, with the pass marks written down before the test, exactly as
   in the [synthetic test](RESULTS.md). If it fails, the tool stays in monitor-only mode: live load
   against each player's normal, without risk percentages.
3. **Shadow season.** The tool runs live in every match, but staff do not see it. At the end of
   the season the same four tests are run again on what it would have shown.
4. **Live pilot.** Only then do the cards reach the bench, with the actions and the card budget
   agreed in advance, and every decision logged.

What the synthetic tests show is that the machinery is ready for this: when a real effect is in
the data, the model finds it, keeps its totals right and puts its cards where the injuries are.
Whether real running data carries that effect for real players is the question a club's history
answers.
