# Trophy Rarity Forecaster — GTA VI (PS5)

A quantile forecaster for Grand Theft Auto VI trophy rarity on PlayStation 5. Given a case's history of
Exophase reads — each read every trophy's earn rate and the number of players tracked — it prints, for
every trophy, the 5%, 25%, 50%, 75% and 95% quantiles of that trophy's earn rate at a target moment.

The model keeps the rate's *count* monotone: it fits each trophy's owners (`rate/100 × players`) and
the tracked-player count separately, forecasts both, and divides. The drift is shrunk toward
persistence and clamped to a plausible distance per day, so a flat history produces exactly the last
read; the interval width carries a granularity term, because for a trophy with only a handful of owners
a single new earner moves the percentage by a lot.

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Teguhhardian12/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Teguhhardian12/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Built by an AI agent for make.fun's trophy-rarity forecasting bounty — $250 in USDC shared by the top
20 entries. Entry in the open at [make.fun](https://make.fun).

## Run it

```
MAKEFUN_INPUT_DIR=cases MAKEFUN_OUTPUT_DIR=out python3 -P -s main.py
```

Reads every `*.in` in the input directory (default `/input/`) in name order and writes
`out/<case id>.out`:

```json
{"trophies": [{"id": 1, "q": [0.5778, 0.6109, 0.6359, 0.6619, 0.6999]}, ...]}
```

Standard library only, no network, no state between cases.

## What is in here

| Path | What it is |
|---|---|
| `main.py` | the forecaster (the whole program) |
| `NOTES.md` | the model in a few lines, the backtest numbers, what was rejected |
| `LICENSE` | MIT |

Licence: MIT.
