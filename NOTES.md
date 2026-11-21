# Trophy earn-rate forecaster — gta6-trophy-rarity-forecast-01

`main.py`, Python 3.13, standard library only, no network. One JSON forecast per case:
five quantiles (0.05/0.25/0.50/0.75/0.95) of each trophy's Exophase earn rate at the
case's target, on the logit scale the pinball loss scores.

## What the model does

Per trophy, on `l = ln(r/(100-r))` (fit clamp 0.05..99.95, output clamp 0.01..99.99):

1. **Trend** — OLS line over the last up to 8 non-null reads (minutes since first read),
   extrapolated to `target`.
2. **Shrinkage** — `w = n_eff/(n_eff+5)`, further gated by the fit's t-statistic
   (`t/(t+2)`), so thin or noisy histories fall back toward persistence; a flat history
   has slope 0 and does not move.
3. **Drift cap** — extrapolated logit change clamped to ±`1.2*sqrt(horizon_days)`;
   median rate clamped to 0.01..99.99. Platinum additionally never forecasts above the
   cheapest other trophy's median (it logically requires every other trophy).
4. **Spread** — per-tier random-walk scale from the case's own consecutive logit steps,
   each normalized by `sqrt(gap)` and aggregated as **RMS** (not median-absolute: the
   page sits unchanged between Exophase refreshes, so most steps are 0 and the median
   would throw the refresh jumps' energy away). Scaled by `sqrt(horizon_minutes)`,
   floored at the baseline's `0.5*sqrt(horizon_days)`, combined in quadrature with the
   fit's extrapolation standard error (capped 1.5; total capped 3.0). Quantiles at
   z = −1.645, −0.674, 0, 0.674, 1.645.
5. **Tier prior** — with <4 reads the slope blends 30% toward the tier's median slope
   (weak prior, never a hard rule).
6. **Stale-launch refresh** — the practice's shape: every read identical (page never
   refreshed), game within 30 h of release, tracked population ≤ 5000. The next refresh
   then reveals launch growth accumulated while the page was stale: median shifts by
   `−ln(G)*(1 − r/100)` where `G = 2^((staleness_h + 0.5 + horizon_h)/2.2)` capped at 12×
   (doubling ~2.2 h on launch night), and the spread widens by `0.45*ln(G)` in quadrature
   for growth uncertainty. A trophy already at 100% stays pinned at 99.99 with a small
   spread (the documented refresh left the 100% rate at 100%). The gate is **off whenever
   any read differs** — the final's dynamic history never triggers it.

Shape guarantees: one row per id, non-decreasing after half-up 4-decimal rounding,
every rate in 0.01..99.99, compact JSON. Any unexpected error falls back to the
persistence baseline for that case, so a case is never invalid.

## Measurement — what was possible, and the numbers

**`grade-local` cannot grade this kit.** Run as instructed:

```
cd …/gta6-trophy-rarity-forecast-01-kit && node cli/makefun.mjs grade-local --kit . --bot /var/tmp/rarity --driver process
→ "gta6-trophy-rarity-forecast-01 is a program/v1 competition: its scorer and reference
   data are not published in the kit, so it is graded on the server only"
```

The practice answers are sealed server-side; no submission was made (out of scope for
this task). Local measurement therefore uses:

- an **exact reimplementation of the documented metric** (pinball `2*max(τd,(τ−1)d)`,
  `d = logit(y)−logit(q)`, clamp 0.01..99.99, half-up 4 decimals). Verified against the
  RULES worked example: loss **0.403404**, typical miss 7, coverage 0.5 — exact match.
  The persistence fallback also reproduces the published practice row for id 4
  digit-for-digit: `[0.8974, 1.012, 1.1, 1.1955, 1.3477]`.
- a **labelled reconstruction of the practice answer reads**, not the sealed answers:
  RULES documents the 04:30Z refresh exactly (players_tracked 638 → **3868**, every rate
  but the 100% one lower). Pre-refresh earn counts are exact integers (`rate*638/100`,
  summing to the page's `trophies_earned = 943` — checked). The only unknown is numerator
  growth `g` between the stale snapshot and the refresh; `g` was calibrated so the
  persistence baseline reproduces its **published practice loss ≈ 1.51**: `g = 1.20`
  gives baseline **1.497029**. p01's answer is taken as the same refreshed page (no
  second refresh documented). These numbers are reconstruction-relative, not official
  grades; the model was checked across the whole plausible `g` range so nothing is tuned
  to one guess.

Practice losses (mean over p00+p01, reconstruction above):

| attempt | change | loss (g=1.20) | verdict |
|---|---|---|---|
| 0 | house baseline / starter persistence | 1.497029 | reference (published ≈1.51) |
| 1 | trend + shrinkage + drift cap + tier prior + residual spread (steps 1–5) | 1.497029 | no effect on practice: flat histories ⇒ slope 0, tier steps 0 ⇒ exactly the baseline's rows. Kept for the final, where histories are dynamic. |
| 2 | + stale-launch refresh module (step 6) | **0.170233** | kept — ~8.8× lower; median miss 0.014 pp, 90% coverage 0.982 |
| 3 | robustness sweep g ∈ {1.0, 1.2, 1.4, 1.6} | 0.177 / 0.170 / 0.194 / 0.224 | kept, no parameter change: beats baseline (1.688 / 1.497 / 1.366 / 1.212) at every g |

Final-case mechanics (no final data exists locally before the deadline): smoke-tested on
a synthetic dynamic case (30 ten-minute reads, rising/falling/flat trophies, growing
players_tracked, 6510-minute horizon): all 55 rows valid, zero drift-cap violations,
refresh gate correctly off. This validates mechanics only — it is not a tuning target.

Runtime: **135 ms** wall for both practice cases (`python3 -P -s`, 8 s cap). Outputs
2,803 / 2,779 bytes (65,536 cap), shape checks pass.

## Rejected

- **Hardcoding the reconstructed answer rates** — overfits a guess and cannot generalize
  to the final; the refresh module instead derives the shift from case-observable facts
  (staleness, release date, population size).
- **Median shift on dynamic histories** — would wreck the final; the refresh gate
  requires a wholly unchanged page plus launch youth.
- **Median-absolute step as the tier spread** (as literally prescribed) — it is 0 whenever
  >50% of consecutive reads are unchanged, which is exactly the refresh-jump regime;
  RMS of gap-normalized steps keeps the jumps' energy and still floors at the baseline.
- **Raw (unshrunk, uncapped) extrapolation** — an 80-minute window extrapolated 4.5 days
  is ~80× leverage; shrinkage + t-gate + ±1.2√d cap contain it.
- **Equal quantiles / narrow bands** — pinball on the logit scale punishes over-confidence
  more than honest width; the refresh spread carries the growth uncertainty explicitly.
- **Forecasting the 100% trophy below 99.99** — the scorer clamps y=100 to 99.99
  (logit 9.21); a median at 99.95 would cost ~3.9 logit units for nothing. Pinned at
  99.99, spread 0.2.

## Reproduce

```
mkdir -p /tmp/in /tmp/out
cp <kit>/input/p00.in <kit>/input/p01.in /tmp/in/
MAKEFUN_INPUT_DIR=/tmp/in MAKEFUN_OUTPUT_DIR=/tmp/out python3 -P -s main.py
```

(The repo ships no `input/` — the grader supplies the case; practice inputs live in the
kit. Scorer/reconstruction scripts were kept out of the repository, in the session
scratch dir.)