# Trophy earn-rate forecaster

`main.py` — Python 3.13, standard library only, no network. Reads every `*.in`
from `$MAKEFUN_INPUT_DIR` (else `/input/`) in name order, writes
`$MAKEFUN_OUTPUT_DIR/<id>.out` (else `/output/`). Run: `python3 -P -s main.py`.

## The model (five lines)

1. Logit space: `lg(r)=ln(clamp(r,0.05,99.95)/(100-clamp(r,0.05,99.95)))`, `inv(x)=100/(1+exp(-x))`; `base_sigma=0.5*sqrt(max(h_days,1e-6))`, `h_days=horizon_minutes/1440`.
2. Median: decay-weighted (`0.85**(n-1-k)`) line of logit-rate vs minutes-since-first-read, evaluated at `target`; `drift=fit-lg(last_rate)`, clamped to `±1.2*sqrt(h_days)`, shrunk by `w=n/(n+6)`; `median=lg(last_rate)+w*drift`.
3. Boundary: a last read of 0/100 carries the last interior read forward (frozen stays frozen); an all-boundary series emits `lg(last_rate)` with `base_sigma`.
4. Spread: `granularity=0.5/max(last_rate/100*players_tracked,0.5)`; `sigma=min(max(base_sigma,base_sigma*(1+granularity)),1.6*base_sigma)` — never narrower than base, never wider than 1.6×.
5. Quantiles at `z=-1.6449,-0.6745,0,0.6745,1.6449` around the median, back to percent, half-up to 4 decimals, clamped `0.01..99.99`, forced non-decreasing; one row per id 1..55.

## Backtest numbers

`python3 /var/tmp/backtest_rarity.py /var/tmp/rarity-practice /var/tmp/rarity`
(truncate the last read, predict it, score pinball on the logit scale):

```
base: mean pinball 0.007176  n=110
ours: mean pinball 0.007849  n=110   (threshold <= 0.0085 -> PASS)
improvement: -0.000673 (-9.4%)
```

Practice shape/median check: 55 rows, five non-decreasing quantiles each,
worst median vs last read **0.0005** (< 0.02). Outputs 2,654 bytes/case
(< 65,536). The small negative "improvement" is expected and correct: on a flat
history the truth equals the last read, so the baseline's narrow band is
near-optimal and our honest granularity widening adds a little loss — still
under the 0.0085 ceiling. The spread was implemented to spec, not tuned.

## Why a flat history does not drift

In p00/p01 every read is identical, so each trophy's logit series is constant.
A weighted least-squares line through a constant series has slope 0 and fits
that same constant at every x, including `target`: `fit = lg(last_rate)`, hence
`drift = 0` (the code also short-circuits `max(ys)==min(ys)` to exactly 0).
With `drift=0` the shrinkage and cap are irrelevant and
`median = lg(last_rate)`, so `q50` rounds back to the last read's rate exactly.
The lone 100% trophy is an all-boundary series: it emits `lg(100)=lg(99.95)`
with `base_sigma`, pinning `q50` at 99.95 (0.05% off, within tolerance). There
is no staleness/launch-growth heuristic anywhere — nothing can move the median
off a frozen page.
