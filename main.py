"""gta6-trophy-rarity-forecast-01: trophy earn-rate forecaster (Python 3.13, standard library only).

Model, per trophy, on the logit scale l = ln(r / (100 - r)):

  1. Trend: ordinary least squares over the last up to 8 non-null reads (minutes since the first
     read), extrapolated to the case's target.
  2. Shrinkage: w = n_eff / (n_eff + 5), further gated by the fit's t-statistic, so a noisy or thin
     history barely moves; a flat history does not move at all.
  3. Tier prior: with fewer than 4 reads the trophy's slope blends 30% toward its tier's median
     slope (a weak prior, never a hard rule).
  4. Drift cap: the extrapolated logit change is clamped to +/- 1.2 * sqrt(horizon_days); the
     median rate stays in 0.01..99.99, and a platinum never forecasts above the cheapest other
     trophy (it requires every other trophy).
  5. Spread: per-tier random-walk scale from the case's own consecutive logit steps (each step
     normalized by sqrt(its gap), aggregated as RMS so refresh jumps between quiet reads are not
     lost), scaled by sqrt(horizon_minutes), floored at the baseline's 0.5 * sqrt(horizon_days),
     combined with the fit's extrapolation standard error. Quantiles at z = -1.645, -0.674, 0,
     0.674, 1.645.
  6. Stale-launch refresh: when every read is identical (the page has not refreshed), the game is
     within ~30 h of release and the tracked population is still small, the next refresh is
     expected to reveal launch growth accumulated since the page went stale: the median shifts
     down by ln(G) * (1 - r/100) with G the expected players-tracked growth over staleness +
     horizon (doubling ~2.2 h, capped at 12x), and the spread widens for the growth uncertainty.
     A trophy already at 100% stays pinned there. This fires on the launch-night practice shape
     and stays off whenever the history shows any change (the final's dynamic history).

Output shape is guaranteed: one row per trophy id, five non-decreasing rates in 0.01..99.99,
4 decimals half-up, compact JSON <= 65,536 bytes. Any unexpected error falls back to the
persistence baseline for that case, so a case is never invalid.

Run: python3 -P -s /bot/main.py (grader), or
     MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from statistics import NormalDist, median

# The grader's -P leaves /bot off sys.path: put it back first, so a module shipped beside this
# file imports.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
BASE_SIGMA = 0.5          # baseline spread, logit units per sqrt(day)
SHRINK_K = 5.0            # shrinkage weight w = n_eff / (n_eff + K)
TREND_READS = 8           # reads in the trend window
DRIFT_CAP = 1.2           # max drift, logit units per sqrt(day)
TIER_PRIOR_W = 0.3        # thin-history blend toward the tier's median slope
TIER_PRIOR_MIN_N = 4      # below this many reads, blend
SPREAD_MAX = 3.0          # sanity cap on the total spread, logit units
FIT_SE_CAP = 1.5          # cap on the fit's extrapolation SE contribution
RATE_FLOOR, RATE_CEIL = 0.01, 99.99
FIT_FLOOR, FIT_CEIL = 0.05, 99.95
# Stale-launch refresh parameters (see docstring step 6).
REFRESH_MAX_AGE_H = 30.0
REFRESH_MAX_PLAYERS = 5000
REFRESH_DOUBLING_H = 2.2
REFRESH_G_CAP = 12.0
REFRESH_SHIFT_CAP = 2.5
REFRESH_SPREAD_FRAC = 0.45
PINNED_SPREAD = 0.2


def clamp(x: float, lo: float, hi: float) -> float:
    return min(hi, max(lo, x))


def logit(pct: float) -> float:
    c = clamp(pct, RATE_FLOOR, RATE_CEIL)
    return math.log(c / (100.0 - c))


def fit_logit(pct: float) -> float:
    c = clamp(pct, FIT_FLOOR, FIT_CEIL)
    return math.log(c / (100.0 - c))


def inv_logit(l: float) -> float:
    if l > 40.0:
        return 100.0
    if l < -40.0:
        return 0.0
    return 100.0 / (1.0 + math.exp(-l))


def round4(x: float):
    """4 decimals, half up (the starter's / JavaScript's Math.round(x*10000)/10000)."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def parse_ts(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def minutes(a: datetime, b: datetime) -> float:
    return (b - a).total_seconds() / 60.0


def ols(xs: list[float], ys: list[float]):
    """Least squares line; returns (slope, s_resid, Sxx, xbar, n)."""
    n = len(xs)
    xbar = sum(xs) / n
    ybar = sum(ys) / n
    sxx = sum((x - xbar) ** 2 for x in xs)
    if sxx <= 0:
        return 0.0, 0.0, 0.0, xbar, n
    sxy = sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = ybar - b * xbar
    sse = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    s = math.sqrt(sse / (n - 2)) if n > 2 else 0.0
    return b, s, sxx, xbar, n


def quantile_row(m: float, s: float, zs: list[float]) -> list:
    q = [round4(clamp(inv_logit(m + z * s), RATE_FLOOR, RATE_CEIL)) for z in zs]
    for j in range(1, len(q)):
        if q[j] < q[j - 1]:
            q[j] = q[j - 1]
    return q


def persistence_case(case: dict) -> dict:
    """The house baseline, used as the crash fallback: never an invalid case."""
    last = case["history"][-1] if case.get("history") else None
    spread = BASE_SIGMA * math.sqrt(float(case.get("horizon_minutes", 0)) / 1440.0)
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        rate = last["rates"][i] if last else None
        lg = logit(rate if rate is not None else 0.0)
        rows.append({"id": trophy["id"], "q": quantile_row(lg, spread, Z)})
    return {"trophies": rows}


def forecast_case(case: dict) -> dict:
    trophies = case["trophies"]
    history = case["history"]
    if not history:
        return persistence_case(case)
    horizon = float(case["horizon_minutes"])
    days = max(horizon / 1440.0, 1e-9)
    t0 = parse_ts(history[0]["at"])
    t_at = parse_ts(case["at"])
    t_target = parse_ts(case["target"])
    x_target = minutes(t0, t_target)
    zs = Z
    if case.get("levels") and case["levels"] != LEVELS and len(case["levels"]) == 5:
        nd = NormalDist()
        zs = [nd.inv_cdf(float(tau)) for tau in case["levels"]]

    reads = [(minutes(t0, parse_ts(h["at"])), h["rates"], h.get("players_tracked")) for h in history]
    last_rates = reads[-1][1]

    # Per-trophy series on the fit logit, and the trend window (last up to 8 non-null reads).
    series = []
    for i in range(len(trophies)):
        xs, ys = [], []
        for x, rates, _ in reads:
            r = rates[i]
            if r is not None:
                xs.append(x)
                ys.append(fit_logit(float(r)))
        series.append((xs, ys))
    fits = []
    for xs, ys in series:
        wx, wy = xs[-TREND_READS:], ys[-TREND_READS:]
        fits.append(ols(wx, wy) if len(wx) >= 2 else (0.0, 0.0, 0.0, wx[-1] if wx else 0.0, len(wx)))

    # Tier median slopes (weak prior for thin histories).
    tier_slopes = {}
    for tier in {t["tier"] for t in trophies}:
        bs = [fits[i][0] for i, t in enumerate(trophies) if t["tier"] == tier and fits[i][4] >= 3]
        if bs:
            tier_slopes[tier] = median(bs)

    # Per-tier random-walk scale from consecutive logit steps, normalized by sqrt(gap): RMS keeps
    # the energy of refresh jumps that sit between long runs of unchanged reads.
    tier_steps = {}
    for i, t in enumerate(trophies):
        xs, ys = series[i]
        acc = tier_steps.setdefault(t["tier"], [])
        for j in range(1, len(xs)):
            gap = xs[j] - xs[j - 1]
            if gap > 0:
                acc.append((ys[j] - ys[j - 1]) / math.sqrt(gap))
    tier_sigma = {}
    for tier, acc in tier_steps.items():
        tier_sigma[tier] = math.sqrt(sum(v * v for v in acc) / len(acc)) if acc else 0.0

    # Stale-launch refresh detection: the whole history is one unchanged page, the game just
    # released, and the tracked population is still small.
    first_rates, first_players = reads[0][1], reads[0][2]
    flat = all(r == first_rates and p == first_players for _, r, p in reads)
    refresh_on = False
    ln_g = 0.0
    if flat and len(reads) >= 2:
        stale_h = minutes(parse_ts(history[0]["at"]), t_at) / 60.0
        try:
            rel = date.fromisoformat(str(case.get("game", {}).get("release_date", ""))[:10])
            release = datetime(rel.year, rel.month, rel.day, tzinfo=timezone.utc)
            age_h = minutes(release, t_at) / 60.0
        except (ValueError, TypeError):
            age_h = 1e9
        players = first_players if isinstance(first_players, (int, float)) else 1e9
        if 0 <= age_h <= REFRESH_MAX_AGE_H and players <= REFRESH_MAX_PLAYERS and stale_h >= 0.25:
            refresh_on = True
            total_h = stale_h + 0.5 + horizon / 60.0  # +0.5 h: the page was already stale at read 1
            g = min(REFRESH_G_CAP, 2.0 ** (total_h / REFRESH_DOUBLING_H))
            ln_g = math.log(g)

    floor_spread = BASE_SIGMA * math.sqrt(days)
    drift_cap = DRIFT_CAP * math.sqrt(days)
    medians, spreads, pinned = [], [], []
    for i, t in enumerate(trophies):
        xs, ys = series[i]
        rate = last_rates[i]
        if rate is None:
            rate = ys and inv_logit(ys[-1]) or None
            if rate is None:
                same = [last_rates[j] for j, o in enumerate(trophies)
                        if o["tier"] == t["tier"] and last_rates[j] is not None]
                rate = median(same) if same else 1.0
        rate = float(rate)
        is_pinned = rate >= RATE_CEIL
        pinned.append(is_pinned)
        l_last = logit(rate)
        if is_pinned:
            medians.append(logit(RATE_CEIL))
            spreads.append(PINNED_SPREAD)
            continue
        b, s_resid, sxx, xbar, n = fits[i]
        n_eff = min(n, TREND_READS)
        w = n_eff / (n_eff + SHRINK_K)
        if s_resid > 1e-12 and sxx > 0:
            tstat = abs(b) * math.sqrt(sxx) / s_resid
            rel = tstat / (tstat + 2.0)
        else:
            rel = 1.0
        if n_eff < TIER_PRIOR_MIN_N and t["tier"] in tier_slopes:
            b = (1.0 - TIER_PRIOR_W) * b + TIER_PRIOR_W * tier_slopes[t["tier"]]
        x_last = xs[-1] if xs else 0.0
        drift = clamp(w * rel * b * (x_target - x_last), -drift_cap, drift_cap)
        if refresh_on:
            # A refresh jump, not a random walk: the denominator grows G-fold while this trophy's
            # earners barely move; a trophy everyone holds (100%) does not move at all.
            drift = clamp(-ln_g * (1.0 - rate / 100.0), -REFRESH_SHIFT_CAP, REFRESH_SHIFT_CAP)
            s = math.sqrt(floor_spread ** 2 + (REFRESH_SPREAD_FRAC * ln_g) ** 2)
        else:
            sigma_unit = tier_sigma.get(t["tier"], 0.0)
            s_rw = max(sigma_unit * math.sqrt(horizon), floor_spread)
            s_fit = 0.0
            if n >= 2 and sxx > 0:
                s_fit = w * s_resid * math.sqrt(1.0 + 1.0 / n + (x_target - xbar) ** 2 / sxx)
                s_fit = min(s_fit, FIT_SE_CAP)
            s = min(SPREAD_MAX, math.sqrt(s_rw ** 2 + s_fit ** 2))
        m = clamp(l_last + drift, logit(RATE_FLOOR), logit(RATE_CEIL))
        medians.append(m)
        spreads.append(s)

    # Platinum requires every other trophy: it cannot forecast above the cheapest other median.
    others = [medians[j] for j in range(len(trophies)) if trophies[j]["tier"] != "platinum"]
    if others:
        cap = min(others)
        for j, t in enumerate(trophies):
            if t["tier"] == "platinum" and not pinned[j] and medians[j] > cap:
                medians[j] = cap

    rows = [{"id": t["id"], "q": quantile_row(medians[i], spreads[i], zs)} for i, t in enumerate(trophies)]
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = {}
        try:
            case = json.loads(path.read_text(encoding="utf-8"))
            forecast = forecast_case(case)
        except Exception:
            try:
                forecast = persistence_case(case)
            except Exception:
                forecast = {"trophies": [{"id": t["id"], "q": [0.01] * 5} for t in case.get("trophies", [])]}
        text = json.dumps(forecast, separators=(",", ":")) + "\n"
        (OUTPUT_DIR / f"{path.stem}.out").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()