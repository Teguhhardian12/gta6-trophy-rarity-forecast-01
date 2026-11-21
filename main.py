#!/usr/bin/env python3
"""Trophy-rate forecaster (pinball loss on the logit scale).

Model, per trophy:
  * Work in logit space: lg(r) = ln(clamp(r,.05,99.95)/(100-clamp(...))).
  * Median = lg(last_rate) + shrunk, capped logit drift from a decay-weighted
    linear fit of the history against minutes-since-first-read, evaluated at
    the target time. A flat history gives drift 0 -> median is exactly the
    last read (no degradation on frozen rates).
  * Spread = base_sigma widened by owner granularity (0.5/owners), floored at
    base_sigma and capped at 1.6*base_sigma. base_sigma = 0.5*sqrt(h_days).
  * Five quantiles at z = -1.6449,-0.6745,0,0.6745,1.6449, back to percent,
    rounded half-up to 4 decimals, clamped to 0.01..99.99, non-decreasing.

Contract: python3 -P -s main.py ; reads *.in from $MAKEFUN_INPUT_DIR (else
/input/) in name order, writes $MAKEFUN_OUTPUT_DIR/<id>.out (else /output/).
Standard library only.
"""

import glob
import json
import math
import os
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP

# Quantile z-scores for levels 0.05, 0.25, 0.50, 0.75, 0.95.
Z = (-1.6449, -0.6745, 0.0, 0.6745, 1.6449)
LOGIT_LO, LOGIT_HI = 0.05, 99.95      # clamp inside lg()
OUT_LO, OUT_HI = 0.01, 99.99          # clamp on emitted percentages
DECAY = 0.85                          # per-read recency weight
SHRINK_K = 6.0                        # drift shrinkage: w = n/(n+K)
DRIFT_CAP = 1.2                       # |drift| <= 1.2*sqrt(h_days)
SIG_CAP = 1.6                         # sigma <= 1.6*base_sigma
MIN_OWNERS = 0.5
N_TROPHIES = 55


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def lg(r):
    r = clamp(r, LOGIT_LO, LOGIT_HI)
    return math.log(r / (100.0 - r))


def inv(x):
    """100 * sigmoid(x), overflow-safe for extreme logits."""
    if x >= 0.0:
        return 100.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return 100.0 * e / (1.0 + e)


def parse_ts(ts):
    s = str(ts).strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def read_time(read):
    # Canonical read clock: the slot (matches target/horizon); fall back to at.
    return parse_ts(read.get("slot") or read["at"])


def round4(x):
    return float(Decimal(str(x)).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP))


def wls_value(xs, ys, ws, x_target):
    """Decay-weighted least-squares line, evaluated at x_target.

    Centered form so a constant series yields exactly its own value (slope 0).
    """
    W = sum(ws)
    if W <= 0.0 or not xs:
        return ys[-1] if ys else 0.0
    xbar = sum(w * x for w, x in zip(ws, xs)) / W
    ybar = sum(w * y for w, y in zip(ws, ys)) / W
    sxx = sum(w * (x - xbar) ** 2 for w, x in zip(ws, xs))
    if sxx <= 0.0:
        return ybar
    sxy = sum(w * (x - xbar) * (y - ybar) for w, x, y in zip(ws, xs, ys))
    slope = sxy / sxx
    return ybar + slope * (x_target - xbar)


def quantiles(median, sigma):
    qs = []
    for z in Z:
        v = round4(inv(median + z * sigma))
        qs.append(clamp(v, OUT_LO, OUT_HI))
    for k in range(1, len(qs)):
        if qs[k] < qs[k - 1]:
            qs[k] = qs[k - 1]
    return qs


def forecast_trophy(i, history, target_ts, h_days, base_sigma):
    n = len(history)
    raw_last = history[-1]["rates"][i]

    # A frozen trophy stays frozen: if the last read is at a boundary, carry the
    # last interior read forward. If the whole series is boundary, emit lg(last)
    # with base_sigma and stop.
    if raw_last <= 0.0 or raw_last >= 100.0:
        eff = None
        for read in reversed(history):
            r = read["rates"][i]
            if 0.0 < r < 100.0:
                eff = r
                break
        if eff is None:
            return quantiles(lg(raw_last), base_sigma)
        last_rate = eff
    else:
        last_rate = raw_last

    players = history[-1].get("players_tracked") or 0
    count_k = max(last_rate / 100.0 * players, MIN_OWNERS)

    # Median: shrunk, capped logit drift from the weighted fit.
    t0 = read_time(history[0])
    xs = [(read_time(r) - t0) / 60.0 for r in history]
    ys = [lg(r["rates"][i]) for r in history]
    x_target = (target_ts - t0) / 60.0
    if max(ys) == min(ys):
        drift = 0.0                      # flat history -> no drift, exactly
    else:
        ws = [DECAY ** (n - 1 - k) for k in range(n)]
        drift = wls_value(xs, ys, ws, x_target) - lg(last_rate)
    cap = DRIFT_CAP * math.sqrt(max(h_days, 1e-6))
    drift = clamp(drift, -cap, cap)
    w = n / (n + SHRINK_K)
    median = lg(last_rate) + w * drift

    # Spread: granularity-widened base_sigma, floored and capped.
    granularity = 0.5 / count_k
    sigma = min(max(base_sigma, base_sigma * (1.0 + granularity)), SIG_CAP * base_sigma)
    return quantiles(median, sigma)


def persistence_rows(history, base_sigma):
    """Defensive fallback: last read carried forward with base_sigma."""
    n_t = len(history[-1]["rates"]) if history else N_TROPHIES
    rows = []
    for i in range(n_t):
        r = history[-1]["rates"][i] if history else 50.0
        rows.append({"id": i + 1, "q": quantiles(lg(r), base_sigma)})
    return rows


def build_rows(case):
    history = case.get("history") or []
    horizon = float(case.get("horizon_minutes") or 0.0)
    h_days = horizon / 1440.0
    base_sigma = 0.5 * math.sqrt(max(h_days, 1e-6))
    if history:
        target_ts = parse_ts(case["target"]) if case.get("target") else read_time(history[-1])
        n_t = len(history[-1]["rates"])
        return [
            {"id": i + 1, "q": forecast_trophy(i, history, target_ts, h_days, base_sigma)}
            for i in range(n_t)
        ]
    return persistence_rows(history, base_sigma)


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input/"
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output/"
    os.makedirs(out_dir, exist_ok=True)

    for path in sorted(glob.glob(os.path.join(in_dir, "*.in"))):
        case = None
        cid = os.path.splitext(os.path.basename(path))[0]
        try:
            with open(path, "r", encoding="utf-8") as f:
                case = json.load(f)
            cid = case.get("id", cid)
            rows = build_rows(case)
        except Exception:
            # Never leave a case without an output; fall back to persistence.
            try:
                history = (case or {}).get("history") or []
                horizon = float((case or {}).get("horizon_minutes") or 0.0)
                base_sigma = 0.5 * math.sqrt(max(horizon / 1440.0, 1e-6))
                rows = persistence_rows(history, base_sigma)
            except Exception:
                rows = [{"id": i + 1, "q": quantiles(lg(50.0), 0.5)} for i in range(N_TROPHIES)]
        out_path = os.path.join(out_dir, str(cid) + ".out")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"trophies": rows}, f, separators=(",", ":"))


if __name__ == "__main__":
    main()
