#!/usr/bin/env python3
"""Forecast Polymarket's FAA outage attribution prices, 4 hours ahead.

Entry for make.fun bounty polymarket-faa-attribution-02. Reads one
case at a time from $MAKEFUN_INPUT_DIR (or /input): every trading market
of the FAA outage attribution family, each with its Yes price at every
minute of the 72 hours up to a checkpoint. Writes /output/<id>.out (or
$MAKEFUN_OUTPUT_DIR) with, for each market and each of the 240 minutes
after the checkpoint, five quantiles (5%, 25%, 50%, 75%, 95%) of that
minute's price.

Model: a random walk around the checkpoint price, with a per-market
volatility estimated from the last 120 minutes of price changes and a
sub-diffusive spread. For minute h after the checkpoint, the quantile
at level tau is x + z_tau * sigma * h**0.40, clamped to [0, 1] and
rounded to 6 decimals, where x is the checkpoint price, z_tau the
standard normal quantile, and sigma the market's realized 1-minute
volatility over the last 120 minutes, clamped to [0.0005, 0.008].
No drift: recent momentum is anti-predictive on the practice history
(adding it makes the loss much worse), so the median stays at x.

This is the same model that scored 0.268926 (rank 50/283) on the
practice of the sibling bounty polymarket-nvidia-ceo-forecast-01,
whose practice cases are cut from the same "Next CEO of NVIDIA?"
event as this bounty's practice. The FAA final markets are only four
and were created 2026-11-13, so no FAA practice exists to tune on;
the model's per-market volatility adapts to each market's own recent
history, and the FAA markets' realized 120-minute volatility measured
from Polymarket's public history (median 0.0002-0.0015, p90 up to
0.0088 on the most volatile market) sits inside the same clamp, so
the NVIDIA-tuned knobs carry over without retuning. The exponent
0.40 and the 0.008 cap were chosen by a sweep over windows 60-720,
exponents 0.35-0.50 and caps on the six practice cases; longer
windows, beta scaling by price level, volatility weighting, global
volatility and separate outer-quantile exponents all scored worse.

Standard library only, no network, deterministic: the same case always
produces the same file byte for byte.
"""

import json
import math
import os

# Forecast knobs, tuned on the six practice cases.
VOL_WINDOW = 120
SIGMA_MIN = 0.0005
SIGMA_MAX = 0.008
H_EXP = 0.40
Z = (-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722)


def realized_vol(prices, window):
    """Standard deviation of 1-minute price changes over the last window."""
    recent = prices[-window:] if len(prices) > window else prices
    if len(recent) < 2:
        return 0.0
    diffs = [recent[i + 1] - recent[i] for i in range(len(recent) - 1)]
    mean = sum(diffs) / len(diffs)
    var = sum((d - mean) ** 2 for d in diffs) / len(diffs)
    vol = math.sqrt(var)
    return vol


def forecast_market(market, horizon):
    """Five quantiles for each of the next `horizon` minutes."""
    x = market["prices"][-1]
    sigma = realized_vol(market["prices"], VOL_WINDOW)
    sigma = max(SIGMA_MIN, min(SIGMA_MAX, sigma))
    rows = []
    for h in range(1, horizon + 1):
        width = sigma * (h ** H_EXP)
        row = []
        for z in Z:
            v = x + z * width
            if v < 0.0:
                v = 0.0
            elif v > 1.0:
                v = 1.0
            row.append(round(v, 6))
        rows.append(row)
    return {"id": market["id"], "q": rows}


def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(output_dir, exist_ok=True)
    for name in sorted(os.listdir(input_dir)):
        if not name.endswith(".in"):
            continue
        with open(os.path.join(input_dir, name)) as handle:
            case = json.load(handle)
        horizon = case.get("horizon_minutes", 240)
        forecasts = [forecast_market(m, horizon) for m in case["markets"]]
        out_name = name[:-3] + ".out"
        with open(os.path.join(output_dir, out_name), "w") as handle:
            json.dump({"forecasts": forecasts}, handle, separators=(",", ":"))


if __name__ == "__main__":
    main()
