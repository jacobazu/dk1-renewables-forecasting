"""Offshore wind capture price: what a DK1 offshore wind farm earns per MWh in the day-ahead market, the price
weighted by its output. Forecast for tomorrow before the auction, from our offshore forecast and the price
forecast (jacobazu/dk1-price-forecasting), and scored against the real one.

Rules, written before the backtest was run (12 Nov 2026):
- Days: 2026 backtest days where both the offshore backtest (results/final_forecasts.csv) and the price backtest
  (the price repo's results/backtest_forecasts.csv) exist. Danish days, 24 hours, no missing hour.
- Ours: sum(offshore forecast x price forecast) / sum(offshore forecast), the price forecast being the ensemble.
- Benchmark, no wind forecast: the price forecast's day average x the capture ratio of the last 30 days
  (real capture price / real day average), as someone with only a price forecast would guess it.
- Score: absolute error per day, €/MWh. Ours is better if its mean absolute error is lower, Diebold-Mariano with
  Newey-West (7 lags), p < 0.05.
- Also shown, not tested: the same with Energinet's wind forecast in place of ours (published after the auction).

    python capture.py      writes results/capture_backtest.csv and results/capture_tests.csv
"""
import io
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

PRICE_REPO = "https://raw.githubusercontent.com/jacobazu/dk1-price-forecasting/master/"


def price_file(name, local=Path("../dk1-price-forecasting")):
    if (local / name).exists():
        return pd.read_csv(local / name)
    with urllib.request.urlopen(PRICE_REPO + name, timeout=60) as r:
        return pd.read_csv(io.BytesIO(r.read()))


def capture(w, p):
    w = np.clip(np.asarray(w, float), 0, None)
    return float((w * p).sum() / w.sum()) if w.sum() > 0 else np.nan


def nw_t(d, lags=7):
    d = np.asarray(d, float); d = d[~np.isnan(d)]; n = len(d); m = d.mean(); e = d - m
    v = e @ e / n + 2 * sum((1 - k / (lags + 1)) * (e[k:] @ e[:-k]) / n for k in range(1, lags + 1))
    return m, m / np.sqrt(v / n)


def main():
    f = pd.read_csv("results/final_forecasts.csv", parse_dates=["time"])
    f = f[f["target"] == "offshore"].copy()
    local = f["time"].dt.tz_localize("UTC").dt.tz_convert("Europe/Copenhagen")
    f["date"], f["hour"] = local.dt.strftime("%Y-%m-%d"), local.dt.hour
    f = f.groupby(["date", "hour"], as_index=False)[["actual", "forecast", "energinet_scaled"]].mean()
    p = price_file("results/backtest_forecasts.csv")
    p["ensemble"] = p[["bmtx", "learx", "xgbx"]].mean(axis=1)
    d = f.merge(p[["date", "hour", "actual", "ensemble"]].rename(columns={"actual": "price"}), on=["date", "hour"])

    rows = []
    for day, g in d.groupby("date"):
        if len(g) != 24 or g[["actual", "forecast", "price", "ensemble"]].isna().any().any():
            continue
        rows.append({"date": day, "real": capture(g["actual"], g["price"]), "ours": capture(g["forecast"], g["ensemble"]),
                     "energinet": capture(g["energinet_scaled"], g["ensemble"]),
                     "day_avg_real": g["price"].mean(), "day_avg_fc": g["ensemble"].mean()})
    out = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    ratio = (out["real"] / out["day_avg_real"]).rolling(30, min_periods=20).mean().shift(1)
    out["no_wind_forecast"] = out["day_avg_fc"] * ratio
    out = out.dropna(subset=["no_wind_forecast"]).round(2)
    out.to_csv("results/capture_backtest.csv", index=False)

    err = {k: (out[k] - out["real"]).abs() for k in ["ours", "no_wind_forecast", "energinet"]}
    m, t = nw_t(err["no_wind_forecast"] - err["ours"])
    from math import erf, sqrt
    p_val = 2 * (1 - 0.5 * (1 + erf(abs(t) / sqrt(2))))
    tests = pd.DataFrame([{"days": len(out), "first": out["date"].min(), "last": out["date"].max(),
                           "mae_ours": err["ours"].mean(), "mae_no_wind_forecast": err["no_wind_forecast"].mean(),
                           "mae_energinet": err["energinet"].mean(), "bias_ours": (out["real"] - out["ours"]).mean(),
                           "capture_ratio_real": (out["real"] / out["day_avg_real"]).mean(),
                           "gain": m, "dm_t": t, "dm_p": p_val, "ours_better": bool(m > 0 and p_val < 0.05)}]).round(3)
    tests.to_csv("results/capture_tests.csv", index=False)
    print(tests.T.to_string())


if __name__ == "__main__":
    main()
