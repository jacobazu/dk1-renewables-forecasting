# tomorrow's dk1 offshore wind forecast, made before the noon auction with the model chosen in improve.py.
# only offshore runs live: onshore and solar didn't beat energinet on 2026 (see the readme).
# refit on all data up to 10 days before the first of the month, as in the backtest. each day is forecast
# once and never changed; a missed day stays a gap
#   python live.py

import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

LOG = Path("forecasts/live_forecasts.csv")
now = pd.Timestamp.now(tz="Europe/Copenhagen")
tomorrow = pd.Timestamp(now.date()) + pd.Timedelta(days=1)

if now.hour >= 12:
    print(f"{now:%H:%M}: the auction has closed, no forecast for {tomorrow:%d %b} (never made late)")
    sys.exit(0)
if LOG.exists() and (pd.read_csv(LOG, parse_dates=["date"])["date"] == tomorrow).any():
    print(f"the forecast for {tomorrow:%d %b} is already saved")
    sys.exit(0)

import improve as im  # noqa: E402 (loads the data, so only after the checks above)

cfgs = json.loads(Path("results/improve_choice.json").read_text())

# tomorrow's hours need ecmwf's run from today (00 utc). it is in the archive from about 07 utc
test = im.X[im.X["date"] == tomorrow]
if len(test) < 23 or test["ec_wind_land"].isna().sum() > 1 or test["wind_land"].isna().sum() > 1:
    sys.exit(f"weather for {tomorrow:%d %b} isn't complete yet ({len(test)} hours, "
             f"{test['ec_wind_land'].notna().sum()} with today's ecmwf run). try again later")

cutoff = tomorrow.replace(day=1) - pd.Timedelta(days=im.DELAY)
rows = []
for t in ["offshore"]:
    cfg = cfgs[t]
    cols = im.columns(cfg)
    cap = im.fleet(im.prod[t], cfg.get("fleet_days", 60))
    cap = cap.reindex(pd.date_range(cap.index.min(), tomorrow)).ffill()
    d = im.X.join(im.prod[t].rename("actual"), how="inner")
    d["cap"] = d["date"].map(cap)
    d = d.dropna(subset=["actual", "cap", "wind_land"])
    train = d[d["date"] <= cutoff]
    m = lgb.LGBMRegressor(**{**im.PARAMS, **im.PARAM_TRIES.get(cfg["params"], {})})
    m.fit(train[cols], train["actual"] / train["cap"])
    c = cap[tomorrow]
    rows.append(pd.DataFrame({"issued": now.tz_convert("UTC").strftime("%Y-%m-%d %H:%M"), "date": tomorrow,
                              "time": test.index, "target": t,
                              "forecast": np.clip(m.predict(test[cols]), 0, 1) * c, "cap": c}))
    print(f"{t}: trained on {train['date'].min():%d %b %Y} to {train['date'].max():%d %b %Y}, fleet {c:,.0f} MW, "
          f"tomorrow {rows[-1]['forecast'].mean():,.0f} MW on average")

new = pd.concat(rows)
new["date"] = new["date"].dt.strftime("%Y-%m-%d")
LOG.parent.mkdir(exist_ok=True)
new.round(1).to_csv(LOG, mode="a", header=not LOG.exists(), index=False)
print(f"saved {len(new)} rows for {tomorrow:%d %b %Y} to {LOG}")
