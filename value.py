# what the offshore forecast is worth to a wind farm that sells its forecast day-ahead and settles the miss
# at the imbalance price. 2026 backtest hours, per MW of fleet. run once, result in results/value.csv
#   python value.py            (prices from energi data service)
#   PRICES=file.csv python value.py   (hourly csv with time, spot, imb)

import os

import numpy as np
import pandas as pd

from db import get

f = pd.read_csv("results/final_forecasts.csv", parse_dates=["time"])
f = f[f["target"] == "offshore"].set_index("time")

if os.environ.get("PRICES"):
    px = pd.read_csv(os.environ["PRICES"])
    px["time"] = pd.to_datetime(px["time"], format="mixed").dt.tz_localize(None)
else:
    js = get("https://api.energidataservice.dk/dataset/ImbalancePrice",
             {"start": f"{f.index.min():%Y-%m-%d}", "end": f"{f.index.max() + pd.Timedelta(days=1):%Y-%m-%d}",
              "limit": 0, "filter": '{"PriceArea":["DK1"]}'})
    raw = pd.DataFrame(js["records"])
    px = pd.DataFrame({"time": pd.to_datetime(raw["TimeUTC"]).dt.floor("h"), "spot": raw["SpotPriceEUR"],
                       "imb": raw["ImbalancePriceEUR"]}).groupby("time", as_index=False).mean()
d = f.join(px.set_index("time"), how="inner").dropna(subset=["spot", "imb", "forecast", "energinet_scaled"])

# sold f at spot, the miss (actual - f) settles at the imbalance price. against knowing the output in advance
# the cost is -(actual - f) * (imb - spot), in euro per hour
rows = []
for k, name in [("forecast", "Ours"), ("energinet_scaled", "Energinet, scaled to the fleet"), ("energinet", "Energinet, as published")]:
    cost = -(d["actual"] - d[k]) * (d["imb"] - d["spot"])
    daily = cost.groupby(d.index.date).sum()
    rows.append({"forecast": name, "hours": len(d), "eur_per_mw_year": cost.sum() / d["cap"].mean() / len(d) * 8760,
                 "eur_per_mwh": cost.sum() / d["actual"].sum(), "daily_sd": daily.std()})
out = pd.DataFrame(rows)
# is the gap to energinet real? newey-west on the daily difference
diff = ((-(d["actual"] - d["energinet_scaled"]) * (d["imb"] - d["spot"])) - (-(d["actual"] - d["forecast"]) * (d["imb"] - d["spot"])))
dd = diff.groupby(d.index.date).sum().to_numpy()
n, u = len(dd), dd - dd.mean()
v = (u @ u) / n + sum(2 * (1 - k / 8) * (u[k:] @ u[:-k]) / n for k in range(1, 8))
out["t_vs_ours"] = [np.nan, float(dd.mean() / np.sqrt(v / n)), np.nan]
print(out.round(3).to_string(index=False))
out.round(3).to_csv("results/value.csv", index=False)
