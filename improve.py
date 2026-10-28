# trying to beat energinet, by the rules in the readme
#   python improve.py        tries each addition on march-december 2025 and saves what is kept

import json
import sys
from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd

TARGETS = ["onshore", "offshore", "solar"]
LAND = ["west_jutland", "north_jutland", "mid_jutland", "south_jutland", "east_jutland", "funen"]
SEA = ["horns_rev", "thor", "anholt"]
TUNE = pd.date_range("2025-03-01", "2025-12-01", freq="MS")
DELAY = 10
PARAMS = dict(objective="l1", n_estimators=600, learning_rate=0.05, num_leaves=31, min_child_samples=50,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, random_state=1)
PARAM_TRIES = {"fewer leaves": dict(num_leaves=15), "more leaves": dict(num_leaves=63, min_child_samples=100),
               "slower": dict(learning_rate=0.025, n_estimators=1200)}
Path("results").mkdir(exist_ok=True)

con = duckdb.connect("data/dk1.duckdb", read_only=True)
prod = con.execute("SELECT * FROM production ORDER BY time").df().set_index("time")
energinet = con.execute("SELECT * FROM energinet").df().set_index("time")
base = con.execute("SELECT * FROM weather").df()
runs = con.execute("SELECT * FROM weather_runs").df()
models = con.execute("SELECT * FROM weather_models").df()
con.close()
if runs.empty or models.empty:
    sys.exit("run get_weather_runs.py and get_weather_models.py first")


def local_date(t):
    return pd.to_datetime(t.tz_localize("UTC").tz_convert("Europe/Copenhagen").date)


def wide(w, prefix):
    w = w.copy()
    angle = np.radians(w.pop("direction_100m"))
    w["dir_sin"], w["dir_cos"] = np.sin(angle), np.cos(angle)
    x = w.pivot(index="time", columns="point")
    x.columns = [f"{prefix}{v}_{p}" for v, p in x.columns]
    for name, pts in [("land", LAND), ("sea", SEA)]:
        x[f"{prefix}wind_{name}"] = x[[f"{prefix}wind_100m_{p}" for p in pts]].mean(axis=1)
        x[f"{prefix}wind_{name}_sd"] = x[[f"{prefix}wind_100m_{p}" for p in pts]].std(axis=1)
    x[f"{prefix}sun"] = x[[f"{prefix}radiation_{p}" for p in LAND]].mean(axis=1)
    return x


# weather made two days before (base), and ecmwf's 00 utc run from the day before each danish day
B = wide(base, "")
runs = runs[runs["run"] == local_date(pd.DatetimeIndex(runs["time"])) - pd.Timedelta(days=1)].drop(columns="run")
E = wide(runs, "ec_")

# other models, averaged over land and sea
mod = []
for m, g in models.groupby("model"):
    land = g[g["point"].isin(LAND)].groupby("time")
    mod.append(pd.DataFrame({f"{m}_wind_land": land["wind_100m"].mean(), f"{m}_sun": land["radiation"].mean(),
                             f"{m}_cloud": land["cloud_cover"].mean(),
                             f"{m}_wind_sea": g[g["point"].isin(SEA)].groupby("time")["wind_100m"].mean()}))
M = pd.concat(mod, axis=1)

X = B.join(E, how="left").join(M, how="left")
local = X.index.tz_localize("UTC").tz_convert("Europe/Copenhagen")
X["hour"], X["day_of_year"], X["date"] = local.hour, local.dayofyear, local_date(X.index)

# the sun's height mid-hour over mid-jutland, and how clear the sky is forecast to be
mid = X.index - pd.Timedelta(minutes=30)
decl = np.radians(23.44) * np.sin(2 * np.pi * (284 + mid.dayofyear) / 365)
angle = np.radians(15 * (mid.hour + mid.minute / 60 + 9.5 / 15 - 12))
lat = np.radians(56.2)
sin_h = np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl) * np.cos(angle)
X["sun_height"] = np.degrees(np.arcsin(sin_h))
clear = np.where(sin_h > 0.05, 1098 * sin_h * np.exp(-0.057 / np.clip(sin_h, 0.05, None)), np.nan)
for p in ["", "ec_"]:
    X[f"{p}clear_sky"] = np.clip(X[f"{p}sun"] / clear, 0, 1.5)

CALENDAR = ["hour", "day_of_year"]


def columns(cfg):
    p, src = ("ec_", E.columns) if cfg["weather"] == "ecmwf" else ("", B.columns)
    cols = [c for c in src if cfg["extras"] or c not in {p + "wind_land_sd", p + "wind_sea_sd"}]
    if cfg["extras"]:
        cols += [p + "clear_sky", "sun_height"]
    if cfg["models"]:
        cols += list(M.columns)
    return cols + CALENDAR


def fleet(y):
    daily = y.groupby(local_date(y.index)).max()
    return daily.rolling("60D").max().shift(DELAY, freq="D")


def backtest(t, cfg, months):
    cap = fleet(prod[t])
    d = X.join(prod[t].rename("actual"), how="inner")
    d["cap"] = d["date"].map(cap)
    d = d.dropna(subset=["actual", "cap", "wind_land"])
    d["share"] = d["actual"] / d["cap"]
    cols = columns(cfg)
    out = []
    for month in months:
        train = d[d["date"] <= month - pd.Timedelta(days=DELAY)]
        test = d[(d["date"] >= month) & (d["date"] < month + pd.offsets.MonthBegin())]
        if test.empty:
            continue
        m = lgb.LGBMRegressor(**{**PARAMS, **PARAM_TRIES.get(cfg["params"], {})})
        m.fit(train[cols], train["share"])
        out.append(pd.DataFrame({"actual": test["actual"], "cap": test["cap"], "date": test["date"],
                                 "forecast": np.clip(m.predict(test[cols]), 0, 1) * test["cap"]}))
    return pd.concat(out)


def mae(f):
    return (f["forecast"] - f["actual"]).abs().mean()


if len(sys.argv) == 1:
    log, chosen = [], {}
    for t in TARGETS:
        cfg = {"weather": "base", "models": False, "extras": False, "params": "base"}
        best = mae(backtest(t, cfg, TUNE))
        log.append({"target": t, "try": "start", "mae": best, "kept": True})
        print(f"{t} start: {best:.1f}")
        tries = [("ecmwf 00 utc run", {"weather": "ecmwf"}), ("other models", {"models": True}),
                 ("more features", {"extras": True})] + [(k, {"params": k}) for k in PARAM_TRIES]
        for name, change in tries:
            score = mae(backtest(t, {**cfg, **change}, TUNE))
            kept = score < best
            log.append({"target": t, "try": name, "mae": score, "kept": kept})
            print(f"{t} {name}: {score:.1f} {'kept' if kept else ''}")
            if kept:
                cfg, best = {**cfg, **change}, score
        chosen[t] = cfg
    pd.DataFrame(log).round(2).to_csv("results/improve_tuning.csv", index=False)
    Path("results/improve_choice.json").write_text(json.dumps(chosen, indent=2))
    print(json.dumps(chosen, indent=2))
