# dashboard for the offshore wind forecast: docs/index.html (live forecasts and their track record) and
# docs/backtest.html (2026, run once, with onshore and solar as tried but not beaten)
#   python dashboard.py

from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import plotly.graph_objects as go

ACCENT, GREEN, TEAL, RED = "#62d6e8", "#4fd08a", "#d45c88", "#ff7a7a"
INK, GREY, LINE, GRID, BEIGE = "#e9f0f7", "#8fa6bd", "rgba(255,255,255,0.15)", "rgba(255,255,255,0.07)", "#4a6079"
FONT = "Instrument Sans, Segoe UI, Helvetica Neue, Arial, sans-serif"
PLOTLY_JS = "https://cdn.jsdelivr.net/npm/plotly.js-dist-min@4.1.1/plotly.min.js"
TARGETS = {"offshore": "Offshore wind"}
DELAY = 10
GITHUB_ICON = ('<svg width="18" height="18" viewBox="0 0 16 16" fill="currentColor"><path d="M8 0C3.58 0 0 3.58 0 8c0 '
               '3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94'
               '-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78'
               '-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27'
               's1.36.09 2 .27c1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 '
               '3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8"/>'
               '</svg>')


def local(t):
    return pd.DatetimeIndex(t).tz_localize("UTC").tz_convert("Europe/Copenhagen")


def mw(x):
    return f"{x:,.0f} MW"


def style(fig, height, legend_y=-0.18):
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=10, b=10), font=dict(family=FONT, color=GREY, size=12),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", hovermode="x unified",
        legend=dict(orientation="h", x=0, y=legend_y, font=dict(color=INK)),
        hoverlabel=dict(bgcolor="#15324f", bordercolor="#2b4a6b", font=dict(color=INK, family=FONT)))
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor=LINE, color=GREY)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=BEIGE, color=GREY)
    return fig.to_html(full_html=False, include_plotlyjs=False, config=dict(displayModeBar=False, responsive=True))


def kpi(label, value, sub, accent):
    return (f"<div class='card kpi' style='--accent:{accent}'><div class='kpi-label'>{label}</div>"
            f"<div class='kpi-value'>{value}</div><div class='kpi-sub'>{sub}</div></div>")


def card(title, body, note=""):
    return f"<div class='card'><div class='card-title'>{title}</div>{body}{note and f'<p class=note>{note}</p>'}</div>"


def scaled(energinet, actual):
    """energinet's forecast scaled to the whole fleet: output over its forecast in the 60 days up to 10 days before"""
    both = pd.concat([actual.rename("a"), energinet.rename("e")], axis=1).dropna()
    daily = both.groupby(local(both.index).date).sum()
    daily.index = pd.to_datetime(daily.index)
    ratio = (daily["a"].rolling("60D").sum() / daily["e"].rolling("60D").sum()).shift(DELAY, freq="D")
    ratio = ratio.reindex(pd.date_range(ratio.index.min(), ratio.index.max() + pd.Timedelta(days=DELAY))).ffill()
    days = pd.to_datetime(local(energinet.index).date)
    return energinet * ratio.reindex(days).to_numpy()


# charts shared by both pages

def trend_chart(f, window):
    """rolling mean absolute error by day: is the gap to energinet steady or changing"""
    d = f.assign(day=pd.to_datetime(local(f.index).date))
    err = pd.DataFrame({k: (d[k] - d["actual"]).abs() for k in ["forecast", "energinet", "energinet_scaled"]})
    daily = err.groupby(d["day"]).mean().rolling(window, min_periods=max(1, window // 2)).mean()
    fig = go.Figure()
    for k, name, color, dash in [("energinet", "Energinet, as published", GREY, "dot"),
                                 ("energinet_scaled", "Energinet, scaled to the fleet", GREY, "solid"),
                                 ("forecast", "Ours", ACCENT, "solid")]:
        fig.add_trace(go.Scatter(x=daily.index, y=daily[k], name=name, mode="lines",
                                 line=dict(color=color, width=2.5 if k == "forecast" else 1.6, dash=dash),
                                 hovertemplate=f"{name}: %{{y:,.0f}} MW<extra></extra>"))
    fig.update_yaxes(ticksuffix=" MW", rangemode="tozero")
    return style(fig, 320)


def scatter_chart(f, cap):
    """forecast against what was produced, hour by hour: on the diagonal is a perfect forecast"""
    s = f.sample(min(len(f), 2500), random_state=1).sort_index()
    top = max(f["actual"].max(), f["forecast"].max(), f["energinet"].max()) * 1.03
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, top], y=[0, top], mode="lines", line=dict(color=BEIGE, dash="dash"),
                             hoverinfo="skip", showlegend=False))
    for k, name, color in [("energinet", "Energinet, as published", GREY), ("forecast", "Ours", ACCENT)]:
        fig.add_trace(go.Scatter(x=s[k], y=s["actual"], name=name, mode="markers",
                                 marker=dict(color=color, size=4, opacity=0.35),
                                 hovertemplate=f"{name} %{{x:,.0f}} MW, produced %{{y:,.0f}} MW<extra></extra>"))
    fig.update_layout(hovermode="closest")
    fig.update_xaxes(title="forecast, MW", range=[0, top])
    fig.update_yaxes(title="produced, MW", range=[0, top])
    return style(fig, 360, -0.32)


def level_chart(f):
    """error by how much was forecast, as a share of the fleet: where on the power curve the misses are"""
    share = (f["forecast"] / f["cap"]).clip(0, 0.999)
    bins = (share * 10).astype(int) * 10
    err = pd.DataFrame({k: (f[k] - f["actual"]).abs() for k in ["forecast", "energinet_scaled"]})
    g = err.groupby(bins).mean()
    n = bins.value_counts().reindex(g.index)
    labels = [f"{b}-{b + 10}%" for b in g.index]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=g["energinet_scaled"], name="Energinet, scaled to the fleet",
                         marker=dict(color=BEIGE, cornerradius=4), customdata=n,
                         hovertemplate="%{x}: %{y:,.0f} MW (%{customdata} hours)<extra></extra>"))
    fig.add_trace(go.Bar(x=labels, y=g["forecast"], name="Ours", marker=dict(color=ACCENT, cornerradius=4),
                         customdata=n, hovertemplate="%{x}: %{y:,.0f} MW (%{customdata} hours)<extra></extra>"))
    fig.update_layout(barmode="group", hovermode="closest")
    fig.update_xaxes(title="our forecast as a share of the fleet", tickangle=0)
    fig.update_yaxes(ticksuffix=" MW")
    return style(fig, 300, -0.32)


def hour_chart(f):
    """error by hour of the day, danish time"""
    h = local(f.index).hour
    g = pd.DataFrame({k: (f[k] - f["actual"]).abs() for k in ["forecast", "energinet_scaled"]}).groupby(h).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=g.index, y=g["energinet_scaled"], name="Energinet, scaled to the fleet",
                             line=dict(color=GREY, width=1.6), hovertemplate="%{y:,.0f} MW<extra></extra>"))
    fig.add_trace(go.Scatter(x=g.index, y=g["forecast"], name="Ours", line=dict(color=ACCENT, width=2.5),
                             hovertemplate="%{y:,.0f} MW<extra></extra>"))
    fig.update_xaxes(title="hour, Danish time", dtick=3)
    fig.update_yaxes(ticksuffix=" MW", rangemode="tozero")
    return style(fig, 280, -0.32)


def gap(ours, other):
    d = 1 - ours / other
    return f"ours is {abs(d):.0%} {'lower' if d >= 0 else 'higher'}"


def scores(f):
    e = lambda k: (f[k] - f["actual"]).abs().mean()
    return e("forecast"), e("energinet"), e("energinet_scaled")


def error_bands(bt_f, live_f, cap):
    """5/25/75/95% of the miss at each output level, from the 2026 backtest. hours with a similar forecast
    (as a share of the fleet) are pooled: the miss is small near zero and full output, widest in between"""
    share = (bt_f["forecast"] / bt_f["cap"]).clip(0, 1).to_numpy()
    err = ((bt_f["actual"] - bt_f["forecast"]) / bt_f["cap"]).to_numpy()
    out = []
    for x in (live_f / cap).clip(0, 1):
        near = np.abs(share - x) <= 0.1
        q = np.quantile(err[near], [0.05, 0.25, 0.75, 0.95]) if near.sum() >= 50 else [np.nan] * 4
        out.append(np.clip((x + np.array(q)) * cap, 0, cap))
    return pd.DataFrame(out, index=live_f.index, columns=["q05", "q25", "q75", "q95"])


def daily_chart(f):
    """mean absolute error per day; pale bars: scored on real-time output, settled numbers still to come"""
    d = f.assign(day=local(f.index).normalize().tz_localize(None))
    g = d.groupby("day").apply(lambda x: pd.Series({k: (x[k] - x["actual"]).abs().mean() for k in
                               ["forecast", "energinet", "energinet_scaled"]} | {"prov": x["provisional"].any()}))
    fig = go.Figure()
    for col, nm, c in [("energinet", "Energinet, as published", BEIGE), ("energinet_scaled", "Energinet, scaled", GREY),
                       ("forecast", "Ours", ACCENT)]:
        fig.add_trace(go.Bar(x=g.index, y=g[col], name=nm, marker=dict(color=c, cornerradius=3,
                             opacity=[0.45 if p else 1 for p in g["prov"]]),
                             hovertemplate=nm + " %{y:,.0f} MW<extra></extra>"))
    fig.update_layout(barmode="group", bargap=0.3)
    fig.update_xaxes(tickformat="%d %b")
    fig.update_yaxes(ticksuffix=" MW")
    return style(fig, 280) + ("<p class=note>Pale: scored on real-time output, settled numbers still to come.</p>"
                              if g["prov"].any() else "")


def last_day_chart(f):
    days = local(f.index).normalize()
    d = f[days == days.max()]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=local(d.index), y=d["actual"], name="Produced", line=dict(color=INK, width=2.5, shape="hv"),
                             hovertemplate="Produced %{y:,.0f} MW<extra></extra>"))
    fig.add_trace(go.Scatter(x=local(d.index), y=d["energinet"], name="Energinet", line=dict(color=GREY, width=1.6,
                             dash="dot", shape="hv"), hovertemplate="Energinet %{y:,.0f} MW<extra></extra>"))
    fig.add_trace(go.Scatter(x=local(d.index), y=d["forecast"], name="Ours", line=dict(color=ACCENT, width=2.5, shape="hv"),
                             hovertemplate="Ours %{y:,.0f} MW<extra></extra>"))
    fig.update_yaxes(ticksuffix=" MW", rangemode="tozero")
    return style(fig, 280)


# live page

bt_all = pd.read_csv("results/final_forecasts.csv", parse_dates=["time"]).set_index("time")
log = pd.read_csv("forecasts/live_forecasts.csv", parse_dates=["date", "time"]) \
    if Path("forecasts/live_forecasts.csv").exists() else pd.DataFrame(columns=["date", "time", "target", "forecast"])
prod = energinet = rt = None
if Path("data/dk1.duckdb").exists():
    con = duckdb.connect("data/dk1.duckdb", read_only=True)
    prod = con.execute("SELECT * FROM production ORDER BY time").df().set_index("time")
    energinet = con.execute("SELECT * FROM energinet ORDER BY time").df().set_index("time")
    tables = con.execute("SELECT table_name FROM information_schema.tables").df()["table_name"].tolist()
    if "production_rt" in tables:
        rt = con.execute("SELECT * FROM production_rt ORDER BY time").df().set_index("time")
    con.close()

live_tabs = {}
latest = log["date"].max() if len(log) else None
for t, name in TARGETS.items():
    lt = log[log["target"] == t].set_index("time").sort_index()
    blocks = []
    if latest is not None:
        day = lt[lt["date"] == latest]
        fig = go.Figure()
        if energinet is not None and energinet[t].reindex(day.index).notna().any():
            fig.add_trace(go.Scatter(x=local(day.index), y=energinet[t].reindex(day.index), name="Energinet, as published",
                                     line=dict(color=GREY, width=1.6, dash="dot", shape="hv"),
                                     hovertemplate="Energinet %{y:,.0f} MW<extra></extra>"))
        b = error_bands(bt_all[bt_all["target"] == t], day["forecast"], day["cap"].iloc[0])
        for lo, hi, a, nm in [("q05", "q95", .10, "9 in 10 hours inside"), ("q25", "q75", .20, "Half the hours inside")]:
            fig.add_trace(go.Scatter(x=local(day.index), y=b[hi], line=dict(width=0, shape="hv"), showlegend=False,
                                     hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=local(day.index), y=b[lo], line=dict(width=0, shape="hv"), fill="tonexty",
                                     fillcolor=f"rgba(98,214,232,{a})", name=nm, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=local(day.index), y=day["forecast"], name="Ours", line=dict(color=ACCENT, width=2.5,
                                 shape="hv"), customdata=b[["q05", "q95"]].to_numpy(),
                                 hovertemplate="Ours %{y:,.0f} MW · 9 in 10: %{customdata[0]:,.0f} to %{customdata[1]:,.0f}<extra></extra>"))
        fig.update_yaxes(ticksuffix=" MW", rangemode="tozero")
        made = pd.Timestamp(day["issued"].iloc[0]).tz_localize("UTC").tz_convert("Europe/Copenhagen")
        blocks.append("<div class='kpis'>" +
                      kpi(f"{name}, {latest:%a %d %b}", mw(day["forecast"].mean()), "average over the day", ACCENT) +
                      kpi("Highest hour", mw(day["forecast"].max()),
                          f"at {local(day.index)[day['forecast'].to_numpy().argmax()]:%H:%M}", ACCENT) +
                      kpi("Fleet size used", mw(day["cap"].iloc[0]), "highest hour in the period before", GREY) +
                      "</div>")
        blocks.append(card(f"Forecast for {latest:%A %d %B}, made {made:%d %b %H:%M} (Danish time)", style(fig, 320),
                           "LightGBM on ECMWF, ICON-EU and GFS weather, made before the noon auction. Energinet's forecast "
                           "is added when it is published, after the auction. How it works: Method tab."))
    # track record: settled output where it is out (8-10 days late), real-time output until then
    done = pd.DataFrame()
    if prod is not None and len(lt):
        actual = prod[t]
        if rt is not None:
            actual = actual.combine_first(rt[t])
        done = lt[["forecast", "cap"]].join(actual.rename("actual"), how="inner")
        done["provisional"] = ~done.index.isin(prod[t].dropna().index)
        done["energinet"] = energinet[t].reindex(done.index)
        done["energinet_scaled"] = scaled(energinet[t], prod[t]).reindex(done.index)
        done = done.dropna(subset=["forecast", "actual", "energinet"])
        done["energinet_scaled"] = done["energinet_scaled"].fillna(done["energinet"])
    if len(done) >= 24:
        ours, en, ens = scores(done)
        days = local(done.index).normalize()
        n, k = len(days.unique()), len(days[done["provisional"]].unique())
        sub = f"{n} day{'s' * (n != 1)}" + (f", {k} provisional" if k else "")
        blocks.append("<div class='kpis'>" +
                      kpi("Our error since launch", mw(ours), sub, ACCENT) +
                      kpi("Energinet, as published", mw(en), gap(ours, en), GREY) +
                      kpi("Energinet, scaled to the fleet", mw(ens), gap(ours, ens), GREY) + "</div>")
        blocks.append("<div class='grid2'>" + card("Error each day", daily_chart(done)) +
                      card(f"Last day, {days.max():%a %d %b}", last_day_chart(done)) + "</div>")
        if n >= 14:
            blocks.append(card("Error by day, 7-day average", trend_chart(done, 7)))
        blocks.append(card("Forecast against what was produced, every live hour", scatter_chart(done, done["cap"])))
    else:
        blocks.append(card("Track record", "<p>Scored from the day after the first forecast: first with Energinet's "
                                           "real-time output, then with the settled numbers 8 to 10 days later.</p>"))
    live_tabs[name] = "".join(blocks)


# backtest page: 2026, run once

bt = pd.read_csv("results/final_forecasts.csv", parse_dates=["time"]).set_index("time")
fs = pd.read_csv("results/final_scores.csv").set_index("target")
bt_tabs = {}
for t, name in TARGETS.items():
    f = bt[bt["target"] == t]
    ours, en, ens = scores(f)
    r = fs.loc[t]
    verdict = ("Beaten: lower error than both, Diebold-Mariano p < 0.05" if r["beaten"] else
               f"Not beaten (p = {r['p_vs_energinet']:.2f} against the published forecast, "
               f"{r['p_vs_scaled']:.2f} against the scaled one)")
    month = f.groupby(local(f.index).strftime("%Y-%m")).apply(
        lambda g: pd.Series({k: (g[k] - g["actual"]).abs().mean() for k in ["forecast", "energinet_scaled"]}))
    figm = go.Figure()
    figm.add_trace(go.Bar(x=pd.to_datetime(month.index).strftime("%b"), y=month["energinet_scaled"],
                          name="Energinet, scaled to the fleet", marker=dict(color=BEIGE, cornerradius=4),
                          hovertemplate="%{y:,.0f} MW<extra></extra>"))
    figm.add_trace(go.Bar(x=pd.to_datetime(month.index).strftime("%b"), y=month["forecast"], name="Ours",
                          marker=dict(color=ACCENT, cornerradius=4), hovertemplate="%{y:,.0f} MW<extra></extra>"))
    figm.update_layout(barmode="group")
    figm.update_yaxes(ticksuffix=" MW")
    bt_tabs[name] = (
        "<div class='kpis'>" +
        kpi("Our error", mw(ours), "mean absolute error per hour", ACCENT) +
        kpi("Energinet, as published", mw(en), gap(ours, en), GREY) +
        kpi("Energinet, scaled to the fleet", mw(ens), gap(ours, ens), GREY) +
        "</div>" +
        card("Verdict", f"<p>{verdict}.</p>") +
        card("Is the gap steady? Error by day, 14-day average", trend_chart(f, 14),
             "Energinet's published forecast seems to miss part of the fleet; scaled up, it is what a trader could "
             "get by correcting it with the last 60 settled days.") +
        "<div class='grid2'>" +
        card("Forecast against what was produced (2,500 random hours)", scatter_chart(f, f["cap"]),
             "On the dashed line the forecast was exactly right.") +
        card("Where on the power curve the misses are", level_chart(f),
             "Error by how much we forecast, as a share of the fleet.") +
        "</div><div class='grid2'>" +
        card("Error by month", style(figm, 300)) +
        card("Error by hour of the day", hour_chart(f)) +
        "</div>")


# onshore and solar: tried on the same rules, not beaten, so not run live
names = {"onshore": "Onshore wind", "solar": "Solar", "offshore": "Offshore wind"}
rows = "".join(
    f"<tr><td>{names[t]}</td><td class=n>{mw(r['ours'])}</td><td class=n>{mw(r['energinet'])}</td>"
    f"<td class=n>{mw(r['energinet_scaled'])}</td><td>{'yes' if r['beaten'] else 'no'}</td></tr>"
    for t, r in fs.loc[["offshore", "onshore", "solar"]].iterrows())
TRIED = card("Also tried: onshore wind and solar",
             "<table><tr><th>2026, mean absolute error</th><th class=n>Ours</th><th class=n>Energinet</th>"
             f"<th class=n>Energinet scaled</th><th>Beaten</th></tr>{rows}</table>",
             "The same rules and the same weather were used for all three. Onshore was lower than Energinet's "
             "published forecast but not significantly (p = 0.06), and behind it scaled to the fleet. Solar was "
             "behind both. Energinet knows every wind farm and solar plant and forecasts later, so only offshore, "
             "where its forecast misses part of the fleet, runs live.")
if Path("results/value.csv").exists():
    val = pd.read_csv("results/value.csv").set_index("forecast")
    order = ["Ours", "Energinet, scaled to the fleet", "Energinet, as published"]
    cols = [ACCENT, GREY, BEIGE]

    def vbar(col, fmt, suffix):
        fig = go.Figure(go.Bar(x=val.loc[order, col], y=order, orientation="h", marker=dict(color=cols, cornerradius=4),
                               text=[fmt(v) for v in val.loc[order, col]], textposition="outside", cliponaxis=False,
                               hovertemplate="%{y}: %{text}<extra></extra>"))
        fig.update_yaxes(autorange="reversed")
        fig.update_xaxes(range=[0, val[col].max() * 1.3], ticksuffix=suffix)
        return style(fig, 200)

    bt_tabs["Offshore wind"] += ("<div class='grid2'>" +
        card("What the misses cost: imbalance bill per MWh produced", vbar("eur_per_mwh", lambda v: f"{v:.2f} €", " €"),
             "A wind farm that sells its forecast day-ahead settles the miss at the imbalance price. The average bill "
             "is about the same for all three: it depends on which way the grid leans, not on the size of the miss.") +
        card("How much the bill swings from day to day", vbar("daily_sd", lambda v: f"{v / 1000:,.0f} k€", ""),
             "Standard deviation of the daily bill for the whole fleet. A smaller miss means a steadier bill.") +
        "</div>")
bt_tabs["Offshore wind"] += TRIED
live_tabs["Offshore wind"] += card("Why only offshore", "<p>Onshore wind and solar were tried with the same rules "
                                   "but didn't beat Energinet on 2026. See the <a href='backtest.html'>backtest</a>."
                                   "</p>")


# page

def render(title, tabs, link, banner=""):
    built = datetime.now(timezone.utc)
    nav = "".join(f"<button data-tab='t{i}'{' class=active' if i == 0 else ''}>{k}</button>" for i, k in enumerate(tabs))
    panes = "".join(f"<section id='t{i}'{' class=active' if i == 0 else ''}>{v}</section>"
                    for i, v in enumerate(tabs.values()))
    updated = pd.Timestamp(built).tz_convert("Europe/Copenhagen").strftime("%d %b %Y %H:%M")
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Instrument+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
<script src="{PLOTLY_JS}"></script>
<style>
  html {{ background: #0a1626; }}
  body {{ margin: 0; min-height: 100vh; color: {INK}; font: 15px/1.5 {FONT};
         background: radial-gradient(1200px 700px at 70% 0%, #15324f 0%, #0c1c30 55%, #0a1626 100%) fixed; }}
  header {{ padding: 10px 20px 0; display: flex; flex-wrap: wrap; align-items: center; gap: 4px 20px; }}
  header h1 {{ font-size: 17px; font-weight: 600; margin: 10px 8px 10px 0; }}
  nav {{ display: flex; flex-wrap: wrap; gap: 2px; }}
  nav button {{ font: 500 14.5px {FONT}; color: {GREY}; background: none; border: 0; padding: 7px 15px; border-radius: 99px; cursor: pointer; }}
  nav button:hover {{ color: {INK}; }}
  nav button.active {{ color: #0c1c30; background: {INK}; }}
  header .links {{ margin-left: auto; font-size: 14px; display: flex; align-items: center; gap: 16px; }}
  header .links a {{ color: {GREY}; text-decoration: none; }}
  header .links svg {{ display: block; fill: {INK}; }}
  .banner {{ background: rgba(98,214,232,0.08); border: 1px solid rgba(98,214,232,0.25); color: #c9d6e2; font-size: 14px;
            padding: 8px 16px; border-radius: 10px; max-width: 1248px; margin: 12px auto 0; }}
  main {{ max-width: 1280px; margin: 0 auto; padding: 18px 16px 40px; }}
  section {{ display: none; }}
  section.active {{ display: block; }}
  .card {{ background: rgba(255,255,255,0.035); border: 1px solid rgba(255,255,255,0.07); border-radius: 14px;
          padding: 16px 20px; margin-bottom: 18px; min-width: 0; }}
  .card-title {{ font-weight: 500; margin-bottom: 8px; }}
  .kpis {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 18px; }}
  .kpi-label {{ color: {GREY}; font-size: 13px; }}
  .kpi-value {{ font-size: 30px; font-weight: 500; letter-spacing: -0.02em; margin: 2px 0; font-variant-numeric: tabular-nums; }}
  .kpi-sub {{ color: {GREY}; font-size: 13px; }}
  .grid2 {{ display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 18px; }}
  .grid2 > * {{ min-width: 0; }}
  @media (max-width: 900px) {{ .grid2 {{ grid-template-columns: 1fr; }} header .links {{ margin-left: 0; }} }}
  table {{ border-collapse: collapse; width: 100%; font-size: 14px; }}
  th, td {{ text-align: left; padding: 7px 8px 7px 0; border-bottom: 1px solid rgba(255,255,255,0.08); }}
  th {{ color: {GREY}; font-weight: 500; font-size: 13px; }}
  .n {{ text-align: right; font-variant-numeric: tabular-nums; }}
  p {{ color: #c9d6e2; }}
  .note {{ color: {GREY}; font-size: 13px; max-width: 900px; margin-bottom: 0; }}
  a {{ color: {ACCENT}; }}
  footer {{ color: {GREY}; font-size: 12px; text-align: center; padding-bottom: 24px; }}
{METHOD_CSS}  .stale {{ position: fixed; left: 0; right: 0; bottom: 0; z-index: 1000; background: #3a1620; color: #ffb3b3;
           border-top: 1px solid #6b2a35; font-size: 14px; padding: 10px 20px; text-align: center; }}
</style>
</head>
<body>
<header><h1>{title}</h1><nav>{nav}</nav>
<span class="links"><a href="https://jacobazu.github.io/">jacobazu.github.io</a>{link}<a href="https://github.com/jacobazu/dk1-renewables-forecasting" aria-label="GitHub">{GITHUB_ICON}</a></span></header>
{banner}
<main>{panes}</main>
<footer>Updated {updated} (Danish time). Data: Energi Data Service (Energinet), Open-Meteo.com (CC BY 4.0)</footer>
<div id="stale" class="stale" data-built="{built.isoformat()}" hidden></div>
<script>
(() => {{
  const el = document.getElementById("stale"), built = new Date(el.dataset.built);
  if (Date.now() - built > 30 * 3600 * 1000) {{
    el.textContent = "Not updated since " + built.toLocaleString("en-GB", {{ timeZone: "Europe/Copenhagen",
      day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" }}) + " (Danish time). The daily update has not run.";
    el.hidden = false;
  }}
}})();
document.querySelectorAll("nav button").forEach(b => b.onclick = () => {{
  document.querySelectorAll("nav button, section").forEach(e => e.classList.remove("active"));
  b.classList.add("active");
  document.getElementById(b.dataset.tab).classList.add("active");
  document.querySelectorAll("section.active .plotly-graph-div").forEach(g => Plotly.Plots.resize(g));
}});
</script>
</body>
</html>
"""


# method tab: the pipeline as boxes, then the facts

def step(n, title, body):
    return (f"<div class='step'><div class='num'>{n}</div><div><div class='st'>{title}</div>"
            f"<div class='sb'>{body}</div></div></div>")


ro = fs.loc["offshore"]
facts = [("Model", "LightGBM (gradient-boosted trees), absolute-error loss, 1,200 trees at learning rate 0.025"),
         ("Target", "Offshore output as a share of the fleet, times the fleet size"),
         ("Weather", "ECMWF's 00 UTC run from the morning before (ready about 07 UTC), plus ICON-EU and GFS. "
                     "Wind at 100 m and 10 m, direction, gusts and the spread over the points"),
         ("Points", "Horns Rev, Thor and Anholt at sea, plus six points over Jutland and Funen"),
         ("Fleet size", "The highest hourly output in the 60 days before, so new parks such as Thor are picked up"),
         ("Training", "All hours from March 2024 up to 10 days before the first of the month (settled output is "
                      "8-10 days late). Refit monthly"),
         ("Chosen on", "2025 only. Tested once on 2026, then frozen"),
         ("Scored by", "Mean absolute error per hour against Energinet's settled output; Energinet's own "
                       "day-ahead forecast as the yardstick")]
bars = go.Figure(go.Bar(x=[ro["ours"], ro["energinet_scaled"], ro["energinet"]],
                        y=["Ours", "Energinet, scaled", "Energinet, as published"], orientation="h",
                        marker=dict(color=[ACCENT, GREY, BEIGE], cornerradius=4),
                        text=[mw(v) for v in [ro["ours"], ro["energinet_scaled"], ro["energinet"]]],
                        textposition="outside", cliponaxis=False, hovertemplate="%{y}: %{x:,.0f} MW<extra></extra>"))
bars.update_yaxes(autorange="reversed")
bars.update_xaxes(ticksuffix=" MW", range=[0, ro["energinet"] * 1.25])
live_tabs["Method"] = (
    "<div class='card'><div class='card-title'>How tomorrow's forecast is made, every morning</div><div class='flow'>" +
    step(1, "07:00", "ECMWF's morning run is out") +
    step(2, "10:07", "Weather at 9 points goes into the model") +
    step(3, "Share × fleet", "The model gives output as a share of the fleet") +
    step(4, "Saved", "Before the noon auction, never changed") +
    step(5, "Scored", "Real-time output the next day, settled numbers 8-10 days later") +
    "</div></div>" +
    "<div class='grid2'>" +
    card("The model", "<table class='facts'>" + "".join(f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in facts) + "</table>") +
    card("Backtest 2026: error per hour", style(bars, 220),
         "Lower is better. <a href='backtest.html'>Full backtest</a>.") +
    "</div>")
METHOD_CSS = """
  .flow { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 10px; }
  @media (max-width: 900px) { .flow { grid-template-columns: 1fr; } }
  .step { display: flex; gap: 10px; align-items: flex-start; background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.07); border-radius: 12px; padding: 10px; }
  .step .num { flex: 0 0 26px; height: 26px; border-radius: 50%; background: #62d6e8; color: #0c1c30; font-weight: 600;
               display: grid; place-items: center; font-size: 13px; }
  .step .st { font-weight: 600; font-size: 14px; }
  .step .sb { color: #8fa6bd; font-size: 13px; line-height: 1.35; }
  .facts th { width: 110px; vertical-align: top; }
"""


Path("docs").mkdir(exist_ok=True)
Path("docs/index.html").write_text(render("DK1 offshore wind forecast", live_tabs,
                                          "<a href='backtest.html'>Backtest</a>"), encoding="utf-8")
span = f"{local(bt.index).min():%d %b %Y} to {local(bt.index - pd.Timedelta(hours=1)).max():%d %b %Y}"
Path("docs/backtest.html").write_text(render(
    "DK1 offshore wind forecast: backtest", bt_tabs, "<a href='index.html'>&larr; Live</a>",
    f"<div class='banner'>Tested on {span}. The model was picked on earlier data, then run once. Each forecast used "
    f"only weather out before the auction.</div>"), encoding="utf-8")
print(f"docs/index.html: latest forecast {latest:%d %b %Y}" if latest is not None else "docs/index.html: no forecasts yet")
print(f"docs/backtest.html: {span}")
