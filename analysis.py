"""
Mini-project: Aggregator (FoSE) portfolio of 3 RES parks in the Greek Day-Ahead Market
Input : greece_res_merged.csv  (hourly: pv/wind forecast + actual MW, DAM price EUR/MWh)
Usage : python analysis.py [path_to_csv]
Output: ./outputs/summary_by_park.csv, monthly_capture.csv, sensitivity_noise.csv, charts/*.png
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# 0. Settings and assumptions (document all of these in the README)
# ----------------------------------------------------------------------------
CSV = sys.argv[1] if len(sys.argv) > 1 else "greece_res_merged.csv"
OUT = Path("outputs")
(OUT / "charts").mkdir(parents=True, exist_ok=True)

# Simulated parks: technology, installed MW, relative efficiency vs the national average
PARKS = {
    "PV-1":   dict(tech="pv",   mw=5.0, efficiency=1.00),
    "PV-2":   dict(tech="pv",   mw=3.0, efficiency=0.92),
    "WIND-1": dict(tech="wind", mw=8.0, efficiency=0.97),
}

# Park-specific noise on the ACTUAL output (local weather vs national average):
# multiplicative AR(1) noise, stationary std = NOISE_SIGMA, persistence = NOISE_PHI
NOISE_SIGMA = 0.05
NOISE_PHI = 0.80
SEED = 42

# Imbalance spread: surplus is paid P - x|P|, deficit is charged P + x|P|.
X = 0.10

# Rescale the national forecast so it covers the same fleet as the "actual" series.
CALIBRATE = {"pv": True, "wind": False}

# ----------------------------------------------------------------------------
# 1. Load and clean
# ----------------------------------------------------------------------------
df = pd.read_csv(CSV, index_col=0)
df.index = pd.to_datetime(df.index, utc=True).tz_convert("Europe/Athens")
df = df[~df.index.duplicated()].sort_index()
df = df.rename(columns={"da_price_eur_mwh": "price"})

if df.isna().any().any():
    print("WARNING: missing values found, filling by interpolation")
    print(df.isna().sum())
    df = df.interpolate(limit=3)

# PV cannot produce when the forecast is zero (small night values in ENTSO-E actuals)
df.loc[df["pv_forecast_mw"] == 0, "pv_actual_mw"] = 0.0

P = df["price"]
BASE = P.mean()  # baseload average price of the period. Used to compute capture rate 

# ----------------------------------------------------------------------------
# 2. Calibrate the national forecast
# ----------------------------------------------------------------------------
k = {}
for tech in ("pv", "wind"):
    k[tech] = df[f"{tech}_actual_mw"].sum() / df[f"{tech}_forecast_mw"].sum()
    factor = k[tech] if CALIBRATE[tech] else 1.0
    df[f"{tech}_forecast_cal"] = df[f"{tech}_forecast_mw"] * factor
print(f"Actual/forecast energy ratio: PV={k['pv']:.3f}, wind={k['wind']:.3f}")
print(f"Calibration applied: {CALIBRATE}")


# ----------------------------------------------------------------------------
# 3. Build the parks: normalise -> scale to our own parks MW -> apply efficiency -> noise
# ----------------------------------------------------------------------------
def ar1_noise(rng, n, sigma, phi):
    """Stationary AR(1) noise with standard deviation sigma."""
    if sigma == 0:
        return np.zeros(n)
    z = rng.standard_normal(n)
    e = np.zeros(n)
    e[0] = sigma * z[0]
    for t in range(1, n):
        e[t] = phi * e[t - 1] + sigma * np.sqrt(1 - phi**2) * z[t]
    return np.clip(e, -0.5, 0.5)


def build_parks(sigma=NOISE_SIGMA, seed=SEED):
    rng = np.random.default_rng(seed)
    pk = pd.DataFrame(index=df.index)
    for name, p in PARKS.items():
        tech, mw = p["tech"], p["mw"]
        ref = df[f"{tech}_actual_mw"].max()                  # national peak = capacity proxy
        cf_F = (df[f"{tech}_forecast_cal"] / ref).clip(0, 1)  # capacity factor, forecast
        cf_A = (df[f"{tech}_actual_mw"] / ref).clip(0, 1)     # capacity factor, actual
        eps = ar1_noise(rng, len(df), sigma, NOISE_PHI)
        pk[f"{name}_F"] = (cf_F * p["efficiency"] * mw).clip(0, mw)
        pk[f"{name}_A"] = (cf_A * p["efficiency"] * mw * (1 + eps)).clip(0, mw)
    return pk


# ----------------------------------------------------------------------------
# 4. Settlement: DAM revenue and imbalance result
# ----------------------------------------------------------------------------
def settle(F, A, price=P, x=X):
    """F, A in MWh (hourly, so MW == MWh). Returns dam revenue, imbalance result, deviation."""
    D = A - F
    dam = F * price
    imb_price = np.where(D >= 0, price - x * price.abs(), price + x * price.abs())
    imb = D * imb_price
    return dam, imb, D


def metrics(name, F, A, cap):
    dam, imb, D = settle(F, A)
    net = dam + imb
    perfect = A * P
    return {
        "park": name,
        "capacity_MW": cap,
        "forecast_MWh": F.sum(),
        "actual_MWh": A.sum(),
        "DAM_revenue_EUR": dam.sum(),
        "imbalance_result_EUR": imb.sum(),
        "net_revenue_EUR": net.sum(),
        "imbalance_cost_EUR": (perfect - net).sum(),  # = x * sum(|D| * |P|)
        "realised_price_EUR_MWh": net.sum() / A.sum(),
        "capture_rate": (A * P).sum() / A.sum() / BASE,
        "MAE_MW": D.abs().mean(),
        "NMAE_pct": D.abs().mean() / cap * 100,
        "bias_MW": D.mean(),
    }


def evaluate(pk):
    rows = [metrics(n, pk[f"{n}_F"], pk[f"{n}_A"], p["mw"]) for n, p in PARKS.items()]
    cap_tot = sum(p["mw"] for p in PARKS.values())
    F_tot = sum(pk[f"{n}_F"] for n in PARKS)
    A_tot = sum(pk[f"{n}_A"] for n in PARKS)
    port = metrics("PORTFOLIO", F_tot, A_tot, cap_tot)   # imbalance on the NET position
    rows.append(port)
    summary = pd.DataFrame(rows).set_index("park")
    standalone = summary.loc[list(PARKS), "imbalance_cost_EUR"].sum()
    pooled = port["imbalance_cost_EUR"]
    weighted_nmae = summary.loc[list(PARKS), "MAE_MW"].sum() / cap_tot * 100
    return summary, dict(
        standalone_cost=standalone,
        pooled_cost=pooled,
        saving=standalone - pooled,
        saving_pct=(standalone - pooled) / standalone,
        weighted_nmae=weighted_nmae,
        portfolio_nmae=port["NMAE_pct"],
        nmae_reduction=1 - port["NMAE_pct"] / weighted_nmae,
    )


pk = build_parks()
summary, agg = evaluate(pk)
port = summary.loc["PORTFOLIO"]
standalone_cost, pooled_cost = agg["standalone_cost"], agg["pooled_cost"]
summary.round(2).to_csv(OUT / "summary_by_park.csv")

# ----------------------------------------------------------------------------
# 4b. Export hourly tables for the SQL step (local wall-clock time as text) OPTIONAL
# ----------------------------------------------------------------------------
ts_local = df.index.strftime("%Y-%m-%d %H:%M:%S")
base_cols = pd.DataFrame({
    "ts_local": ts_local,
    "date": df.index.strftime("%Y-%m-%d"),
    "month": df.index.month,
    "hour": df.index.hour,
})
market = base_cols.copy()
market["price"] = P.values
for c in ["pv_forecast_mw", "pv_actual_mw", "wind_forecast_mw", "wind_actual_mw"]:
    market[c] = df[c].values
market.to_csv(OUT / "market.csv", index=False)

long_rows = []
for n, p in PARKS.items():
    F, A = pk[f"{n}_F"], pk[f"{n}_A"]
    dam, imb, D = settle(F, A)
    part = base_cols.copy()
    part["park"] = n
    part["tech"] = p["tech"]
    part["price"] = P.values
    part["forecast_mwh"] = F.values
    part["actual_mwh"] = A.values
    part["dev_mwh"] = D.values
    part["dam_revenue"] = dam.values
    part["imb_result"] = imb.values
    part["net_revenue"] = (dam + imb).values
    part["imb_cost"] = (A * P - (dam + imb)).values
    long_rows.append(part)
pd.concat(long_rows).to_csv(OUT / "park_hourly.csv", index=False)

# ----------------------------------------------------------------------------
# 5. Capture rate per technology and month (national actual profile)
# ----------------------------------------------------------------------------
months = df.index.month
cap_rows = []
for m in sorted(set(months)):
    sub = df[months == m]
    base_m = sub["price"].mean()
    for tech in ("pv", "wind"):
        a = sub[f"{tech}_actual_mw"]
        cap_rows.append({
            "month": m, "tech": tech,
            "baseload_EUR_MWh": base_m,
            "weighted_price_EUR_MWh": (a * sub["price"]).sum() / a.sum(),
            "capture_rate": (a * sub["price"]).sum() / a.sum() / base_m,
        })
monthly = pd.DataFrame(cap_rows)
monthly.round(3).to_csv(OUT / "monthly_capture.csv", index=False)

# ----------------------------------------------------------------------------
# 6. Sensitivity: how much of the pooling gain depends on the noise assumption?
# ----------------------------------------------------------------------------
sens = []
for sigma in (0.0, 0.05, 0.10):
    res = [evaluate(build_parks(sigma, s))[1] for s in range(100)]
    sp = np.array([r["saving_pct"] for r in res])
    nr = np.array([r["nmae_reduction"] for r in res])
    sens.append({
        "noise_sigma": sigma,
        "pooling_saving_mean_pct": sp.mean() * 100,
        "pooling_saving_p5_pct": np.percentile(sp, 5) * 100,
        "pooling_saving_p95_pct": np.percentile(sp, 95) * 100,
        "nmae_reduction_mean_pct": nr.mean() * 100,
    })
sens = pd.DataFrame(sens)
sens.round(2).to_csv(OUT / "sensitivity_noise.csv", index=False)

# ----------------------------------------------------------------------------
# 7. Print the key results
# ----------------------------------------------------------------------------
pd.set_option("display.width", 200, "display.max_columns", 30)
print("\n=== Summary by park ===")
print(summary.round(2).T)
print("\n=== Capture rate by month ===")
print(monthly.round(2))
print("\n=== Value of aggregation (seed 42) ===")
print(f"Stand-alone imbalance cost : {standalone_cost:,.0f} EUR")
print(f"Pooled (portfolio) cost    : {pooled_cost:,.0f} EUR")
print(f"Saving from pooling        : {agg['saving']:,.0f} EUR ({agg['saving_pct']:.1%})")
print(f"NMAE weighted parks / portfolio: {agg['weighted_nmae']:.2f}% / {agg['portfolio_nmae']:.2f}%"
      f"  (reduction {agg['nmae_reduction']:.1%})")
print("\n=== Sensitivity to the noise assumption (100 seeds each) ===")
print(sens.round(2))
print(f"\nBaseload {BASE:.1f} EUR/MWh | hours with price <= 0: {(P <= 0).sum()} "
      f"({(P <= 0).mean():.1%})")

# ----------------------------------------------------------------------------
# 8. Charts
# ----------------------------------------------------------------------------
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False})
hr = df.index.hour

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(range(24), df.groupby(hr)["price"].mean(), marker="o", color="tab:red", label="DAM price")
ax.set_xlabel("Hour of day (Athens time)")
ax.set_ylabel("Average DAM price (EUR/MWh)")
ax2 = ax.twinx()
ax2.spines["right"].set_visible(True)
ax2.fill_between(range(24), df.groupby(hr)["pv_actual_mw"].mean(), alpha=0.25, color="gold", label="PV actual")
ax2.plot(range(24), df.groupby(hr)["wind_actual_mw"].mean(), color="tab:blue", label="Wind actual")
ax2.set_ylabel("Average national generation (MW)")
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc="upper left")
ax.set_title("Price vs solar and wind: average daily profile")
fig.tight_layout()
fig.savefig(OUT / "charts" / "1_price_vs_generation_profile.png")
plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(range(24), df.groupby(hr)["pv_forecast_mw"].mean(), label="Forecast (raw)", ls="--")
ax.plot(range(24), df.groupby(hr)["pv_forecast_cal"].mean(), label=f"Forecast (x{k['pv']:.2f})")
ax.plot(range(24), df.groupby(hr)["pv_actual_mw"].mean(), label="Actual", lw=2)
ax.set_xlabel("Hour of day (Athens time)")
ax.set_ylabel("National PV (MW)")
ax.set_title("PV data check: forecast vs actual")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "charts" / "2_pv_forecast_vs_actual.png")
plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 4.5))
piv = monthly.pivot(index="month", columns="tech", values="capture_rate")
piv.index = [pd.Timestamp(2026, m, 1).strftime("%b") for m in piv.index]
piv.plot.bar(ax=ax, rot=0, color={"pv": "gold", "wind": "tab:blue"})
ax.axhline(1.0, color="grey", ls=":")
ax.set_ylabel("Capture rate (weighted price / baseload)")
ax.set_xlabel("")
ax.set_title("Capture rate: PV vs wind")
fig.tight_layout()
fig.savefig(OUT / "charts" / "3_capture_rate.png")
plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 4.5))
s = summary["NMAE_pct"]
ax.bar(s.index, s.values, color=["gold", "gold", "tab:blue", "tab:green"])
for i, v in enumerate(s.values):
    ax.text(i, v + 0.1, f"{v:.1f}%", ha="center")
ax.set_ylabel("NMAE (% of installed capacity)")
ax.set_title("Forecast error: single parks vs portfolio")
fig.tight_layout()
fig.savefig(OUT / "charts" / "4_nmae_portfolio_effect.png")
plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 4.5))
ax.bar(["Stand-alone\n(sum of 3 parks)", "Pooled\n(portfolio)"],
       [standalone_cost, pooled_cost], color=["tab:grey", "tab:green"])
for i, v in enumerate([standalone_cost, pooled_cost]):
    ax.text(i, v, f"{v:,.0f} EUR", ha="center", va="bottom")
ax.set_ylabel("Imbalance cost (EUR)")
ax.set_title(f"Value of aggregation (spread x = {X:.0%})")
fig.tight_layout()
fig.savefig(OUT / "charts" / "5_imbalance_cost_pooled.png")
plt.close(fig)

# 6: imbalance cost by hour of day (all parks, stand-alone)
cost_hour = pd.concat(long_rows).groupby("hour")["imb_cost"].sum()
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.bar(cost_hour.index, cost_hour.values, color="tab:red", alpha=0.8)
ax.set_xlabel("Hour of day (Athens time)")
ax.set_ylabel("Imbalance cost (EUR)")
ax.set_title("When do imbalances cost the most?")
fig.tight_layout()
fig.savefig(OUT / "charts" / "6_imbalance_cost_by_hour.png")
plt.close(fig)

# ----------------------------------------------------------------------------
# 7 and 8: forecast vs actual per park and totals, together with the DAM price
# ----------------------------------------------------------------------------
F_pv = pk["PV-1_F"] + pk["PV-2_F"]
A_pv = pk["PV-1_A"] + pk["PV-2_A"]
F_all = sum(pk[f"{n}_F"] for n in PARKS)
A_all = sum(pk[f"{n}_A"] for n in PARKS)
SERIES = [
    ("PV-1 (5 MW)", pk["PV-1_F"], pk["PV-1_A"]),
    ("PV-2 (3 MW)", pk["PV-2_F"], pk["PV-2_A"]),
    ("WIND-1 (8 MW) = total wind", pk["WIND-1_F"], pk["WIND-1_A"]),
    ("Total PV (PV-1 + PV-2)", F_pv, A_pv),
    ("Portfolio total (PV + wind)", F_all, A_all),
]
C_F, C_A, C_P = "tab:blue", "black", "tab:red"


def price_axis(ax, x, y, show_label):
    ax2 = ax.twinx()
    ax2.spines["right"].set_visible(True)
    ax2.plot(x, y, color=C_P, lw=1, alpha=0.7)
    ax2.set_ylim(min(0, np.nanmin(y) - 5), np.nanmax(y) * 1.1)
    ax2.tick_params(axis="y", colors=C_P)
    if show_label:
        ax2.set_ylabel("DAM price (EUR/MWh)", color=C_P)
    return ax2


# 7: average daily profile, 3x2 grid (5 series + price panel)
fig, axes = plt.subplots(3, 2, figsize=(11, 10), sharex=True)
hours = np.arange(24)
price_prof = P.groupby(P.index.hour).mean().values
for idx, (name, F, A) in enumerate(SERIES):
    ax = axes.flat[idx]
    f = F.groupby(F.index.hour).mean().values
    a_ = A.groupby(A.index.hour).mean().values
    ax.plot(hours, f, color=C_F, ls="--", label="Forecast (day-ahead)")
    ax.plot(hours, a_, color=C_A, lw=2, label="Actual")
    ax.fill_between(hours, f, a_, color="grey", alpha=0.2)
    ax.set_title(name, fontsize=10)
    ax.set_ylabel("Average MWh per hour")
    price_axis(ax, hours, price_prof, show_label=(idx % 2 == 1))
    if idx == 0:
        ax.legend(loc="upper left", fontsize=8)
ax = axes.flat[5]
ax.plot(hours, price_prof, color=C_P, marker="o")
ax.axhline(BASE, color="grey", ls=":")
ax.text(0.3, BASE + 3, f"baseload {BASE:.1f}", fontsize=8, color="grey")
ax.set_title("DAM price, average by hour", fontsize=10)
ax.set_ylabel("EUR/MWh")
for ax in axes[-1]:
    ax.set_xlabel("Hour of day (Athens time)")
fig.suptitle("Day-ahead forecast vs actual production, with the DAM price (Apr-May 2026, average day)",
             fontsize=11)
fig.tight_layout()
fig.savefig(OUT / "charts" / "7_forecast_vs_actual_parks.png")
plt.close(fig)

# 8: the 7-day window with the highest imbalance cost, hour by hour
cost_t = pd.Series(np.sum([r["imb_cost"].values for r in long_rows], axis=0), index=df.index)
end = cost_t.rolling(168).sum().idxmax()
start = end - pd.Timedelta(hours=167)
sl = slice(start, end)
panels = [SERIES[3], SERIES[2], SERIES[4]]
fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
xs = F_all[sl].index.tz_localize(None)       # local Athens wall-clock time, no UTC shift
for ax, (name, F, A) in zip(axes, panels):
    ax.plot(xs, F[sl].values, color=C_F, ls="--", label="Forecast (day-ahead)")
    ax.plot(xs, A[sl].values, color=C_A, lw=1.6, label="Actual")
    ax.fill_between(xs, F[sl].values, A[sl].values, color="grey", alpha=0.25)
    ax.set_title(name, fontsize=10)
    ax.set_ylabel("MWh per hour")
    ax.set_xlim(xs[0], xs[-1])
    price_axis(ax, xs, P[sl].values, show_label=True)
axes[0].legend(loc="upper left", fontsize=8)
axes[-1].xaxis.set_major_locator(matplotlib.dates.DayLocator())
axes[-1].xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%d %b"))
axes[-1].set_xlabel("Date (Athens time, ticks at midnight)")
fig.suptitle(f"Hourly forecast vs actual and DAM price: week with the highest imbalance cost "
             f"({start:%d %b %H:%M} - {end:%d %b %H:%M}, 2026)", fontsize=11)
fig.tight_layout()
fig.savefig(OUT / "charts" / "8_week_forecast_vs_actual.png")
plt.close(fig)
print(f"Chart 8 window: {start} -> {end}")

print(f"\nDone. Results in {OUT.resolve()}")