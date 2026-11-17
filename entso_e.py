import os
import pandas as pd
from entsoe import EntsoePandasClient

API_KEY = os.environ["ENTSOE_API_KEY"]  # use your own key from entso-e
client = EntsoePandasClient(api_key=API_KEY)

TZ = "Europe/Athens"
COUNTRY = "GR"
START = pd.Timestamp("2026-04-01", tz=TZ)
END = pd.Timestamp("2026-06-01", tz=TZ)      # end is exclusive


def month_chunks(start, end):
    edges = pd.date_range(start, end, freq="MS", tz=TZ)
    edges = edges.union([start, end])
    return list(zip(edges[:-1], edges[1:]))


def fetch_chunked(fn, *args, **kwargs):
    parts = []
    for s, e in month_chunks(START, END):
        try:
            out = fn(*args, start=s, end=e, **kwargs)
            if isinstance(out, pd.Series):
                out = out.to_frame(out.name or "value")
            parts.append(out)
        except Exception as ex:   # NoMatchingDataError etc.
            print(f"{fn.__name__} {s.date()}: {type(ex).__name__}")
    out = pd.concat(parts)
    return out[~out.index.duplicated(keep="first")].sort_index()


# 1. Day-ahead wind & solar forecast
df_fc = fetch_chunked(client.query_wind_and_solar_forecast, COUNTRY, psr_type=None)

# 2. Actual generation per type
df_act = fetch_chunked(client.query_generation, COUNTRY, psr_type=None)
if isinstance(df_act.columns, pd.MultiIndex):
    df_act = df_act.xs("Actual Aggregated", axis=1, level=1)

# 3. Day-ahead prices (EUR/MWh)
df_price = fetch_chunked(client.query_day_ahead_prices, COUNTRY)

def to_hourly(obj):
    obj = obj.copy()
    obj.index = obj.index.tz_convert(TZ)
    return obj.resample("1h").mean()


merged = pd.DataFrame({
    "pv_forecast_mw":   to_hourly(df_fc["Solar"]),
    "pv_actual_mw":     to_hourly(df_act["Solar"]),
    "wind_forecast_mw": to_hourly(df_fc["Wind Onshore"]),
    "wind_actual_mw":   to_hourly(df_act["Wind Onshore"]),
    "da_price_eur_mwh": to_hourly(df_price.iloc[:, 0]),
})
merged.index.name = "timestamp_athens"

print(merged.isna().sum())
dropped = merged.dropna()
print(f"Dropped {len(merged) - len(dropped)} rows with missing values")
merged.to_csv("greece_res_merged.csv")
print(merged.head())