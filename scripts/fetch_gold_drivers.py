"""
fetch_gold_drivers.py
DXY + US-Realzins 10Y (TIPS / DFII10) ab 2008.
"""
from __future__ import annotations
import os
import sys
from pathlib import Path
import pandas as pd
import requests
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()
FRED_API_KEY = os.getenv("FRED_API_KEY")

START_DATE = "2008-01-01"
OUT_RAW       = Path("data/raw/gold_drivers.parquet")
OUT_PROCESSED = Path("data/processed/gold_drivers.parquet")
OUT_RAW.parent.mkdir(parents=True, exist_ok=True)
OUT_PROCESSED.parent.mkdir(parents=True, exist_ok=True)
FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"


def fetch_dxy() -> pd.DataFrame:
    print("  DXY (DX-Y.NYB) ...", end="", flush=True)
    df = yf.download("DX-Y.NYB", start=START_DATE, progress=False, auto_adjust=True, threads=False)
    if df is None or df.empty:
        print(" FEHLER")
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    df.columns = [c.lower() for c in df.columns]
    df = df[["date", "close"]].dropna()
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df = df.rename(columns={"close": "dxy"})
    print(f" {len(df)} Tage")
    return df


def fetch_real_yield() -> pd.DataFrame:
    print("  Realzins 10Y (DFII10) ...", end="", flush=True)
    if not FRED_API_KEY:
        print(" FEHLER: kein API-Key")
        return pd.DataFrame()
    params = {"series_id": "DFII10", "api_key": FRED_API_KEY,
              "file_type": "json", "observation_start": START_DATE}
    r = requests.get(FRED_BASE, params=params, timeout=30)
    r.raise_for_status()
    obs = r.json().get("observations", [])
    if not obs:
        print(" FEHLER: keine Daten")
        return pd.DataFrame()
    df = pd.DataFrame(obs)[["date", "value"]]
    df["date"]  = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna().rename(columns={"value": "real_yield"})
    print(f" {len(df)} Tage")
    return df


def main() -> int:
    print("=" * 60)
    print("Gold-Treiber Pull (DXY + US-Realzins 10Y)")
    print(f"Startdatum: {START_DATE}")
    print("=" * 60)

    dxy = fetch_dxy()
    ry  = fetch_real_yield()

    if dxy.empty or ry.empty:
        print("FEHLER: Quelle leer.")
        return 1

    merged = pd.merge(dxy, ry, on="date", how="outer").sort_values("date")
    full_idx = pd.date_range(merged["date"].min(), merged["date"].max(), freq="D")
    merged = merged.set_index("date").reindex(full_idx).ffill().reset_index()
    merged = merged.rename(columns={"index": "date"})
    merged.to_parquet(OUT_RAW, index=False)
    print(f"\n-> {OUT_RAW} ({len(merged)} Zeilen)")

    proc = merged.copy()
    proc["dxy_chg_90d"]        = proc["dxy"]        - proc["dxy"].shift(90)
    proc["real_yield_chg_90d"] = proc["real_yield"] - proc["real_yield"].shift(90)
    proc.to_parquet(OUT_PROCESSED, index=False)
    print(f"-> {OUT_PROCESSED}")

    last = proc.iloc[-1]
    print(f"\nAktueller Stand ({last['date'].date()}):")
    print(f"   DXY:               {last['dxy']:.2f}    (90d-Trend: {last['dxy_chg_90d']:+.2f})")
    print(f"   Realzins 10Y (%):  {last['real_yield']:+.2f}  (90d-Trend: {last['real_yield_chg_90d']:+.2f}pp)")
    print("\nFertig.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
