"""
fetch_rates.py
--------------
Holt Leitzinsen / Geldmarktproxies via FRED-API.
Mind. 5 Jahre Historie (Start 2018-01-01).

Robust: Pro Notenbank wird eine Liste von Fallback-Serien probiert.
Sobald eine Serie funktioniert (=Daten liefert), wird sie genommen.

Speichert nach data/raw/rates.parquet
"""

from __future__ import annotations
import os, sys, time
from pathlib import Path
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()
FRED_API_KEY = os.getenv("FRED_API_KEY")

# Pro Notenbank: Liste von Fallback-Serien (in Reihenfolge probiert)
SERIES = {
    "FED": {
        "candidates": ["DFF", "FEDFUNDS", "DFEDTARU"],
        "name": "US Federal Reserve",  "currency": "USD",
    },
    "ECB": {
        "candidates": ["ECBDFR", "ECBESTRVOLWGTTRMDMNRT"],
        "name": "Europ. Zentralbank",  "currency": "EUR",
    },
    "BOE": {
        "candidates": ["IUDSOIA", "BOERUKM"],
        "name": "Bank of England",     "currency": "GBP",
    },
    "BOJ": {
        "candidates": ["IRSTCB01JPM156N", "INTDSRJPM193N", "IR3TIB01JPM156N"],
        "name": "Bank of Japan",       "currency": "JPY",
    },
    "RBA": {
        "candidates": ["IR3TIB01AUM156N", "IRSTCB01AUM156N", "INTDSRAUM193N"],
        "name": "Reserve Bank Australia","currency": "AUD",
    },
    "BOC": {
        "candidates": ["IRSTCI01CAM156N", "IRSTCB01CAM156N", "INTDSRCAM193N", "IR3TIB01CAM156N"],
        "name": "Bank of Canada",      "currency": "CAD",
    },
}

START_DATE = "2018-01-01"
OUT_PATH   = Path("data/raw/rates.parquet")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"


def fetch_one(series_id: str) -> pd.DataFrame:
    params = {
        "series_id":         series_id,
        "api_key":           FRED_API_KEY,
        "file_type":         "json",
        "observation_start": START_DATE,
    }
    r = requests.get(FRED_BASE, params=params, timeout=30)
    if r.status_code != 200:
        return pd.DataFrame()
    obs = r.json().get("observations", [])
    if not obs:
        return pd.DataFrame()
    df = pd.DataFrame(obs)[["date", "value"]]
    df["date"]  = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.dropna().sort_values("date").reset_index(drop=True)


def fetch_with_fallback(cb: str, cands: list[str]) -> tuple[pd.DataFrame, str]:
    """Probiere die Serien der Reihe nach durch."""
    for sid in cands:
        df = fetch_one(sid)
        if not df.empty:
            return df, sid
    return pd.DataFrame(), ""


def main() -> int:
    if not FRED_API_KEY:
        print("FEHLER: FRED_API_KEY nicht gefunden.")
        return 1

    print("=" * 60)
    print("Zins-Daten Pull (FRED, mit Fallback)")
    print(f"Startdatum: {START_DATE}")
    print("=" * 60)

    frames, fails = [], []
    for cb, info in SERIES.items():
        print(f"  {cb:3s} ...", end="", flush=True)
        df, used_id = fetch_with_fallback(cb, info["candidates"])
        if df.empty:
            print(f" KEINE DATEN (alle Fallbacks fehlgeschlagen)")
            fails.append(cb)
            continue
        df["central_bank"] = cb
        df["bank_name"]    = info["name"]
        df["currency"]     = info["currency"]
        df["series_id"]    = used_id
        frames.append(df[["date","central_bank","bank_name","currency","series_id","value"]])
        first = df["date"].min().date()
        last  = df["date"].max().date()
        print(f" Serie={used_id:<22s} {len(df):5d} Z.  ({first} -> {last})  letzter: {df['value'].iloc[-1]:.2f}%")
        time.sleep(0.3)

    if not frames:
        print("\nFEHLER: Keine Notenbank konnte geladen werden.")
        return 1

    out = pd.concat(frames, ignore_index=True)
    out.to_parquet(OUT_PATH, index=False)
    print(f"\nGespeichert: {OUT_PATH}  ({len(out)} Z., {out['central_bank'].nunique()} Notenbanken)")
    if fails:
        print(f"WARNUNG: keine Daten fuer: {', '.join(fails)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
