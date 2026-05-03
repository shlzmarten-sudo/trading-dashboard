"""
fetch_cot.py
------------
Holt die wöchentlichen COT-Reports der CFTC fuer Martens 10 Maerkte.
Datenquelle: CFTC Public Reporting (Socrata API, kostenlos, ohne Key).

- Legacy-Report  (Futures Only) fuer alle 10 Maerkte
- Disaggregated-Report fuer GC, SI, CL (zusaetzlich)

Speichert die Rohdaten als Parquet unter data/raw/.
Aufruf:
    python scripts/fetch_cot.py
"""

from __future__ import annotations
import sys
import time
from pathlib import Path
import pandas as pd
import requests

# -----------------------------------------------------------
# 1) Konfiguration: deine 10 Maerkte und ihre CFTC-Codes
# -----------------------------------------------------------
# "cftc_contract_market_code" ist der eindeutige Schluessel pro Markt.
MARKETS = {
    "6E": {"name": "Euro FX",          "code": "099741"},
    "6B": {"name": "British Pound",    "code": "096742"},
    "6J": {"name": "Japanese Yen",     "code": "097741"},
    "6A": {"name": "Australian Dollar","code": "232741"},
    "6C": {"name": "Canadian Dollar",  "code": "090741"},
    "GC": {"name": "Gold",             "code": "088691"},
    
}

# Maerkte, fuer die wir zusaetzlich den Disaggregated-Report wollen
DISAGG_SYMBOLS = ["GC"]

# Startdatum der Historie
START_DATE = "2010-01-01"

# CFTC SODA Endpoints (Open Data, JSON)
LEGACY_URL  = "https://publicreporting.cftc.gov/resource/6dca-aqww.json"
DISAGG_URL  = "https://publicreporting.cftc.gov/resource/72hh-3qpy.json"

# Wieviel pro Anfrage holen? Socrata erlaubt bis 50.000.
PAGE_SIZE = 50000

# Output-Ordner
OUT_DIR = Path("data/raw")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def fetch_one(url: str, market_code: str, start_date: str) -> pd.DataFrame:
    """
    Holt alle Zeilen fuer einen Markt-Code von der angegebenen URL.
    Filtert serverseitig nach Markt-Code und Startdatum.
    """
    params = {
        "$where": (
            f"cftc_contract_market_code='{market_code}' "
            f"AND report_date_as_yyyy_mm_dd >= '{start_date}T00:00:00.000'"
        ),
        "$order": "report_date_as_yyyy_mm_dd ASC",
        "$limit": PAGE_SIZE,
    }
    headers = {"Accept": "application/json"}
    r = requests.get(url, params=params, headers=headers, timeout=30)
    r.raise_for_status()
    data = r.json()
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    # Datum sauber konvertieren
    df["report_date"] = pd.to_datetime(df["report_date_as_yyyy_mm_dd"]).dt.tz_localize(None)
    return df


def fetch_legacy_all() -> pd.DataFrame:
    """Holt den Legacy-Report fuer alle 10 Maerkte und konkateniert sie."""
    frames = []
    for sym, info in MARKETS.items():
        print(f"  [Legacy] {sym:3s} {info['name']:<20s} (code={info['code']}) ...", end="", flush=True)
        df = fetch_one(LEGACY_URL, info["code"], START_DATE)
        df["symbol"] = sym
        df["market_name"] = info["name"]
        frames.append(df)
        print(f" {len(df):4d} Zeilen")
        time.sleep(0.3)  # freundlich zur API
    return pd.concat(frames, ignore_index=True)


def fetch_disagg_subset() -> pd.DataFrame:
    """Holt den Disaggregated-Report nur fuer GC, SI, CL."""
    frames = []
    for sym in DISAGG_SYMBOLS:
        info = MARKETS[sym]
        print(f"  [Disagg] {sym:3s} {info['name']:<20s} (code={info['code']}) ...", end="", flush=True)
        df = fetch_one(DISAGG_URL, info["code"], START_DATE)
        df["symbol"] = sym
        df["market_name"] = info["name"]
        frames.append(df)
        print(f" {len(df):4d} Zeilen")
        time.sleep(0.3)
    return pd.concat(frames, ignore_index=True)


def main() -> int:
    print("=" * 60)
    print("CFTC COT-Daten Pull")
    print(f"Startdatum: {START_DATE}")
    print("=" * 60)

    # --- Legacy ---
    print("\n[1/2] Lade Legacy-Reports ...")
    df_legacy = fetch_legacy_all()
    legacy_path = OUT_DIR / "cot_legacy.parquet"
    df_legacy.to_parquet(legacy_path, index=False)
    print(f"-> Gespeichert: {legacy_path}  ({len(df_legacy)} Zeilen)")

    # --- Disaggregated ---
    print("\n[2/2] Lade Disaggregated-Reports (GC, SI, CL) ...")
    df_disagg = fetch_disagg_subset()
    disagg_path = OUT_DIR / "cot_disaggregated.parquet"
    df_disagg.to_parquet(disagg_path, index=False)
    print(f"-> Gespeichert: {disagg_path}  ({len(df_disagg)} Zeilen)")

    # --- Mini-Sanity-Check ---
    print("\n[OK] Aktuellster Legacy-Report-Datum pro Markt:")
    last = (
        df_legacy.groupby("symbol")["report_date"]
        .max()
        .sort_values(ascending=False)
    )
    for sym, dt in last.items():
        print(f"   {sym}: {dt.date()}")

    print("\nFertig.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
