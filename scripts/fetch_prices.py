"""
fetch_prices.py
---------------
Holt historische Tagespreise via yfinance fuer Martens 10 Maerkte.
Mindestens 15 Jahre Historie (Start: 2008-01-01).

Speichert pro Symbol eine Parquet-Datei in data/raw/prices/.
Aufruf:
    python scripts/fetch_prices.py
"""

from __future__ import annotations
import sys
import time
from pathlib import Path
import pandas as pd
import yfinance as yf

# -------------------------------------------------
# Mapping: dein Symbol -> Yahoo Ticker
# -------------------------------------------------
TICKERS = {
    "6E": "EURUSD=X",   # EUR/USD Spot
    "6B": "GBPUSD=X",   # GBP/USD Spot
    "6J": "JPY=X",      # USD/JPY Spot  (wird invertiert!)
    "6A": "AUDUSD=X",   # AUD/USD Spot
    "6C": "CAD=X",      # USD/CAD Spot  (wird invertiert!)
    "GC": "GC=F",       # Gold Continuous Future
    
}

# Bei diesen Symbolen ist der Yahoo-Kurs aus der Sicht des USD,
# wir brauchen aber die Future-Perspektive (Yen long / CAD long).
INVERT = {"6J", "6C"}

START_DATE = "2008-01-01"   # ergibt mind. 15 Jahre Historie
OUT_DIR    = Path("data/raw/prices")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def fetch_one(symbol: str, ticker: str) -> pd.DataFrame:
    """Holt Tagespreise via yfinance und gibt ein sauberes DataFrame zurueck."""
    df = yf.download(
        ticker,
        start=START_DATE,
        progress=False,
        auto_adjust=True,
        threads=False,
    )
    if df is None or df.empty:
        return pd.DataFrame()

    # yfinance liefert ggf. MultiIndex-Spalten -> flach machen
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()
    df.columns = [c.lower() for c in df.columns]

    # Wir brauchen nur Datum + Schlusskurs
    if "close" not in df.columns:
        return pd.DataFrame()
    df = df[["date", "close"]].dropna()
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df["symbol"] = symbol
    df["ticker"] = ticker

    # Bei USD/JPY und USD/CAD invertieren (1/Kurs), damit "steigend" = JPY/CAD stark
    if symbol in INVERT:
        df["close"] = 1.0 / df["close"]

    return df[["date", "symbol", "ticker", "close"]]


def main() -> int:
    print("=" * 60)
    print("Preis-Daten Pull (yfinance)")
    print(f"Startdatum: {START_DATE}")
    print("=" * 60)

    total = 0
    for sym, tkr in TICKERS.items():
        print(f"  {sym:3s} ({tkr:<10s}) ...", end="", flush=True)
        try:
            df = fetch_one(sym, tkr)
        except Exception as e:
            print(f" FEHLER: {e}")
            continue
        if df.empty:
            print(" KEINE DATEN")
            continue

        out_path = OUT_DIR / f"{sym}.parquet"
        df.to_parquet(out_path, index=False)
        first = df["date"].min().date()
        last  = df["date"].max().date()
        print(f" {len(df):5d} Zeilen  ({first} -> {last})")
        total += len(df)
        time.sleep(0.4)  # freundlich zum Yahoo-Server

    print(f"\nFertig. Insgesamt {total} Tageskurse gespeichert in {OUT_DIR}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
