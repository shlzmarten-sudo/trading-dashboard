"""
sma_crossover_es.py
--------------------
SMA-Crossover-Strategie fuer ES (E-mini S&P 500 Future) auf Basis
von Intraday-Daten der letzten Woche (yfinance).

Logik:
  - Fast-SMA kreuzt Slow-SMA von unten nach oben  -> LONG-Signal
  - Fast-SMA kreuzt Slow-SMA von oben nach unten   -> SHORT/FLAT-Signal

Speichert:
  - data/processed/es_sma_crossover.parquet  (Kurse, SMAs, Signale, Strategie-Equity)

Aufruf:
    python scripts/sma_crossover_es.py
"""

from __future__ import annotations
import sys
from pathlib import Path

import pandas as pd
import yfinance as yf

TICKER   = "ES=F"     # E-mini S&P 500 Future (Continuous) auf Yahoo Finance
PERIOD   = "7d"        # "die letzte Woche"
INTERVAL = "15m"       # Intraday-Aufloesung (yfinance erlaubt 15m fuer bis zu 60 Tage)

SMA_FAST = 9
SMA_SLOW = 21

OUT_PATH = Path("data/processed/es_sma_crossover.parquet")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)


def fetch_data() -> pd.DataFrame:
    """Holt Intraday-Kurse fuer ES=F via yfinance."""
    df = yf.download(
        TICKER,
        period=PERIOD,
        interval=INTERVAL,
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if df is None or df.empty:
        raise RuntimeError(f"Keine Daten fuer {TICKER} erhalten.")

    # yfinance liefert ggf. MultiIndex-Spalten -> flach machen
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()
    df.columns = [c.lower() for c in df.columns]
    time_col = "datetime" if "datetime" in df.columns else "date"
    df = df.rename(columns={time_col: "datetime"})
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df[["datetime", "open", "high", "low", "close", "volume"]].dropna(subset=["close"])


def compute_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Berechnet SMA-Fast/Slow, Crossover-Signale und eine einfache Strategie-Equity-Kurve."""
    df = df.copy()
    df["sma_fast"] = df["close"].rolling(SMA_FAST).mean()
    df["sma_slow"] = df["close"].rolling(SMA_SLOW).mean()

    # Position: 1 = long (fast > slow), 0 = flat, -1 = short
    df["position"] = 0
    df.loc[df["sma_fast"] > df["sma_slow"], "position"] = 1
    df.loc[df["sma_fast"] < df["sma_slow"], "position"] = -1

    # Signal nur an dem Bar, an dem sich die Position aendert (echter Crossover)
    df["signal"] = df["position"].diff()
    df["event"] = ""
    df.loc[df["signal"] > 0, "event"] = "GOLDEN_CROSS (LONG)"
    df.loc[df["signal"] < 0, "event"] = "DEATH_CROSS (SHORT)"

    # Einfache Strategie-Rendite: gestrige Position * heutige Kursrendite
    df["ret"] = df["close"].pct_change()
    df["strategy_ret"] = df["position"].shift(1) * df["ret"]
    df["equity_buyhold"] = (1 + df["ret"].fillna(0)).cumprod()
    df["equity_strategy"] = (1 + df["strategy_ret"].fillna(0)).cumprod()

    return df


def main() -> int:
    print("=" * 60)
    print(f"SMA-Crossover-Strategie: {TICKER} ({PERIOD}, {INTERVAL})")
    print(f"Fast SMA: {SMA_FAST}  |  Slow SMA: {SMA_SLOW}")
    print("=" * 60)

    try:
        raw = fetch_data()
    except Exception as e:
        print(f"FEHLER beim Laden der Daten: {e}")
        return 1

    print(f"Geladen: {len(raw)} Bars ({raw['datetime'].min()} -> {raw['datetime'].max()})")

    df = compute_signals(raw)
    df.to_parquet(OUT_PATH, index=False)
    print(f"-> Ergebnisse gespeichert unter {OUT_PATH}")

    events = df[df["event"] != ""][["datetime", "close", "sma_fast", "sma_slow", "event"]]
    print(f"\nCrossover-Events der letzten Woche ({len(events)}):")
    if events.empty:
        print("  Keine Crossover in diesem Zeitraum.")
    else:
        for _, row in events.iterrows():
            print(
                f"  {row['datetime']}  Close={row['close']:.2f}  "
                f"SMA{SMA_FAST}={row['sma_fast']:.2f}  SMA{SMA_SLOW}={row['sma_slow']:.2f}  "
                f"-> {row['event']}"
            )

    last = df.iloc[-1]
    aktuelle_position = {1: "LONG", -1: "SHORT", 0: "FLAT"}[last["position"]]
    print(f"\nAktuelle Position: {aktuelle_position}  (Close={last['close']:.2f})")
    print(f"Performance Buy&Hold : {(last['equity_buyhold'] - 1) * 100:+.2f}%")
    print(f"Performance Strategie: {(last['equity_strategy'] - 1) * 100:+.2f}%")

    return 0


if __name__ == "__main__":
    sys.exit(main())
