"""
compute_rate_diffs.py
---------------------
Liest data/raw/rates.parquet und berechnet die 5 Zinsdifferenzen,
die zu Martens FX-Futures passen.

Schreibt:
  - data/processed/rates_wide.parquet   (Datum x Notenbank, ffill)
  - data/processed/rate_diffs.parquet   (Datum x FX-Paar Diff)
"""

from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd

IN_PATH         = Path("data/raw/rates.parquet")
OUT_WIDE        = Path("data/processed/rates_wide.parquet")
OUT_DIFFS       = Path("data/processed/rate_diffs.parquet")
OUT_WIDE.parent.mkdir(parents=True, exist_ok=True)

# FX-Paar -> (Basis-Notenbank, Quote-Notenbank, zugehoeriger Future)
PAIRS = {
    "EURUSD": {"base": "ECB", "quote": "FED", "future": "6E"},
    "GBPUSD": {"base": "BOE", "quote": "FED", "future": "6B"},
    "USDJPY": {"base": "FED", "quote": "BOJ", "future": "6J"},
    "AUDUSD": {"base": "RBA", "quote": "FED", "future": "6A"},
    "USDCAD": {"base": "FED", "quote": "BOC", "future": "6C"},
}


def main() -> int:
    if not IN_PATH.exists():
        print(f"FEHLER: {IN_PATH} fehlt. Erst scripts/fetch_rates.py ausfuehren.")
        return 1

    df = pd.read_parquet(IN_PATH)
    df["date"] = pd.to_datetime(df["date"])
    print(f"Eingelesen: {len(df)} Zeilen, {df['central_bank'].nunique()} Notenbanken")

    # 1) Wide-Tabelle: Zeile = Datum, Spalte = Notenbank, Wert = Zins
    wide = (
        df.pivot_table(index="date", columns="central_bank", values="value", aggfunc="last")
          .sort_index()
    )

    # Tagesachse (alle Werktage zwischen Min und Max), dann ffill
    full_idx = pd.date_range(wide.index.min(), wide.index.max(), freq="D")
    wide = wide.reindex(full_idx).ffill()
    wide.index.name = "date"

    wide.to_parquet(OUT_WIDE)
    print(f"-> {OUT_WIDE} ({len(wide)} Tage, {wide.shape[1]} Banken)")

    # 2) Zinsdifferenzen pro FX-Paar
    diffs = pd.DataFrame(index=wide.index)
    for pair, cfg in PAIRS.items():
        b, q = cfg["base"], cfg["quote"]
        if b not in wide.columns or q not in wide.columns:
            print(f"  WARNUNG: {pair} nicht moeglich, {b} oder {q} fehlt.")
            continue
        diffs[pair] = wide[b] - wide[q]

    diffs = diffs.dropna(how="all").reset_index()
    diffs.to_parquet(OUT_DIFFS, index=False)
    print(f"-> {OUT_DIFFS} ({len(diffs)} Tage, {diffs.shape[1]-1} FX-Paare)")

    # Sanity-Output: aktuelle Differenzen
    print("\nAktuelle Zinsdifferenzen (letzter Tag):")
    last = diffs.iloc[-1]
    for pair, cfg in PAIRS.items():
        if pair in diffs.columns:
            sign = "+" if last[pair] >= 0 else ""
            carry_curr = pair[:3] if last[pair] >= 0 else pair[3:]
            print(f"   {pair:<7s} ({cfg['future']}): {sign}{last[pair]:+.2f}%  -> Carry zu {carry_curr}")

    print("\nFertig.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
