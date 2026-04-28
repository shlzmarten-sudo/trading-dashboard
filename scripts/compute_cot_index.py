"""
compute_cot_index.py
--------------------
Liest die Legacy-COT-Rohdaten und berechnet:
  - Net Positions pro Trader-Gruppe (Commercials, Large Specs, Small Specs)
  - COT-Index nach Williams (26 Wochen + 156 Wochen)
  - Veraenderungen zur Vorwoche (absolut + prozentual)

Speichert das Ergebnis als data/processed/cot_metrics.parquet
Aufruf:
    python scripts/compute_cot_index.py
"""

from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
import numpy as np

LEGACY_PATH    = Path("data/raw/cot_legacy.parquet")
OUT_PATH       = Path("data/processed/cot_metrics.parquet")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# Trader-Gruppen aus dem Legacy-Report
# Die Spaltennamen folgen dem Socrata-Schema der CFTC.
GROUPS = {
    "commercials": {
        "long":  "comm_positions_long_all",
        "short": "comm_positions_short_all",
        "label": "Commercials",
    },
    "large_specs": {
        "long":  "noncomm_positions_long_all",
        "short": "noncomm_positions_short_all",
        "label": "Large Speculators",
    },
    "small_specs": {
        "long":  "nonrept_positions_long_all",
        "short": "nonrept_positions_short_all",
        "label": "Small Speculators",
    },
}

# Lookback-Fenster fuer den COT-Index (in Wochen)
LOOKBACKS = [26, 156]


def williams_cot_index(series: pd.Series, window: int) -> pd.Series:
    """
    Berechnet den Williams-COT-Index (0-100) auf einer Serie von Net Positions.
    Vergleichszeitraum: rollendes Fenster der letzten 'window' Wochen
    (inklusive aktueller Woche).
    """
    rolling_min = series.rolling(window=window, min_periods=window).min()
    rolling_max = series.rolling(window=window, min_periods=window).max()
    span = rolling_max - rolling_min
    # Schutz gegen Division durch 0 (wenn Min == Max)
    idx = np.where(span == 0, np.nan, (series - rolling_min) / span * 100.0)
    return pd.Series(idx, index=series.index)


def process_one_market(df_market: pd.DataFrame) -> pd.DataFrame:
    """Verarbeitet die wöchentlichen Daten eines einzelnen Marktes."""
    df = df_market.sort_values("report_date").reset_index(drop=True).copy()

    # Positions-Spalten in Zahlen wandeln (Socrata liefert oft Strings)
    cols_to_num = []
    for g in GROUPS.values():
        cols_to_num += [g["long"], g["short"]]
    for c in cols_to_num:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Pro Gruppe: Net Position + COT-Index + Wochenveraenderungen
    for key, g in GROUPS.items():
        net_col = f"net_{key}"
        df[net_col] = df[g["long"]] - df[g["short"]]

        # Wochenveraenderungen (1 Woche = 1 Report)
        df[f"net_{key}_chg_abs"] = df[net_col].diff()
        df[f"net_{key}_chg_pct"] = df[net_col].pct_change(fill_method=None) * 100.0

        # COT-Index fuer jeden Lookback
        for w in LOOKBACKS:
            idx_col = f"cot_index_{key}_{w}w"
            df[idx_col] = williams_cot_index(df[net_col], w)
            df[f"{idx_col}_chg_abs"] = df[idx_col].diff()

    return df


def main() -> int:
    if not LEGACY_PATH.exists():
        print(f"FEHLER: {LEGACY_PATH} nicht gefunden. Erst scripts/fetch_cot.py ausfuehren.")
        return 1

    print("Lade Legacy-Rohdaten ...")
    raw = pd.read_parquet(LEGACY_PATH)
    print(f"  -> {len(raw)} Zeilen, {raw['symbol'].nunique()} Symbole")

    # Pro Markt verarbeiten und wieder zusammenfuegen
    out_frames = []
    for sym, df_sym in raw.groupby("symbol"):
        processed = process_one_market(df_sym)
        out_frames.append(processed)
        latest = processed.iloc[-1]
        print(
            f"  {sym}: letzter Report {latest['report_date'].date()}  "
            f"| COT-Idx Comm 26W={latest['cot_index_commercials_26w']:.1f}  "
            f"Specs 26W={latest['cot_index_large_specs_26w']:.1f}"
        )

    out = pd.concat(out_frames, ignore_index=True)

    # Behalte alle Original-Spalten + die neu berechneten.
    # Damit die Datei nicht riesig wird, schreiben wir nur die Schluessel-Spalten + Berechnungen.
    keep_basic = ["report_date", "symbol", "market_name"]
    keep_groups = []
    for key, g in GROUPS.items():
        keep_groups += [
            g["long"], g["short"],
            f"net_{key}",
            f"net_{key}_chg_abs", f"net_{key}_chg_pct",
        ]
        for w in LOOKBACKS:
            keep_groups += [
                f"cot_index_{key}_{w}w",
                f"cot_index_{key}_{w}w_chg_abs",
            ]

    out_slim = out[keep_basic + keep_groups].copy()
    out_slim.to_parquet(OUT_PATH, index=False)
    print(f"\nGespeichert: {OUT_PATH}  ({len(out_slim)} Zeilen, {out_slim.shape[1]} Spalten)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
