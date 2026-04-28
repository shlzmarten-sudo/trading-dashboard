"""
compute_seasonality.py  (final, gleichmaessig & glatt)
------------------------------------------------------
Berechnet aus den Tagespreisen (data/raw/prices/) saisonale Statistiken:
  - Monats-Performance pro Jahr (fuer Heatmap)
  - Wochen-Performance pro Jahr (fuer KW-Statistik)
  - Durchschnittlicher Jahresverlauf (Saisonalitaetskurve), glatt!

Schreibt nach data/processed/.
Aufruf:
    python scripts/compute_seasonality.py
"""

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

PRICES_DIR = Path("data/raw/prices")
OUT_DIR    = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_MONTHLY = OUT_DIR / "seasonality_monthly.parquet"
OUT_WEEKLY  = OUT_DIR / "seasonality_weekly.parquet"
OUT_CURVE   = OUT_DIR / "seasonality_curve.parquet"

MAX_DOY = 365  # Schaltjahr-Tag wegwerfen


def load_prices() -> pd.DataFrame:
    frames = []
    for f in sorted(PRICES_DIR.glob("*.parquet")):
        frames.append(pd.read_parquet(f))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["symbol", "date"]).reset_index(drop=True)


def compute_monthly(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for sym, g in df.groupby("symbol"):
        s = g.set_index("date")["close"].sort_index().dropna()
        m = s.resample("ME").last().dropna()
        ret = m.pct_change() * 100.0
        out.append(pd.DataFrame({
            "symbol": sym,
            "year":   ret.index.year,
            "month":  ret.index.month,
            "return_pct": ret.values,
        }).dropna())
    return pd.concat(out, ignore_index=True)


def compute_weekly(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for sym, g in df.groupby("symbol"):
        s = g.set_index("date")["close"].sort_index().dropna()
        w = s.resample("W").last().dropna()
        ret = w.pct_change() * 100.0
        iso = ret.index.isocalendar()
        out.append(pd.DataFrame({
            "symbol": sym,
            "iso_year": iso.year.values,
            "iso_week": iso.week.values,
            "return_pct": ret.values,
        }).dropna())
    return pd.concat(out, ignore_index=True)


def compute_seasonal_curve(df: pd.DataFrame) -> pd.DataFrame:
    """
    Glatte Saisonalitaetskurve:
      Pro Symbol:
        1. Reindex auf taegliche Achse, Preise ueber Wochenenden/Feiertage forward-filled.
        2. Pro Kalenderjahr: kumulierter Return seit Jahresanfang in Prozent.
        3. Pivot: Zeilen = doy (1..365), Spalten = Jahre.
        4. Mittelwert ueber alle Jahre pro doy.
      Damit steht an jedem doy ein Mittelwert aus DERSELBEN Jahres-Stichprobe.
    """
    out = []
    for sym, g in df.groupby("symbol"):
        s = g.set_index("date")["close"].sort_index().dropna()

        # Tagesachse, Wochenenden / Feiertage gefuellt
        full_idx = pd.date_range(s.index.min(), s.index.max(), freq="D")
        s_daily = s.reindex(full_idx).ffill()
        s_daily.index.name = "date"

        df_daily = s_daily.to_frame("close")
        df_daily["year"] = df_daily.index.year
        df_daily["doy"]  = df_daily.index.dayofyear

        # Schaltjahr-Tag raus
        df_daily = df_daily[df_daily["doy"] <= MAX_DOY].copy()

        # Pro Jahr: erster Tageswert als Basis -> kumulierter %-Return
        first_per_year = df_daily.groupby("year")["close"].transform("first")
        df_daily["cum_pct"] = (df_daily["close"] / first_per_year - 1.0) * 100.0

        # Pivot doy x Jahr
        pivot = df_daily.pivot_table(
            index="doy", columns="year", values="cum_pct", aggfunc="last"
        )

        # Aktuelles (laufendes) Jahr: nur bis zum letzten verfuegbaren Tag,
        # NICHT mit ffill ueber zukuenftige Tage hinaus -> sonst zerrt es das aktuelle
        # Jahr ueber das ganze Jahr und verzerrt den Mittelwert.
        last_year = int(df_daily["year"].max())
        last_day_in_last_year = int(df_daily.loc[df_daily["year"] == last_year, "doy"].max())
        if last_year in pivot.columns:
            mask_future = pivot.index > last_day_in_last_year
            pivot.loc[mask_future, last_year] = np.nan

        # Mittelwert ueber alle (verfuegbaren) Jahre pro doy
        avg = pivot.mean(axis=1, skipna=True)

        # Auf 1..365 erzwingen, kleine Luecken (sehr selten) auffuellen
        avg = avg.reindex(range(1, MAX_DOY + 1)).ffill().bfill()

        out.append(pd.DataFrame({
            "symbol": sym,
            "doy": avg.index.values,
            "avg_cum_return_pct": avg.values.round(4),
        }))
    return pd.concat(out, ignore_index=True)


def main() -> int:
    df = load_prices()
    if df.empty:
        print("FEHLER: keine Preisdaten gefunden. Erst scripts/fetch_prices.py ausfuehren.")
        return 1

    print(f"Eingelesen: {len(df)} Tageskurse, {df['symbol'].nunique()} Symbole")
    print(f"Zeitraum:   {df['date'].min().date()} -> {df['date'].max().date()}")

    print("\n[1/3] Monats-Performance ...")
    monthly = compute_monthly(df)
    monthly.to_parquet(OUT_MONTHLY, index=False)
    print(f"   -> {OUT_MONTHLY} ({len(monthly)} Zeilen)")

    print("[2/3] Wochen-Performance ...")
    weekly = compute_weekly(df)
    weekly.to_parquet(OUT_WEEKLY, index=False)
    print(f"   -> {OUT_WEEKLY} ({len(weekly)} Zeilen)")

    print("[3/3] Saisonalitaetskurve (glatt) ...")
    curve = compute_seasonal_curve(df)
    curve.to_parquet(OUT_CURVE, index=False)
    print(f"   -> {OUT_CURVE} ({len(curve)} Zeilen)")

    # Sanity-Output: Eckpunkte je Symbol
    print("\nKurven-Eckpunkte (Tag 1, 90, 180, 270, 365):")
    for sym in sorted(curve["symbol"].unique()):
        sub = curve[curve["symbol"] == sym].set_index("doy")["avg_cum_return_pct"]
        vals = [sub.loc[d] for d in [1, 90, 180, 270, 365] if d in sub.index]
        print(f"   {sym:3s}: " + "  ".join(f"{v:+6.2f}%" for v in vals))

    print("\nFertig.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
