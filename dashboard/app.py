"""
Trading Dashboard - Phase 3 (final)
-----------------------------------
Tabs:
  1. COT-Daten (Williams Index)
  2. Saisonalitaet (Saisonalitaetskurve, Heatmap, KW-Tabelle, Monats-Tabelle)

Aufruf:
    streamlit run dashboard/app.py
"""

from __future__ import annotations
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# -------------------------------------------------
# Seite
# -------------------------------------------------
st.set_page_config(
    page_title="Trading Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

COT_PATH   = Path("data/processed/cot_metrics.parquet")
SEAS_MONTH = Path("data/processed/seasonality_monthly.parquet")
SEAS_WEEK  = Path("data/processed/seasonality_weekly.parquet")
SEAS_CURVE = Path("data/processed/seasonality_curve.parquet")

GROUP_CHOICES = {
    "Commercials":       "commercials",
    "Large Speculators": "large_specs",
    "Small Speculators": "small_specs",
}

MARKET_NAMES = {
    "6E": "Euro FX",
    "6B": "British Pound",
    "6J": "Japanese Yen",
    "6A": "Australian Dollar",
    "6C": "Canadian Dollar",
    "GC": "Gold",
    "SI": "Silver",
    "CL": "WTI Crude Oil",
    "ES": "E-mini S&P 500",
    "NQ": "E-mini Nasdaq-100",
}

MONTH_NAMES_DE = ["Jan", "Feb", "Mär", "Apr", "Mai", "Jun",
                  "Jul", "Aug", "Sep", "Okt", "Nov", "Dez"]


# -------------------------------------------------
# Daten laden
# -------------------------------------------------
@st.cache_data(ttl=600)
def load_cot() -> pd.DataFrame:
    if not COT_PATH.exists():
        return pd.DataFrame()
    df = pd.read_parquet(COT_PATH)
    df["report_date"] = pd.to_datetime(df["report_date"])
    return df

@st.cache_data(ttl=600)
def load_seasonality():
    return {
        "monthly": pd.read_parquet(SEAS_MONTH) if SEAS_MONTH.exists() else pd.DataFrame(),
        "weekly":  pd.read_parquet(SEAS_WEEK)  if SEAS_WEEK.exists()  else pd.DataFrame(),
        "curve":   pd.read_parquet(SEAS_CURVE) if SEAS_CURVE.exists() else pd.DataFrame(),
    }


# -------------------------------------------------
# Header
# -------------------------------------------------
st.title("📊 Trading Dashboard")
st.caption("Persönliches Swing-Trading-Dashboard – Phase 3: COT + Saisonalität")

cot_df = load_cot()
seas   = load_seasonality()

if cot_df.empty:
    st.error("Keine COT-Daten gefunden. Erst `python scripts/fetch_cot.py` und "
             "`python scripts/compute_cot_index.py` ausführen.")
    st.stop()

last_cot = cot_df["report_date"].max()
st.caption(
    f"📅 Letzter COT-Report: **{last_cot.date().strftime('%d.%m.%Y')}**  |  "
    f"Saisonalität-Basis: ab 2008  |  "
    f"Dashboard geladen: {datetime.now().strftime('%d.%m.%Y %H:%M')}"
)


# -------------------------------------------------
# Sidebar
# -------------------------------------------------
with st.sidebar:
    st.header("Markt-Auswahl")
    symbol = st.selectbox(
        "Markt",
        options=list(MARKET_NAMES.keys()),
        format_func=lambda s: f"{s} – {MARKET_NAMES[s]}",
        index=5,  # GC
    )
    st.divider()
    st.header("COT-Optionen")
    group_label = st.radio(
        "Trader-Gruppe",
        options=list(GROUP_CHOICES.keys()),
        index=0,
    )
    group_key = GROUP_CHOICES[group_label]


# -------------------------------------------------
# TABS
# -------------------------------------------------
tab_cot, tab_seas = st.tabs(["📈 COT-Daten", "🗓️ Saisonalität"])


# -------------------------------------------------
# TAB 1: COT
# -------------------------------------------------
with tab_cot:
    df_m = cot_df[cot_df["symbol"] == symbol].sort_values("report_date").reset_index(drop=True)
    latest = df_m.iloc[-1]

    net_col       = f"net_{group_key}"
    chg_abs_col   = f"net_{group_key}_chg_abs"
    chg_pct_col   = f"net_{group_key}_chg_pct"
    idx26_col     = f"cot_index_{group_key}_26w"
    idx156_col    = f"cot_index_{group_key}_156w"
    idx26_chg_col = f"cot_index_{group_key}_26w_chg_abs"

    st.subheader(f"{symbol} – {MARKET_NAMES[symbol]}  ·  {group_label}")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric(
            "COT-Index 26W",
            f"{latest[idx26_col]:.1f}" if pd.notna(latest[idx26_col]) else "n/a",
            delta=f"{latest[idx26_chg_col]:+.1f}" if pd.notna(latest[idx26_chg_col]) else None,
            help="Williams COT-Index, 26-Wochen-Lookback (Hauptindikator)",
        )
    with c2:
        st.metric(
            "COT-Index 156W (3J)",
            f"{latest[idx156_col]:.1f}" if pd.notna(latest[idx156_col]) else "n/a",
        )
    with c3:
        st.metric(
            "Net Position",
            f"{int(latest[net_col]):,}".replace(",", "."),
            delta=f"{int(latest[chg_abs_col]):+,}".replace(",", ".") if pd.notna(latest[chg_abs_col]) else None,
        )
    with c4:
        st.metric(
            "Δ % zur Vorwoche",
            f"{latest[chg_pct_col]:+.2f}%" if pd.notna(latest[chg_pct_col]) else "n/a",
        )

    v = latest[idx26_col]
    if pd.notna(v):
        if v <= 20:
            st.success(f"COT-Index 26W = {v:.1f} → **extrem niedrig** "
                       f"(in den letzten 26 Wochen war die {group_label}-Net-Position fast nie so klein/short).")
        elif v >= 80:
            st.error(f"COT-Index 26W = {v:.1f} → **extrem hoch** "
                     f"(in den letzten 26 Wochen war die {group_label}-Net-Position fast nie so groß/long).")
        else:
            st.info(f"COT-Index 26W = {v:.1f} → neutraler Bereich.")

    st.subheader("Verlauf")
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.55, 0.45], vertical_spacing=0.08,
        subplot_titles=(f"Net Position – {group_label}",
                        f"COT-Index 26W & 156W – {group_label}"),
    )
    fig.add_trace(go.Scatter(x=df_m["report_date"], y=df_m[net_col],
                             mode="lines", name="Net Position",
                             line=dict(width=1.5)),
                  row=1, col=1)
    fig.add_hline(y=0, line_dash="dot", line_width=1, opacity=0.5, row=1, col=1)
    fig.add_trace(go.Scatter(x=df_m["report_date"], y=df_m[idx26_col],
                             mode="lines", name="COT-Index 26W",
                             line=dict(width=1.8)),
                  row=2, col=1)
    fig.add_trace(go.Scatter(x=df_m["report_date"], y=df_m[idx156_col],
                             mode="lines", name="COT-Index 156W",
                             line=dict(width=1.2, dash="dot"), opacity=0.7),
                  row=2, col=1)
    fig.add_hline(y=20, line_dash="dash", line_width=1, opacity=0.4, row=2, col=1)
    fig.add_hline(y=80, line_dash="dash", line_width=1, opacity=0.4, row=2, col=1)
    fig.update_yaxes(title_text="Kontrakte", row=1, col=1)
    fig.update_yaxes(title_text="Index 0–100", range=[0, 100], row=2, col=1)
    fig.update_layout(height=620, margin=dict(l=10, r=10, t=50, b=10),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                      hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Letzte 8 Wochen")
    recent = df_m.tail(8)[["report_date", net_col, chg_abs_col, idx26_col, idx156_col]].copy()
    recent.columns = ["Datum", "Net Position", "Δ Vorwoche", "COT-Index 26W", "COT-Index 156W"]
    recent["Datum"] = recent["Datum"].dt.strftime("%d.%m.%Y")
    st.dataframe(recent.iloc[::-1], use_container_width=True, hide_index=True)


# -------------------------------------------------
# TAB 2: Saisonalitaet
# -------------------------------------------------
with tab_seas:
    st.subheader(f"{symbol} – {MARKET_NAMES[symbol]}  ·  Saisonalität")

    monthly = seas["monthly"]
    weekly  = seas["weekly"]
    curve   = seas["curve"]

    if monthly.empty or weekly.empty or curve.empty:
        st.error("Saisonalitäts-Daten fehlen. Erst `python scripts/fetch_prices.py` "
                 "und `python scripts/compute_seasonality.py` ausführen.")
        st.stop()

    m_sym = monthly[monthly["symbol"] == symbol].copy()
    w_sym = weekly[weekly["symbol"] == symbol].copy()
    c_sym = curve[curve["symbol"] == symbol].copy().sort_values("doy")

    today = datetime.now()
    today_doy = today.timetuple().tm_yday
    today_kw  = today.isocalendar().week

    # ----- Saisonalitaetskurve -----
    st.markdown("##### Durchschnittlicher Jahresverlauf (alle Jahre seit 2008)")
    fig_curve = go.Figure()
    fig_curve.add_trace(go.Scatter(
        x=c_sym["doy"], y=c_sym["avg_cum_return_pct"],
        mode="lines", name="Saisonalitätskurve",
        line=dict(width=2.2),
        hovertemplate="Tag %{x}<br>Ø kumuliert: %{y:.2f}%<extra></extra>",
    ))
    fig_curve.add_hline(y=0, line_dash="dot", line_width=1, opacity=0.5)
    fig_curve.add_vline(
        x=today_doy, line_dash="dash", line_width=2, line_color="#dc2626",
        annotation_text=f"Heute (Tag {today_doy})", annotation_position="top",
    )
    month_starts = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
    for d in month_starts:
        fig_curve.add_vline(x=d, line_dash="dot", line_width=0.5, opacity=0.25)
    fig_curve.update_layout(
        height=380, margin=dict(l=10, r=10, t=30, b=10),
        xaxis=dict(title="Tag im Jahr", tickmode="array",
                   tickvals=month_starts, ticktext=MONTH_NAMES_DE),
        yaxis=dict(title="Ø kumulierter Return (%)"),
        hovermode="x unified",
    )
    st.plotly_chart(fig_curve, use_container_width=True)

    # ----- Heatmap -----
    st.markdown("##### Heatmap: Monatliche Performance pro Jahr")
    pivot = m_sym.pivot(index="year", columns="month", values="return_pct")
    pivot = pivot.reindex(columns=range(1, 13))
    pivot.columns = MONTH_NAMES_DE

    vmax = float(np.nanmax(np.abs(pivot.values))) if pivot.size else 5.0
    vmax = max(vmax, 1.0)

    fig_hm = px.imshow(
        pivot.values,
        x=pivot.columns, y=pivot.index.astype(str),
        aspect="auto",
        color_continuous_scale="RdYlGn",
        zmin=-vmax, zmax=vmax,
        labels=dict(x="Monat", y="Jahr", color="Return %"),
    )
    fig_hm.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10),
                         coloraxis_colorbar=dict(title="Return %"))
    fig_hm.update_traces(hovertemplate="Jahr %{y}<br>%{x}: %{z:.2f}%<extra></extra>")
    st.plotly_chart(fig_hm, use_container_width=True)

    # ----- Monats-Statistik -----
    st.markdown("##### Statistik pro Kalendermonat")
    stats_m = (
        m_sym.groupby("month")
             .agg(avg=("return_pct", "mean"),
                  median=("return_pct", "median"),
                  hit_rate=("return_pct", lambda s: (s > 0).mean() * 100.0),
                  n=("return_pct", "count"))
             .round(2).reset_index()
    )
    stats_m["Monat"] = stats_m["month"].apply(lambda i: MONTH_NAMES_DE[i-1])
    stats_m = stats_m[["Monat", "avg", "median", "hit_rate", "n"]]
    stats_m.columns = ["Monat", "Ø Return %", "Median %", "Trefferquote %", "Jahre"]

    current_month_label = MONTH_NAMES_DE[today.month - 1]
    def _hl_month(row):
        return ["background-color: rgba(220, 38, 38, 0.18)"] * len(row) \
               if row["Monat"] == current_month_label else [""] * len(row)
    st.dataframe(
        stats_m.style.apply(_hl_month, axis=1).format({
            "Ø Return %": "{:+.2f}",
            "Median %": "{:+.2f}",
            "Trefferquote %": "{:.0f}",
        }),
        use_container_width=True, hide_index=True,
    )

    # ----- KW-Statistik: ALLE 1..52, mit Hervorhebung der aktuellen KW -----
    st.markdown(f"##### Statistik pro Kalenderwoche  ·  aktuelle KW: **{today_kw}**")
    stats_w = (
        w_sym.groupby("iso_week")
             .agg(avg=("return_pct", "mean"),
                  median=("return_pct", "median"),
                  hit_rate=("return_pct", lambda s: (s > 0).mean() * 100.0),
                  n=("return_pct", "count"))
             .round(2).reset_index()
    )
    # KW 53 ist selten (nur in bestimmten Jahren) -> auf 1..52 begrenzen
    stats_w = stats_w[stats_w["iso_week"].between(1, 52)].copy()
    stats_w.columns = ["KW", "Ø Return %", "Median %", "Trefferquote %", "Jahre"]

    def _hl_kw(row):
        return ["background-color: rgba(220, 38, 38, 0.18)"] * len(row) \
               if int(row["KW"]) == int(today_kw) else [""] * len(row)
    st.dataframe(
        stats_w.style.apply(_hl_kw, axis=1).format({
            "Ø Return %": "{:+.2f}",
            "Median %": "{:+.2f}",
            "Trefferquote %": "{:.0f}",
        }),
        use_container_width=True, hide_index=True, height=520,
    )

    # ----- Schnell-Einordnung -----
    cur_avg = stats_m.loc[stats_m["Monat"] == current_month_label, "Ø Return %"].iloc[0]
    cur_hit = stats_m.loc[stats_m["Monat"] == current_month_label, "Trefferquote %"].iloc[0]
    if cur_avg > 0:
        st.success(f"**{current_month_label}** ist für {symbol} historisch **positiv** "
                   f"(Ø {cur_avg:+.2f}%, Trefferquote {cur_hit:.0f}%).")
    else:
        st.warning(f"**{current_month_label}** ist für {symbol} historisch **negativ** "
                   f"(Ø {cur_avg:+.2f}%, Trefferquote {cur_hit:.0f}%).")
