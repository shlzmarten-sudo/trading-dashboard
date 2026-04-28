"""
Trading Dashboard - Phase 2 (minimal)
-------------------------------------
Zeigt COT-Daten und den COT-Index nach Williams fuer Martens 10 Maerkte.
Aufruf:
    streamlit run dashboard/app.py
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# -----------------------------------------------------------
# Seiten-Konfiguration
# -----------------------------------------------------------
st.set_page_config(
    page_title="Trading Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_PATH = Path("data/processed/cot_metrics.parquet")

# Trader-Gruppen-Auswahl (Anzeige -> interner Schluessel)
GROUP_CHOICES = {
    "Commercials":         "commercials",
    "Large Speculators":   "large_specs",
    "Small Speculators":   "small_specs",
}

# Symbol -> sprechender Name
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


# -----------------------------------------------------------
# Daten laden (Cache, damit das Dashboard schnell ist)
# -----------------------------------------------------------
@st.cache_data(ttl=600)  # 10 Minuten Cache
def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        return pd.DataFrame()
    df = pd.read_parquet(DATA_PATH)
    df["report_date"] = pd.to_datetime(df["report_date"])
    return df


def color_for_index(value: float) -> str:
    """Farbcodierung fuer den COT-Index (Ampel-Logik)."""
    if pd.isna(value):
        return "gray"
    if value <= 20:
        return "#16a34a"   # gruen = extrem niedrig (potentiell bullish bei Commercials)
    if value >= 80:
        return "#dc2626"   # rot   = extrem hoch
    return "#6b7280"       # grau  = neutral


# -----------------------------------------------------------
# Header
# -----------------------------------------------------------
st.title("📊 Trading Dashboard")
st.caption("Persönliches Swing-Trading-Dashboard – Phase 2: COT-Daten")

df = load_data()

if df.empty:
    st.error(
        "Keine Daten gefunden.\n\n"
        "Bitte zuerst ausführen:\n"
        "1. `python scripts/fetch_cot.py`\n"
        "2. `python scripts/compute_cot_index.py`"
    )
    st.stop()

# Letzte Aktualisierung anzeigen
last_report = df["report_date"].max()
st.caption(f"📅 Letzter COT-Report: **{last_report.date().strftime('%d.%m.%Y')}**  |  "
           f"Datenpunkte: **{len(df):,}**".replace(",", "."))

# -----------------------------------------------------------
# Sidebar - Auswahl
# -----------------------------------------------------------
with st.sidebar:
    st.header("Auswahl")
    symbol = st.selectbox(
        "Markt",
        options=list(MARKET_NAMES.keys()),
        format_func=lambda s: f"{s} – {MARKET_NAMES[s]}",
        index=5,  # GC als Default, da ein klassischer COT-Markt
    )
    group_label = st.radio(
        "Trader-Gruppe (für COT-Index)",
        options=list(GROUP_CHOICES.keys()),
        index=0,
    )
    group_key = GROUP_CHOICES[group_label]

    st.divider()
    st.caption(
        "Der COT-Index nach Williams skaliert die Net Position auf 0–100, "
        "bezogen auf die letzten 26 Wochen. \n\n"
        "**0** = niedrigster Net-Wert der Periode  \n"
        "**100** = höchster Net-Wert der Periode"
    )


# -----------------------------------------------------------
# Daten fuer ausgewaehlten Markt
# -----------------------------------------------------------
df_m = df[df["symbol"] == symbol].sort_values("report_date").reset_index(drop=True)
latest = df_m.iloc[-1]

net_col       = f"net_{group_key}"
chg_abs_col   = f"net_{group_key}_chg_abs"
chg_pct_col   = f"net_{group_key}_chg_pct"
idx26_col     = f"cot_index_{group_key}_26w"
idx156_col    = f"cot_index_{group_key}_156w"
idx26_chg_col = f"cot_index_{group_key}_26w_chg_abs"

# -----------------------------------------------------------
# Hauptbereich - Kennzahlen
# -----------------------------------------------------------
st.subheader(f"{symbol} – {MARKET_NAMES[symbol]}  ·  {group_label}")

# Drei prominente "Big Numbers"
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric(
        label="COT-Index 26W",
        value=f"{latest[idx26_col]:.1f}" if pd.notna(latest[idx26_col]) else "n/a",
        delta=f"{latest[idx26_chg_col]:+.1f}" if pd.notna(latest[idx26_chg_col]) else None,
        help="Williams COT-Index, 26-Wochen-Lookback (dein Hauptindikator)",
    )

with c2:
    st.metric(
        label="COT-Index 156W (3J)",
        value=f"{latest[idx156_col]:.1f}" if pd.notna(latest[idx156_col]) else "n/a",
        help="Vergleichswert mit langem Lookback (3 Jahre)",
    )

with c3:
    st.metric(
        label="Net Position",
        value=f"{int(latest[net_col]):,}".replace(",", "."),
        delta=f"{int(latest[chg_abs_col]):+,}".replace(",", ".") if pd.notna(latest[chg_abs_col]) else None,
        help="Long minus Short, aktuelle Woche",
    )

with c4:
    st.metric(
        label="Veränderung % zur Vorwoche",
        value=f"{latest[chg_pct_col]:+.2f}%" if pd.notna(latest[chg_pct_col]) else "n/a",
    )

# -----------------------------------------------------------
# Schnell-Einordnung des Index-Werts
# -----------------------------------------------------------
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

# -----------------------------------------------------------
# Verlaufs-Chart: Net Position + COT-Index
# -----------------------------------------------------------
st.subheader("Verlauf")

fig = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    row_heights=[0.55, 0.45],
    vertical_spacing=0.08,
    subplot_titles=(f"Net Position – {group_label}",
                    f"COT-Index 26W & 156W – {group_label}"),
)

# Oben: Net Position
fig.add_trace(
    go.Scatter(
        x=df_m["report_date"], y=df_m[net_col],
        mode="lines", name="Net Position",
        line=dict(width=1.5),
    ),
    row=1, col=1,
)
fig.add_hline(y=0, line_dash="dot", line_width=1, opacity=0.5, row=1, col=1)

# Unten: COT-Index 26W + 156W
fig.add_trace(
    go.Scatter(
        x=df_m["report_date"], y=df_m[idx26_col],
        mode="lines", name="COT-Index 26W",
        line=dict(width=1.8),
    ),
    row=2, col=1,
)
fig.add_trace(
    go.Scatter(
        x=df_m["report_date"], y=df_m[idx156_col],
        mode="lines", name="COT-Index 156W",
        line=dict(width=1.2, dash="dot"),
        opacity=0.7,
    ),
    row=2, col=1,
)
fig.add_hline(y=20, line_dash="dash", line_width=1, opacity=0.4, row=2, col=1)
fig.add_hline(y=80, line_dash="dash", line_width=1, opacity=0.4, row=2, col=1)

fig.update_yaxes(title_text="Kontrakte", row=1, col=1)
fig.update_yaxes(title_text="Index 0–100", range=[0, 100], row=2, col=1)
fig.update_layout(
    height=620,
    margin=dict(l=10, r=10, t=50, b=10),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    hovermode="x unified",
)

st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------
# Tabelle: letzte 8 Wochen
# -----------------------------------------------------------
st.subheader("Letzte 8 Wochen")
recent = df_m.tail(8)[["report_date", net_col, chg_abs_col, idx26_col, idx156_col]].copy()
recent.columns = ["Datum", "Net Position", "Δ Vorwoche", "COT-Index 26W", "COT-Index 156W"]
recent["Datum"] = recent["Datum"].dt.strftime("%d.%m.%Y")
st.dataframe(recent.iloc[::-1], use_container_width=True, hide_index=True)
