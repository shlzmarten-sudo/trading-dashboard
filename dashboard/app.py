"""
Trading Dashboard - Phase 5.3 (final)
Features:
  - Cockpit (Übersicht aller 10 Märkte mit Bias-Ampel)
  - COT, Saisonalität, Zinsen (jeweils eigene Tabs)
  - Watchlist + Notizen pro Markt (persistent in JSON)
  - Refresh-Button, Datenalter-Warnungen
  - Bias-Score nach 4-Säulen-Framework
"""

from __future__ import annotations
import json
from datetime import datetime, timedelta
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

# -------------------------------------------------
# Globales CSS
# -------------------------------------------------
st.markdown("""
<style>
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
[data-testid="stMetricValue"] {
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-weight: 600; letter-spacing: 0.3px;
}
[data-testid="stMetricLabel"] {
    text-transform: uppercase; font-size: 0.72rem;
    letter-spacing: 1px; opacity: 0.75;
}
button[data-baseweb="tab"] {
    font-size: 0.95rem; letter-spacing: 0.5px;
    padding-top: 0.7rem; padding-bottom: 0.7rem;
}
section[data-testid="stSidebar"] { border-right: 1px solid #1f2430; }

.cockpit-card {
    background: #161A23; border: 1px solid #1f2430;
    border-radius: 8px; padding: 12px 14px; margin-bottom: 8px;
    transition: border-color 0.15s ease;
}
.cockpit-card:hover { border-color: #FFB000; }
.cockpit-symbol {
    font-family: ui-monospace, monospace; font-weight: 700;
    font-size: 1.05rem; color: #FFB000; letter-spacing: 1px;
}
.cockpit-name {
    color: #9aa0ad; font-size: 0.72rem;
    text-transform: uppercase; letter-spacing: 1px; margin-top: 2px;
}
.cockpit-row {
    display: flex; justify-content: space-between; align-items: baseline;
    margin-top: 6px; font-size: 0.82rem;
}
.cockpit-row .lbl {
    color: #9aa0ad; text-transform: uppercase;
    letter-spacing: 0.8px; font-size: 0.7rem;
}
.cockpit-row .val { font-family: ui-monospace, monospace; font-weight: 600; }
.cot-low  { color: #4ADE80; }
.cot-high { color: #F87171; }
.cot-mid  { color: #E6E8EE; }

.bias-pill {
    display: inline-block; padding: 2px 10px; border-radius: 12px;
    font-family: ui-monospace, monospace; font-weight: 700; font-size: 0.78rem;
    letter-spacing: 1px;
}
.bias-bull-strong { background: rgba(74,222,128,0.20); color: #4ADE80; border: 1px solid #4ADE80; }
.bias-bull        { background: rgba(74,222,128,0.10); color: #86EFAC; }
.bias-neutral     { background: rgba(154,160,173,0.10); color: #9aa0ad; }
.bias-bear        { background: rgba(248,113,113,0.10); color: #FCA5A5; }
.bias-bear-strong { background: rgba(248,113,113,0.20); color: #F87171; border: 1px solid #F87171; }
.bias-na          { background: rgba(154,160,173,0.05); color: #6b7280; }

.star { color: #FFB000; font-weight: 700; }
.muted { color: #9aa0ad; }

[data-testid="stDataFrame"] {
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 0.86rem;
}
.dashboard-header-meta {
    color: #9aa0ad; font-size: 0.78rem;
    letter-spacing: 0.5px; margin-top: -4px;
}
.warn-stale {
    background: rgba(248,113,113,0.10);
    border-left: 3px solid #F87171;
    padding: 6px 10px; border-radius: 4px;
    color: #FCA5A5; font-size: 0.82rem; margin: 6px 0;
}
</style>
""", unsafe_allow_html=True)


# -------------------------------------------------
# Pfade & Konstanten
# -------------------------------------------------
COT_PATH    = Path("data/processed/cot_metrics.parquet")
SEAS_MONTH  = Path("data/processed/seasonality_monthly.parquet")
SEAS_WEEK   = Path("data/processed/seasonality_weekly.parquet")
SEAS_CURVE  = Path("data/processed/seasonality_curve.parquet")
RATES_WIDE  = Path("data/processed/rates_wide.parquet")
RATE_DIFFS  = Path("data/processed/rate_diffs.parquet")
USER_STATE  = Path("data/processed/user_state.json")  # Watchlist + Notizen

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
}
FX_SYMBOLS = {"6E", "6B", "6J", "6A", "6C"}
FUTURE_TO_PAIR = {
    "6E": "EURUSD", "6B": "GBPUSD", "6J": "USDJPY",
    "6A": "AUDUSD", "6C": "USDCAD",
}
CB_ORDER  = ["FED", "ECB", "BOE", "BOJ", "RBA", "BOC"]
CB_LABELS = {"FED":"FED (USD)","ECB":"EZB (EUR)","BOE":"BoE (GBP)",
             "BOJ":"BoJ (JPY)","RBA":"RBA (AUD)","BOC":"BoC (CAD)"}
MONTH_NAMES_DE = ["Jan","Feb","Mär","Apr","Mai","Jun",
                  "Jul","Aug","Sep","Okt","Nov","Dez"]


# -------------------------------------------------
# Watchlist + Notizen (lokales JSON)
# -------------------------------------------------
def load_user_state() -> dict:
    if USER_STATE.exists():
        try:
            return json.loads(USER_STATE.read_text())
        except Exception:
            pass
    return {"watchlist": [], "notes": {}}

def save_user_state(state: dict) -> None:
    USER_STATE.parent.mkdir(parents=True, exist_ok=True)
    USER_STATE.write_text(json.dumps(state, indent=2, ensure_ascii=False))


# -------------------------------------------------
# Daten-Loader
# -------------------------------------------------
@st.cache_data(ttl=600)
def load_cot():
    if not COT_PATH.exists(): return pd.DataFrame()
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

@st.cache_data(ttl=600)
def load_rates():
    wide  = pd.read_parquet(RATES_WIDE) if RATES_WIDE.exists() else pd.DataFrame()
    diffs = pd.read_parquet(RATE_DIFFS) if RATE_DIFFS.exists() else pd.DataFrame()
    if not wide.empty: wide.index = pd.to_datetime(wide.index)
    if not diffs.empty: diffs["date"] = pd.to_datetime(diffs["date"])
    return wide, diffs

@st.cache_data(ttl=600)
def load_gold_drivers():
    p = Path("data/processed/gold_drivers.parquet")
    if not p.exists(): return pd.DataFrame()
    df = pd.read_parquet(p)
    df["date"] = pd.to_datetime(df["date"])
    return df


# -------------------------------------------------
# Bias-Berechnung (4-Säulen-Framework)
# -------------------------------------------------
def score_cot(idx_26w: float) -> int:
    """Score basiert nur auf Commercials (Smart Money).
    Hohe Werte = Commercials kaufen Shorts zurück = bullish.
    Niedrige Werte = Commercials hedgen maximal ab = bearish."""
    if pd.isna(idx_26w): return 0
    if idx_26w >= 90:  return +2   # extrem bullish
    if idx_26w >= 75:  return +1   # leicht bullish
    if idx_26w <= 10:  return -2   # extrem bearish
    if idx_26w <= 25:  return -1   # leicht bearish
    return 0                       # neutraler Bereich (26-74)

def score_season(monthly_df: pd.DataFrame, weekly_df: pd.DataFrame, sym: str) -> int:
    """Saison-Score: Monats-Komponente (max ±2) + KW-Komponente (max ±1),
    gekappt auf [-2, +2]. KW-Komponente wird pro KW einzeln bewertet,
    damit klare Signale einer einzelnen KW nicht durch die Mittelung untergehen."""
    if monthly_df.empty: return 0
    m = monthly_df[monthly_df["symbol"] == sym]
    if m.empty: return 0
    today = datetime.now()
    cur_m  = today.month
    next_m = (cur_m % 12) + 1
    cur_kw = today.isocalendar().week

    # --- Monats-Komponente (max ±2) ---
    avg_cur = m[m["month"] == cur_m]["return_pct"].mean()
    hit_cur = (m[m["month"] == cur_m]["return_pct"] > 0).mean() * 100.0
    avg_nxt = m[m["month"] == next_m]["return_pct"].mean()
    if pd.isna(avg_cur) or pd.isna(avg_nxt):
        month_score = 0
    elif avg_cur >= 1.5 and avg_nxt > 0 and hit_cur >= 65:
        month_score = +2
    elif avg_cur <= -1.5 and avg_nxt < 0 and hit_cur <= 35:
        month_score = -2
    elif avg_cur > 0 and avg_nxt > 0:
        month_score = +1
    elif avg_cur < 0 and avg_nxt < 0:
        month_score = -1
    else:
        month_score = 0

    # --- KW-Komponente (max ±1): aktuelle + nächste KW EINZELN bewerten ---
    def _kw_score_single(kw: int) -> int:
        sub = weekly_df[(weekly_df["symbol"] == sym) & (weekly_df["iso_week"] == kw)]["return_pct"]
        if len(sub) < 8:  # mind. 8 Jahre Daten
            return 0
        avg = sub.mean()
        hit = (sub > 0).mean() * 100.0
        if avg >= 0.5 and hit >= 60:
            return +1
        if avg <= -0.5 and hit <= 40:
            return -1
        return 0

    kw_score = 0
    if not weekly_df.empty:
        next_kw = cur_kw + 1 if cur_kw < 52 else 1
        s_cur = _kw_score_single(cur_kw)
        s_nxt = _kw_score_single(next_kw)
        # Mittelwert der beiden Einzel-Scores, kaufmaennisch gerundet
        avg_kw = (s_cur + s_nxt) / 2.0
        if avg_kw >= 0.5:    kw_score = +1
        elif avg_kw <= -0.5: kw_score = -1
        else:                kw_score = 0

    # --- Kombinieren, Cap bei [-2, +2] ---
    return max(-2, min(+2, month_score + kw_score))

def score_rates(diffs_df: pd.DataFrame, sym: str) -> int:
    """Trend der Zinsdifferenz der letzten 90 Tage."""
    pair = FUTURE_TO_PAIR.get(sym)
    if pair is None or diffs_df.empty or pair not in diffs_df.columns:
        return 0
    sub = diffs_df[["date", pair]].dropna()
    if len(sub) < 100: return 0
    last = sub[pair].iloc[-1]
    prev = sub[pair].iloc[-90]
    delta = last - prev
    if delta >= 0.5:   return +2
    if delta >= 0.15:  return +1
    if delta <= -0.5:  return -2
    if delta <= -0.15: return -1
    return 0

def score_gold_drivers(gd_df: pd.DataFrame) -> int:
    """Bias-Score für Gold-Treiber: Kombiniert DXY und Realzins.
    Beide korrelieren historisch negativ mit Gold.
    Fallender DXY oder fallender Realzins -> bullisch für Gold."""
    if gd_df.empty: return 0
    last = gd_df.dropna().iloc[-1]
    d_dxy = last.get("dxy_chg_90d", float('nan'))
    d_ry  = last.get("real_yield_chg_90d", float('nan'))

    # DXY-Sub-Score (negative Korrelation zu Gold -> Vorzeichen umdrehen)
    if pd.isna(d_dxy):    s_dxy = 0
    elif d_dxy <= -2.0:   s_dxy = +2
    elif d_dxy <= -0.5:   s_dxy = +1
    elif d_dxy >=  2.0:   s_dxy = -2
    elif d_dxy >=  0.5:   s_dxy = -1
    else:                 s_dxy = 0

    # Realzins-Sub-Score (negative Korrelation zu Gold)
    if pd.isna(d_ry):     s_ry = 0
    elif d_ry <= -0.25:   s_ry = +2
    elif d_ry <= -0.10:   s_ry = +1
    elif d_ry >=  0.25:   s_ry = -2
    elif d_ry >=  0.10:   s_ry = -1
    else:                 s_ry = 0

    # Mittelwert, kaufmaennisch gerundet, Cap bei [-2, +2]
    avg = (s_dxy + s_ry) / 2.0
    if avg >= 1.5:    return +2
    if avg >= 0.5:    return +1
    if avg <= -1.5:   return -2
    if avg <= -0.5:   return -1
    return 0


def compute_bias(sym: str, cot_df, monthly_df, weekly_df, diffs_df, gold_drivers_df) -> dict:
    """Bias-Score basiert auf:
       FX (5 Märkte): COT + Saison + Zinsdifferenz
       Gold:           COT + Saison + Gold-Treiber (DXY + Realzins)"""
    sub = cot_df[cot_df["symbol"] == sym].sort_values("report_date")
    cot_idx = sub["cot_index_commercials_26w"].iloc[-1] if not sub.empty else np.nan
    s_cot = score_cot(cot_idx)
    s_sea = score_season(monthly_df, weekly_df, sym)

    if sym in FX_SYMBOLS:
        s_third = score_rates(diffs_df, sym)
        third_label = "Zinsen"
    elif sym == "GC":
        s_third = score_gold_drivers(gold_drivers_df)
        third_label = "Gold-Treiber"
    else:
        s_third = 0
        third_label = "—"

    total = s_cot + s_sea + s_third

    if   total >= 4:  label, css = "STARK BULLISH", "bias-bull-strong"
    elif total >= 2:  label, css = "BULLISH",       "bias-bull"
    elif total <= -4: label, css = "STARK BEARISH", "bias-bear-strong"
    elif total <= -2: label, css = "BEARISH",       "bias-bear"
    else:             label, css = "NEUTRAL",       "bias-neutral"

    return {"cot": s_cot, "sea": s_sea, "rates": s_third,
            "third_label": third_label,
            "total": total, "label": label, "css": css,
            "is_fx": sym in FX_SYMBOLS, "is_gold": sym == "GC"}

    if   total >= 4:  label, css = "STARK BULLISH", "bias-bull-strong"
    elif total >= 2:  label, css = "BULLISH",       "bias-bull"
    elif total <= -4: label, css = "STARK BEARISH", "bias-bear-strong"
    elif total <= -2: label, css = "BEARISH",       "bias-bear"
    else:             label, css = "NEUTRAL",       "bias-neutral"

    return {"cot": s_cot, "sea": s_sea, "rates": s_rat,
            "total": total, "label": label, "css": css,
            "is_fx": sym in FX_SYMBOLS}


# -------------------------------------------------
# Datenalter-Check
# -------------------------------------------------
def stale_warning(name: str, last_dt, max_days: int) -> str | None:
    if last_dt is None: return f"{name}: keine Daten"
    age = (datetime.now() - pd.Timestamp(last_dt)).days
    if age > max_days:
        return f"{name} ist <b>{age} Tage</b> alt (max {max_days})"
    return None


# -------------------------------------------------
# Header
# -------------------------------------------------
st.markdown("## 📊 Trading Dashboard")

cot_df = load_cot()
seas   = load_seasonality()
rates_wide, rate_diffs = load_rates()
gold_drivers = load_gold_drivers()
user_state = load_user_state()

if cot_df.empty:
    st.error("Keine COT-Daten. Erst die Skripte in scripts/ ausführen.")
    st.stop()

last_cot  = cot_df["report_date"].max()
last_rate = rates_wide.index.max() if not rates_wide.empty else None

st.markdown(
    f"<div class='dashboard-header-meta'>"
    f"Letzter COT-Report: <b>{last_cot.date().strftime('%d.%m.%Y')}</b>  ·  "
    f"Saisonalität-Basis: ab 2008  ·  "
    f"Zinsen: <b>{last_rate.date().strftime('%d.%m.%Y') if last_rate is not None else 'n/a'}</b>  ·  "
    f"Geladen: {datetime.now().strftime('%d.%m.%Y %H:%M')}"
    f"</div>",
    unsafe_allow_html=True,
)

# Datenalter-Warnungen
warns = []
w1 = stale_warning("COT-Report", last_cot, 10)
w2 = stale_warning("Zinsdaten", last_rate, 60)
if w1: warns.append(w1)
if w2: warns.append(w2)
for w in warns:
    st.markdown(f"<div class='warn-stale'>⚠️ {w}</div>", unsafe_allow_html=True)


# -------------------------------------------------
# Sidebar
# -------------------------------------------------
with st.sidebar:
    st.markdown("### Markt-Auswahl")
    symbol = st.selectbox(
        "Markt", options=list(MARKET_NAMES.keys()),
        format_func=lambda s: f"{s} – {MARKET_NAMES[s]}", index=5,
    )

    # Watchlist-Toggle für aktuelles Symbol
    in_wl = symbol in user_state["watchlist"]
    btn_label = "★ Aus Watchlist entfernen" if in_wl else "☆ In Watchlist aufnehmen"
    if st.button(btn_label, use_container_width=True):
        if in_wl:
            user_state["watchlist"].remove(symbol)
        else:
            user_state["watchlist"].append(symbol)
        save_user_state(user_state)
        st.rerun()

    # Watchlist-Anzeige
    if user_state["watchlist"]:
        st.markdown("**Watchlist:** " + " · ".join(
            f"<span class='star'>★</span> {s}" for s in user_state["watchlist"]
        ), unsafe_allow_html=True)

    st.divider()
    st.markdown("### COT-Optionen")
    group_label = st.radio("Trader-Gruppe",
                           options=list(GROUP_CHOICES.keys()), index=0)
    group_key = GROUP_CHOICES[group_label]

    st.divider()
    if st.button("🔄 Daten-Cache leeren", use_container_width=True,
                 help="Liest die Parquet-Dateien neu ein (nach manuellem Update)."):
        st.cache_data.clear()
        st.rerun()


# -------------------------------------------------
# Cockpit-Card-Helper
# -------------------------------------------------
def cockpit_card_html(sym, cot_df, seas_monthly, weekly_df, diffs_df, gold_drivers_df, user_state) -> str:
    sub = cot_df[cot_df["symbol"] == sym].sort_values("report_date")
    if sub.empty:
        return f"<div class='cockpit-card'><div class='cockpit-symbol'>{sym}</div></div>"
    last = sub.iloc[-1]
    cot_idx = last.get("cot_index_commercials_26w", np.nan)
    net_chg = last.get("net_commercials_chg_abs", np.nan)

    arrow = "▲" if pd.notna(net_chg) and net_chg > 0 else ("▼" if pd.notna(net_chg) and net_chg < 0 else "·")
    arrow_color = "#4ADE80" if arrow == "▲" else ("#F87171" if arrow == "▼" else "#9aa0ad")

    if pd.notna(cot_idx):
        cls = "cot-low" if cot_idx <= 20 else ("cot-high" if cot_idx >= 80 else "cot-mid")
        cot_str = f"<span class='{cls}'>{cot_idx:.0f}</span>"
    else:
        cot_str = "<span class='cot-mid'>–</span>"

    seas_str = "–"
    if not seas_monthly.empty:
        m = seas_monthly[seas_monthly["symbol"] == sym]
        if not m.empty:
            cur_month = datetime.now().month
            avg = m[m["month"] == cur_month]["return_pct"].mean()
            if pd.notna(avg):
                color = "#4ADE80" if avg > 0 else "#F87171"
                seas_str = f"<span style='color:{color}'>{avg:+.2f}%</span>"

    bias = compute_bias(sym, cot_df, seas_monthly, weekly_df, diffs_df, gold_drivers_df)
    star = "<span class='star'>★</span> " if sym in user_state["watchlist"] else ""

    return (
        f"<div class='cockpit-card'>"
        f"<div class='cockpit-symbol'>{star}{sym}</div>"
        f"<div class='cockpit-name'>{MARKET_NAMES[sym]}</div>"
        f"<div class='cockpit-row'><span class='lbl'>Bias</span>"
        f"<span class='val'><span class='bias-pill {bias['css']}'>{bias['label']}</span></span></div>"
        f"<div class='cockpit-row'><span class='lbl'>COT 26W</span>"
        f"<span class='val'>{cot_str}</span></div>"
        f"<div class='cockpit-row'><span class='lbl'>Net-Trend</span>"
        f"<span class='val' style='color:{arrow_color}'>{arrow}</span></div>"
        f"<div class='cockpit-row'><span class='lbl'>Ø {MONTH_NAMES_DE[datetime.now().month-1]}</span>"
        f"<span class='val'>{seas_str}</span></div>"
        f"</div>"
    )


# -------------------------------------------------
# TABS
# -------------------------------------------------
tab_over, tab_cot, tab_seas, tab_rates, tab_notes = st.tabs(
    ["🧭 Übersicht", "📈 COT-Daten", "🗓️ Saisonalität", "💰 Zinsen / Gold-Treiber", "📝 Notizen"]
)


# =================================================
# TAB 0: Übersicht
# =================================================
with tab_over:
    st.caption(
        "Schneller Überblick. Bias = automatischer 4-Säulen-Score "
        "(COT · Saisonalität · Zinsen). ★ = auf der Watchlist."
    )
    syms = list(MARKET_NAMES.keys())
    cols = st.columns(5)
    for i, sym in enumerate(syms):
        with cols[i % 5]:
            st.markdown(
                cockpit_card_html(sym, cot_df, seas["monthly"], seas["weekly"], rate_diffs, gold_drivers, user_state), 
                unsafe_allow_html=True,
            )

    st.divider()

    # Bias-Tabelle (Detail)
    st.markdown("##### Bias-Details (alle Märkte)")
    rows = []
    for sym in syms:
        b = compute_bias(sym, cot_df, seas["monthly"], seas["weekly"], rate_diffs, gold_drivers)
        rows.append({
            "★": "★" if sym in user_state["watchlist"] else "",
            "Symbol": sym,
            "Markt": MARKET_NAMES[sym],
            "COT": b["cot"],
            "Saison": b["sea"],
            "Zinsen / Treiber": b["rates"] if (b["is_fx"] or b["is_gold"]) else "—",
            "Total": b["total"],
            "Bias": b["label"],
        })
    bias_df = pd.DataFrame(rows)

    def _color_bias(val):
        if "BULL" in str(val):  return "color:#4ADE80; font-weight:700"
        if "BEAR" in str(val):  return "color:#F87171; font-weight:700"
        return "color:#9aa0ad"
    st.dataframe(
        bias_df.style.map(_color_bias, subset=["Bias"]),
        use_container_width=True, hide_index=True,
    )

    if not rate_diffs.empty:
        st.markdown("##### Aktuelle Carry-Übersicht (FX-Paare)")
        last = rate_diffs.iloc[-1]
        rows = []
        for fut, pair in FUTURE_TO_PAIR.items():
            if pair in rate_diffs.columns and pd.notna(last[pair]):
                v = last[pair]
                carry = pair[:3] if v > 0 else (pair[3:] if v < 0 else "—")
                rows.append({"Future": fut, "FX-Paar": pair,
                             "Zinsdifferenz": f"{v:+.2f}%", "Carry zu": carry})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# =================================================
# TAB 1: COT
# =================================================
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
        st.metric("COT-Index 26W",
                  f"{latest[idx26_col]:.1f}" if pd.notna(latest[idx26_col]) else "n/a",
                  delta=f"{latest[idx26_chg_col]:+.1f}" if pd.notna(latest[idx26_chg_col]) else None)
    with c2:
        st.metric("COT-Index 156W (3J)",
                  f"{latest[idx156_col]:.1f}" if pd.notna(latest[idx156_col]) else "n/a")
    with c3:
        st.metric("Net Position",
                  f"{int(latest[net_col]):,}".replace(",", "."),
                  delta=f"{int(latest[chg_abs_col]):+,}".replace(",", ".") if pd.notna(latest[chg_abs_col]) else None)
    with c4:
        st.metric("Δ % zur Vorwoche",
                  f"{latest[chg_pct_col]:+.2f}%" if pd.notna(latest[chg_pct_col]) else "n/a")

    v = latest[idx26_col]
    if pd.notna(v):
        if v <= 20:
            st.success(f"COT-Index 26W = {v:.1f} → **extrem niedrig**.")
        elif v >= 80:
            st.error(f"COT-Index 26W = {v:.1f} → **extrem hoch**.")
        else:
            st.info(f"COT-Index 26W = {v:.1f} → neutraler Bereich.")

    st.subheader("Verlauf")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.55, 0.45], vertical_spacing=0.08,
        subplot_titles=(f"Net Position – {group_label}",
                        f"COT-Index 26W & 156W – {group_label}"))
    fig.add_trace(go.Scatter(x=df_m["report_date"], y=df_m[net_col],
        mode="lines", name="Net Position",
        line=dict(width=1.5, color="#FFB000")), row=1, col=1)
    fig.add_hline(y=0, line_dash="dot", line_width=1, opacity=0.5, row=1, col=1)
    fig.add_trace(go.Scatter(x=df_m["report_date"], y=df_m[idx26_col],
        mode="lines", name="COT-Index 26W",
        line=dict(width=1.8, color="#FFB000")), row=2, col=1)
    fig.add_trace(go.Scatter(x=df_m["report_date"], y=df_m[idx156_col],
        mode="lines", name="COT-Index 156W",
        line=dict(width=1.2, dash="dot", color="#29BEFD"), opacity=0.7), row=2, col=1)
    fig.add_hline(y=20, line_dash="dash", line_width=1, opacity=0.4, row=2, col=1)
    fig.add_hline(y=80, line_dash="dash", line_width=1, opacity=0.4, row=2, col=1)
    fig.update_yaxes(title_text="Kontrakte", row=1, col=1)
    fig.update_yaxes(title_text="Index 0–100", range=[0, 100], row=2, col=1)
    fig.update_layout(height=620, margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        plot_bgcolor="#0E1117", paper_bgcolor="#0E1117", font=dict(color="#E6E8EE"))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Letzte 8 Wochen")
    recent = df_m.tail(8)[["report_date", net_col, chg_abs_col, idx26_col, idx156_col]].copy()
    recent.columns = ["Datum", "Net Position", "Δ Vorwoche", "COT-Index 26W", "COT-Index 156W"]
    recent["Datum"] = recent["Datum"].dt.strftime("%d.%m.%Y")
    st.dataframe(recent.iloc[::-1], use_container_width=True, hide_index=True)


# =================================================
# TAB 2: Saisonalität
# =================================================
with tab_seas:
    st.subheader(f"{symbol} – {MARKET_NAMES[symbol]}  ·  Saisonalität")
    monthly = seas["monthly"]; weekly = seas["weekly"]; curve = seas["curve"]
    if monthly.empty or weekly.empty or curve.empty:
        st.error("Saisonalitäts-Daten fehlen.")
    else:
        m_sym = monthly[monthly["symbol"] == symbol].copy()
        w_sym = weekly[weekly["symbol"] == symbol].copy()
        c_sym = curve[curve["symbol"] == symbol].copy().sort_values("doy")
        today = datetime.now()
        today_doy = today.timetuple().tm_yday
        today_kw  = today.isocalendar().week

        st.markdown("##### Durchschnittlicher Jahresverlauf (alle Jahre seit 2008)")
        fig_curve = go.Figure()
        fig_curve.add_trace(go.Scatter(x=c_sym["doy"], y=c_sym["avg_cum_return_pct"],
            mode="lines", name="Saisonalitätskurve",
            line=dict(width=2.2, color="#FFB000"),
            hovertemplate="Tag %{x}<br>Ø kumuliert: %{y:.2f}%<extra></extra>"))
        fig_curve.add_hline(y=0, line_dash="dot", line_width=1, opacity=0.5)
        fig_curve.add_vline(x=today_doy, line_dash="dash", line_width=2, line_color="#F87171",
            annotation_text=f"Heute (Tag {today_doy})", annotation_position="top")
        month_starts = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
        for d in month_starts:
            fig_curve.add_vline(x=d, line_dash="dot", line_width=0.5, opacity=0.25)
        fig_curve.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10),
            xaxis=dict(title="Tag im Jahr", tickmode="array",
                       tickvals=month_starts, ticktext=MONTH_NAMES_DE),
            yaxis=dict(title="Ø kumulierter Return (%)"),
            hovermode="x unified",
            plot_bgcolor="#0E1117", paper_bgcolor="#0E1117", font=dict(color="#E6E8EE"))
        st.plotly_chart(fig_curve, use_container_width=True)

        st.markdown("##### Heatmap: Monatliche Performance pro Jahr")
        pivot = m_sym.pivot(index="year", columns="month", values="return_pct")
        pivot = pivot.reindex(columns=range(1, 13))
        pivot.columns = MONTH_NAMES_DE
        vmax = float(np.nanmax(np.abs(pivot.values))) if pivot.size else 5.0
        vmax = max(vmax, 1.0)
        fig_hm = px.imshow(pivot.values, x=pivot.columns, y=pivot.index.astype(str),
            aspect="auto", color_continuous_scale="RdYlGn", zmin=-vmax, zmax=vmax,
            labels=dict(x="Monat", y="Jahr", color="Return %"))
        fig_hm.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10),
            coloraxis_colorbar=dict(title="Return %"),
            plot_bgcolor="#0E1117", paper_bgcolor="#0E1117", font=dict(color="#E6E8EE"))
        fig_hm.update_traces(hovertemplate="Jahr %{y}<br>%{x}: %{z:.2f}%<extra></extra>")
        st.plotly_chart(fig_hm, use_container_width=True)

        st.markdown("##### Statistik pro Kalendermonat")
        stats_m = (m_sym.groupby("month")
                   .agg(avg=("return_pct","mean"),
                        median=("return_pct","median"),
                        hit_rate=("return_pct", lambda s:(s>0).mean()*100.0),
                        n=("return_pct","count"))
                   .round(2).reset_index())
        stats_m["Monat"] = stats_m["month"].apply(lambda i: MONTH_NAMES_DE[i-1])
        stats_m = stats_m[["Monat","avg","median","hit_rate","n"]]
        stats_m.columns = ["Monat","Ø Return %","Median %","Trefferquote %","Jahre"]
        cur_month_label = MONTH_NAMES_DE[today.month-1]
        def _hl_month(row):
            return ["background-color: rgba(255,176,0,0.15)"]*len(row) if row["Monat"]==cur_month_label else [""]*len(row)
        st.dataframe(stats_m.style.apply(_hl_month, axis=1).format({
            "Ø Return %":"{:+.2f}", "Median %":"{:+.2f}", "Trefferquote %":"{:.0f}",
        }), use_container_width=True, hide_index=True)

        st.markdown(f"##### Statistik pro Kalenderwoche  ·  aktuelle KW: **{today_kw}**")
        stats_w = (w_sym.groupby("iso_week")
                   .agg(avg=("return_pct","mean"),
                        median=("return_pct","median"),
                        hit_rate=("return_pct", lambda s:(s>0).mean()*100.0),
                        n=("return_pct","count"))
                   .round(2).reset_index())
        stats_w = stats_w[stats_w["iso_week"].between(1,52)].copy()
        stats_w.columns = ["KW","Ø Return %","Median %","Trefferquote %","Jahre"]
        def _hl_kw(row):
            return ["background-color: rgba(255,176,0,0.15)"]*len(row) if int(row["KW"])==int(today_kw) else [""]*len(row)
        st.dataframe(stats_w.style.apply(_hl_kw, axis=1).format({
            "Ø Return %":"{:+.2f}", "Median %":"{:+.2f}", "Trefferquote %":"{:.0f}",
        }), use_container_width=True, hide_index=True, height=520)


# =================================================
# TAB 3: Zinsen / Gold-Treiber
# =================================================
with tab_rates:
    # GOLD: zeigt DXY + Realzins statt FX-Zinsen
    if symbol == "GC":
        st.subheader("Gold-Treiber: DXY + US-Realzins 10Y")

        if gold_drivers.empty:
            st.error("Gold-Treiber-Daten fehlen. Erst `python scripts/fetch_gold_drivers.py` ausführen.")
        else:
            last = gold_drivers.dropna().iloc[-1]
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("DXY", f"{last['dxy']:.2f}",
                          delta=f"{last['dxy_chg_90d']:+.2f} (90d)",
                          delta_color="inverse",
                          help="US Dollar Index. Negativer Trend = bullisch für Gold.")
            with c2:
                st.metric("Realzins 10Y", f"{last['real_yield']:+.2f}%",
                          delta=f"{last['real_yield_chg_90d']:+.2f} pp (90d)",
                          delta_color="inverse",
                          help="TIPS-implizierte 10J-Realrendite. Negativer Trend = bullisch für Gold.")
            with c3:
                # Schnell-Einordnung DXY
                d_dxy = last["dxy_chg_90d"]
                if d_dxy <= -2.0:   d_lbl, d_col = "USD schwach", "#4ADE80"
                elif d_dxy <= -0.5: d_lbl, d_col = "USD leicht schwach", "#86EFAC"
                elif d_dxy >= 2.0:  d_lbl, d_col = "USD stark", "#F87171"
                elif d_dxy >= 0.5:  d_lbl, d_col = "USD leicht stark", "#FCA5A5"
                else:               d_lbl, d_col = "USD seitwärts", "#9aa0ad"
                st.markdown(
                    f"<div style='padding-top:18px'><div style='font-size:0.72rem;"
                    f"text-transform:uppercase;letter-spacing:1px;opacity:0.75'>USD-Trend (90d)</div>"
                    f"<div style='font-size:1.4rem;font-weight:600;color:{d_col};"
                    f"font-family:ui-monospace,monospace'>{d_lbl}</div></div>",
                    unsafe_allow_html=True,
                )
            with c4:
                # Schnell-Einordnung Realzins
                d_ry = last["real_yield_chg_90d"]
                if d_ry <= -0.25:   r_lbl, r_col = "Real fällt", "#4ADE80"
                elif d_ry <= -0.10: r_lbl, r_col = "Real leicht ↓", "#86EFAC"
                elif d_ry >= 0.25:  r_lbl, r_col = "Real steigt", "#F87171"
                elif d_ry >= 0.10:  r_lbl, r_col = "Real leicht ↑", "#FCA5A5"
                else:               r_lbl, r_col = "Real seitwärts", "#9aa0ad"
                st.markdown(
                    f"<div style='padding-top:18px'><div style='font-size:0.72rem;"
                    f"text-transform:uppercase;letter-spacing:1px;opacity:0.75'>Realzins-Trend (90d)</div>"
                    f"<div style='font-size:1.4rem;font-weight:600;color:{r_col};"
                    f"font-family:ui-monospace,monospace'>{r_lbl}</div></div>",
                    unsafe_allow_html=True,
                )

            st.markdown("##### DXY-Verlauf seit 2008")
            fig_dxy = go.Figure()
            fig_dxy.add_trace(go.Scatter(
                x=gold_drivers["date"], y=gold_drivers["dxy"],
                mode="lines", name="DXY",
                line=dict(width=1.6, color="#FFB000"),
                hovertemplate="%{x|%d.%m.%Y}<br>%{y:.2f}<extra></extra>",
            ))
            fig_dxy.update_layout(
                height=320, margin=dict(l=10, r=10, t=10, b=10),
                yaxis=dict(title="DXY"),
                hovermode="x unified",
                plot_bgcolor="#0E1117", paper_bgcolor="#0E1117",
                font=dict(color="#E6E8EE"),
            )
            st.plotly_chart(fig_dxy, use_container_width=True)

            st.markdown("##### US-Realzins 10Y (TIPS) seit 2008")
            fig_ry = go.Figure()
            fig_ry.add_trace(go.Scatter(
                x=gold_drivers["date"], y=gold_drivers["real_yield"],
                mode="lines", name="Realzins 10Y",
                line=dict(width=1.6, color="#29BEFD"),
                hovertemplate="%{x|%d.%m.%Y}<br>%{y:+.2f}%<extra></extra>",
            ))
            fig_ry.add_hline(y=0, line_dash="dot", line_width=1, opacity=0.5)
            fig_ry.update_layout(
                height=320, margin=dict(l=10, r=10, t=10, b=10),
                yaxis=dict(title="Realzins 10Y (%)"),
                hovermode="x unified",
                plot_bgcolor="#0E1117", paper_bgcolor="#0E1117",
                font=dict(color="#E6E8EE"),
            )
            st.plotly_chart(fig_ry, use_container_width=True)

            st.caption(
                "**Lesart:** DXY und Realzins sind die zwei stärksten fundamentalen Treiber für Gold. "
                "Beide korrelieren historisch **negativ** mit dem Goldpreis (~−0.85 für Realzins). "
                "Fallender DXY oder fallender Realzins = bullisch für Gold."
            )

    # FX-MÄRKTE: bisheriger Zinsen-Tab
    else:
        if rates_wide.empty or rate_diffs.empty:
            st.error("Zins-Daten fehlen.")
        else:
            st.subheader("Aktuelle Leitzinsen")
            latest_rates = rates_wide.iloc[-1]
            if len(rates_wide) > 252:
                yoy = rates_wide.iloc[-1] - rates_wide.iloc[-252]
            else:
                yoy = pd.Series([np.nan]*len(latest_rates), index=latest_rates.index)
            cols = st.columns(6)
            for i, cb in enumerate(CB_ORDER):
                with cols[i]:
                    if cb in latest_rates.index and pd.notna(latest_rates[cb]):
                        st.metric(CB_LABELS[cb], f"{latest_rates[cb]:.2f}%",
                                  delta=f"{yoy[cb]:+.2f}% vs. 1J" if pd.notna(yoy[cb]) else None,
                                  delta_color="off")
                    else:
                        st.metric(CB_LABELS[cb], "n/a")

            st.subheader("Historischer Verlauf der Leitzinsen")
            palette = ["#FFB000","#29BEFD","#A259EA","#47E6C1","#F757C1","#78D64B"]
            fig_r = go.Figure()
            for i, cb in enumerate(CB_ORDER):
                if cb in rates_wide.columns:
                    fig_r.add_trace(go.Scatter(x=rates_wide.index, y=rates_wide[cb],
                        mode="lines", name=CB_LABELS[cb],
                        line=dict(width=1.6, color=palette[i % len(palette)])))
            fig_r.add_hline(y=0, line_dash="dot", line_width=1, opacity=0.4)
            fig_r.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10),
                yaxis=dict(title="Leitzins (%)"), xaxis=dict(title="Datum"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                hovermode="x unified",
                plot_bgcolor="#0E1117", paper_bgcolor="#0E1117", font=dict(color="#E6E8EE"))
            st.plotly_chart(fig_r, use_container_width=True)

            st.subheader("Aktuelle Zinsdifferenzen")
            last = rate_diffs.iloc[-1]
            rows = []
            active_pair = FUTURE_TO_PAIR.get(symbol, None)
            for fut, pair in FUTURE_TO_PAIR.items():
                if pair in rate_diffs.columns:
                    v = last[pair]
                    if pd.notna(v):
                        carry = pair[:3] if v > 0 else (pair[3:] if v < 0 else "—")
                        rows.append({"Future": fut, "FX-Paar": pair,
                                     "Zinsdifferenz": f"{v:+.2f}%", "Carry-Vorteil": carry})
            df_diffs = pd.DataFrame(rows)
            def _hl_pair(row):
                return ["background-color: rgba(255,176,0,0.15)"]*len(row) if active_pair and row["FX-Paar"]==active_pair else [""]*len(row)
            st.dataframe(df_diffs.style.apply(_hl_pair, axis=1), use_container_width=True, hide_index=True)

            if active_pair is None:
                st.info(f"Aktuell ist **{symbol} ({MARKET_NAMES[symbol]})** ausgewählt – kein FX-Future.")
            else:
                st.markdown(f"##### Verlauf der Zinsdifferenz: **{active_pair}** (passend zu {symbol})")
                fig_d = go.Figure()
                fig_d.add_trace(go.Scatter(x=rate_diffs["date"], y=rate_diffs[active_pair],
                    mode="lines", name=active_pair,
                    line=dict(width=1.8, color="#FFB000"),
                    hovertemplate="%{x|%d.%m.%Y}<br>%{y:+.2f}%<extra></extra>"))
                fig_d.add_hline(y=0, line_dash="dot", line_width=1, opacity=0.5)
                fig_d.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10),
                    yaxis=dict(title="Zinsdifferenz (%)"), xaxis=dict(title="Datum"),
                    hovermode="x unified",
                    plot_bgcolor="#0E1117", paper_bgcolor="#0E1117", font=dict(color="#E6E8EE"))
                st.plotly_chart(fig_d, use_container_width=True)


## =================================================
# TAB 4: Notizen
# =================================================
with tab_notes:
    st.subheader(f"📝 Notiz für {symbol} – {MARKET_NAMES[symbol]}")
    st.caption("Persistent gespeichert in data/processed/user_state.json. Wird nicht zu GitHub hochgeladen.")

    bias = compute_bias(symbol, cot_df, seas["monthly"], seas["weekly"], rate_diffs, gold_drivers)
    rates_part = f"{bias['rates']:+d}" if (bias["is_fx"] or bias["is_gold"]) else "—"
    st.markdown(
        f"<b>Aktueller Bias:</b> "
        f"<span class='bias-pill {bias['css']}'>{bias['label']}</span>  "
        f"<span class='muted'>(COT {bias['cot']:+d}, Saison {bias['sea']:+d}, "
        f"{bias['third_label']} {rates_part}, Total {bias['total']:+d})</span>",
        unsafe_allow_html=True,
    )

    current_note = user_state["notes"].get(symbol, "")
    new_note = st.text_area(
        "Notiz / Trade-Plan / Beobachtungen",
        value=current_note, height=240,
        placeholder="z.B. Long-Setup ab 1.0850 mit Stop unter 1.0780. COT seit 3 Wochen drehend, Saison spricht für Mai-Stärke...",
    )
    col_a, col_b = st.columns([1, 4])
    with col_a:
        if st.button("💾 Speichern", use_container_width=True):
            user_state["notes"][symbol] = new_note
            save_user_state(user_state)
            st.success("Gespeichert.")
    with col_b:
        last_saved = user_state["notes"].get(symbol, "")
        if last_saved:
            st.caption(f"Letzter gespeicherter Stand: {len(last_saved)} Zeichen.")

    if user_state["notes"]:
        st.divider()
        st.markdown("##### Alle gespeicherten Notizen")
        for sym in sorted(user_state["notes"].keys()):
            note = user_state["notes"][sym]
            if note.strip():
                with st.expander(f"{sym} – {MARKET_NAMES.get(sym, sym)}  ({len(note)} Zeichen)"):
                    st.write(note)
