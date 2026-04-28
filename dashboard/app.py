"""
Trading Dashboard - Phase 5.1/5.2
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

st.set_page_config(
    page_title="Trading Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

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
    margin-top: 8px; font-size: 0.82rem;
}
.cockpit-row .lbl {
    color: #9aa0ad; text-transform: uppercase;
    letter-spacing: 0.8px; font-size: 0.7rem;
}
.cockpit-row .val {
    font-family: ui-monospace, monospace; font-weight: 600;
}
.cot-low  { color: #4ADE80; }
.cot-high { color: #F87171; }
.cot-mid  { color: #E6E8EE; }
[data-testid="stDataFrame"] {
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 0.86rem;
}
.dashboard-header-meta {
    color: #9aa0ad; font-size: 0.78rem;
    letter-spacing: 0.5px; margin-top: -4px;
}
</style>
""", unsafe_allow_html=True)

COT_PATH    = Path("data/processed/cot_metrics.parquet")
SEAS_MONTH  = Path("data/processed/seasonality_monthly.parquet")
SEAS_WEEK   = Path("data/processed/seasonality_weekly.parquet")
SEAS_CURVE  = Path("data/processed/seasonality_curve.parquet")
RATES_WIDE  = Path("data/processed/rates_wide.parquet")
RATE_DIFFS  = Path("data/processed/rate_diffs.parquet")

GROUP_CHOICES = {
    "Commercials":       "commercials",
    "Large Speculators": "large_specs",
    "Small Speculators": "small_specs",
}
MARKET_NAMES = {
    "6E": "Euro FX", "6B": "British Pound", "6J": "Japanese Yen",
    "6A": "Australian Dollar", "6C": "Canadian Dollar",
    "GC": "Gold", "SI": "Silver", "CL": "WTI Crude Oil",
    "ES": "E-mini S&P 500", "NQ": "E-mini Nasdaq-100",
}
FUTURE_TO_PAIR = {
    "6E": "EURUSD", "6B": "GBPUSD", "6J": "USDJPY",
    "6A": "AUDUSD", "6C": "USDCAD",
}
CB_ORDER  = ["FED", "ECB", "BOE", "BOJ", "RBA", "BOC"]
CB_LABELS = {"FED":"FED (USD)","ECB":"EZB (EUR)","BOE":"BoE (GBP)",
             "BOJ":"BoJ (JPY)","RBA":"RBA (AUD)","BOC":"BoC (CAD)"}
MONTH_NAMES_DE = ["Jan","Feb","Mär","Apr","Mai","Jun",
                  "Jul","Aug","Sep","Okt","Nov","Dez"]


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


st.markdown("## 📊 Trading Dashboard")

cot_df = load_cot()
seas   = load_seasonality()
rates_wide, rate_diffs = load_rates()

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

with st.sidebar:
    st.markdown("### Auswahl")
    symbol = st.selectbox(
        "Markt", options=list(MARKET_NAMES.keys()),
        format_func=lambda s: f"{s} – {MARKET_NAMES[s]}", index=5,
    )
    st.divider()
    st.markdown("### COT-Optionen")
    group_label = st.radio("Trader-Gruppe",
                           options=list(GROUP_CHOICES.keys()), index=0)
    group_key = GROUP_CHOICES[group_label]


def cockpit_card_html(sym, cot_df, seas_monthly):
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
    return (
        f"<div class='cockpit-card'>"
        f"<div class='cockpit-symbol'>{sym}</div>"
        f"<div class='cockpit-name'>{MARKET_NAMES[sym]}</div>"
        f"<div class='cockpit-row'><span class='lbl'>COT-Idx 26W (Comm)</span>"
        f"<span class='val'>{cot_str}</span></div>"
        f"<div class='cockpit-row'><span class='lbl'>Net-Trend</span>"
        f"<span class='val' style='color:{arrow_color}'>{arrow}</span></div>"
        f"<div class='cockpit-row'><span class='lbl'>Ø {MONTH_NAMES_DE[datetime.now().month-1]}</span>"
        f"<span class='val'>{seas_str}</span></div>"
        f"</div>"
    )


tab_over, tab_cot, tab_seas, tab_rates = st.tabs(
    ["🧭 Übersicht", "📈 COT-Daten", "🗓️ Saisonalität", "💰 Zinsen"]
)


with tab_over:
    st.caption("Schneller Überblick über alle 10 Märkte. Farbcode COT-Index 26W (Commercials):  🟢 ≤ 20  ·  🔴 ≥ 80  ·  weiß = neutral.")
    syms = list(MARKET_NAMES.keys())
    cols = st.columns(5)
    for i, sym in enumerate(syms):
        with cols[i % 5]:
            st.markdown(cockpit_card_html(sym, cot_df, seas["monthly"]), unsafe_allow_html=True)
    st.divider()
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

        cur_avg = stats_m.loc[stats_m["Monat"]==cur_month_label, "Ø Return %"].iloc[0]
        cur_hit = stats_m.loc[stats_m["Monat"]==cur_month_label, "Trefferquote %"].iloc[0]
        if cur_avg > 0:
            st.success(f"**{cur_month_label}** ist für {symbol} historisch **positiv** (Ø {cur_avg:+.2f}%, Trefferquote {cur_hit:.0f}%).")
        else:
            st.warning(f"**{cur_month_label}** ist für {symbol} historisch **negativ** (Ø {cur_avg:+.2f}%, Trefferquote {cur_hit:.0f}%).")


with tab_rates:
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
            st.info(f"Aktuell ist **{symbol} ({MARKET_NAMES[symbol]})** ausgewählt – kein FX-Future. Wähle 6E/6B/6J/6A/6C in der Sidebar für den passenden Verlauf.")
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
            v = last[active_pair]
            if pd.notna(v):
                if v > 0.5:
                    st.success(f"**{active_pair}** Zinsdifferenz **{v:+.2f}%** → deutlicher Carry-Vorteil zugunsten **{active_pair[:3]}**.")
                elif v < -0.5:
                    st.error(f"**{active_pair}** Zinsdifferenz **{v:+.2f}%** → deutlicher Carry-Vorteil zugunsten **{active_pair[3:]}**.")
                else:
                    st.info(f"**{active_pair}** Zinsdifferenz **{v:+.2f}%** → neutraler Bereich.")