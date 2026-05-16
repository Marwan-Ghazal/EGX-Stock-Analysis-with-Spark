from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

ROOT = Path(__file__).parent
LSTM_PRED_CSV = ROOT / "output" / "lstm_test_predictions.csv"
INTERVALS_CSV = ROOT / "output" / "table_predictions_with_intervals.csv"
PRICES_DIR = ROOT / "egx_data"

Z80 = 1.2816  # 80% two-sided z-score, matches models.ipynb

st.set_page_config(
    page_title="EGX Volatility Forecast",
    page_icon="📈",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def load_lstm_predictions() -> pd.DataFrame:
    df = pd.read_csv(LSTM_PRED_CSV)
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(show_spinner=False)
def load_intervals() -> pd.DataFrame:
    df = pd.read_csv(INTERVALS_CSV, usecols=["ticker", "date", "lstm_lo", "lstm_hi"])
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data(show_spinner=False)
def load_prices(ticker: str) -> pd.DataFrame:
    path = PRICES_DIR / f"{ticker}.csv"
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")
    df["Price"] = (
        df["Price"].astype(str).str.replace(",", "", regex=False).astype(float)
    )
    return df[["Date", "Price"]].sort_values("Date").reset_index(drop=True)


def project_cone(p0: float, sigma: float, n_days: int, z: float = Z80):
    k = np.arange(1, n_days + 1)
    factor = z * sigma * np.sqrt(k)
    return p0 * np.exp(-factor), p0 * np.exp(+factor)


lstm_df = load_lstm_predictions()
intervals_df = load_intervals()
tickers = sorted(lstm_df["ticker"].unique())

with st.sidebar:
    st.title("EGX Volatility Forecast")
    st.caption(
        "Pick a stock and an anchor date. The cone projects the LSTM's "
        "predicted volatility forward as an 80% price band."
    )

    ticker = st.selectbox(
        "Stock",
        tickers,
        index=tickers.index("COMI") if "COMI" in tickers else 0,
    )

    prices_df = load_prices(ticker)
    pred_slice = lstm_df[lstm_df["ticker"] == ticker].sort_values("date").reset_index(drop=True)

    valid_anchor_dates = sorted(
        set(prices_df["Date"].dt.date) & set(pred_slice["date"].dt.date)
    )
    if not valid_anchor_dates:
        st.error("No overlapping dates between prices and predictions for this ticker.")
        st.stop()

    anchor_date = st.date_input(
        "Forecast anchor",
        value=valid_anchor_dates[-1],
        min_value=valid_anchor_dates[0],
        max_value=valid_anchor_dates[-1],
    )
    if anchor_date not in valid_anchor_dates:
        anchor_date = max(d for d in valid_anchor_dates if d <= anchor_date)

    history_days = st.slider(
        "History to show (calendar days)", min_value=14, max_value=180, value=45, step=1
    )
    forecast_days = st.slider(
        "Forecast horizon (trading days)", min_value=1, max_value=20, value=5, step=1
    )

anchor_ts = pd.Timestamp(anchor_date)
anchor_row = prices_df[prices_df["Date"] == anchor_ts].iloc[0]
anchor_price = float(anchor_row["Price"])

pred_row = pred_slice[pred_slice["date"] == anchor_ts].iloc[0]
sigma_lstm = float(pred_row["y_pred"])

future_trading = (
    prices_df[prices_df["Date"] > anchor_ts]
    .head(forecast_days)
    .reset_index(drop=True)
)
n_proj = len(future_trading)

if n_proj == 0:
    cone_dates = [anchor_ts]
    cone_lo_full = np.array([anchor_price])
    cone_hi_full = np.array([anchor_price])
    actual_inside = 0
else:
    cone_lo, cone_hi = project_cone(anchor_price, sigma_lstm, n_proj)
    cone_dates = [anchor_ts] + list(future_trading["Date"])
    cone_lo_full = np.concatenate([[anchor_price], cone_lo])
    cone_hi_full = np.concatenate([[anchor_price], cone_hi])
    actual_prices = future_trading["Price"].to_numpy()
    actual_inside = int(((actual_prices >= cone_lo) & (actual_prices <= cone_hi)).sum())

history_start = anchor_ts - pd.Timedelta(days=history_days)
view_end = cone_dates[-1] if cone_dates else anchor_ts

prices_view = prices_df[
    (prices_df["Date"] >= history_start) & (prices_df["Date"] <= view_end)
].reset_index(drop=True)

vol_view = (
    pred_slice.merge(
        intervals_df[intervals_df["ticker"] == ticker],
        on=["ticker", "date"],
        how="left",
    )
    .loc[lambda d: (d["date"] >= history_start) & (d["date"] <= view_end)]
    .sort_values("date")
    .reset_index(drop=True)
)

st.markdown(
    f"### {ticker} — anchored {anchor_ts:%b %d, %Y}  "
    f"<span style='color:#6b7280;font-weight:400;font-size:0.85em;'>"
    f"σ̂ = {sigma_lstm:.4f} · cone covers {n_proj} trading day(s) forward"
    f"</span>",
    unsafe_allow_html=True,
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Anchor price", f"{anchor_price:.2f} EGP")
c2.metric("Predicted σ (5d fwd)", f"{sigma_lstm:.4f}")
if n_proj > 0:
    cone_pct_hi = (cone_hi_full[-1] / anchor_price - 1) * 100
    c3.metric(
        f"Cone @ day {n_proj}",
        f"±{cone_pct_hi:.2f}%",
        help="Half-width of the 80% price band at the last forecast day, in % of anchor price.",
    )
    c4.metric(
        "Inside cone",
        f"{actual_inside}/{n_proj}",
        help="Actual prices that fell inside the 80% cone over the forecast horizon.",
    )
else:
    c3.metric(f"Cone @ day {forecast_days}", "—")
    c4.metric("Inside cone", "—")

fig = make_subplots(
    rows=2,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.07,
    row_heights=[0.6, 0.4],
    subplot_titles=("Price with LSTM 80% cone", "5-day Forward Volatility"),
)

if n_proj > 0:
    fig.add_vrect(
        x0=anchor_ts,
        x1=cone_dates[-1],
        fillcolor="rgba(250, 204, 21, 0.10)",
        line_width=0,
        row=1,
        col=1,
    )

fig.add_trace(
    go.Scatter(
        x=prices_view["Date"],
        y=prices_view["Price"],
        mode="lines",
        name="Price",
        line=dict(color="#111827", width=1.9),
        hovertemplate="%{x|%b %d, %Y}<br>Price: %{y:.2f} EGP<extra></extra>",
    ),
    row=1,
    col=1,
)

fig.add_trace(
    go.Scatter(
        x=cone_dates,
        y=cone_hi_full,
        mode="lines",
        line=dict(width=0),
        name="cone-hi",
        showlegend=False,
        hoverinfo="skip",
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=cone_dates,
        y=cone_lo_full,
        mode="lines",
        line=dict(width=0),
        fill="tonexty",
        fillcolor="rgba(153, 53, 86, 0.22)",
        name="LSTM 80% cone",
        hovertemplate="%{x|%b %d, %Y}<br>Lower: %{y:.2f} EGP<extra></extra>",
    ),
    row=1,
    col=1,
)

fig.add_trace(
    go.Scatter(
        x=[anchor_ts, anchor_ts],
        y=[prices_view["Price"].min(), prices_view["Price"].max()],
        mode="lines",
        line=dict(color="#6b7280", width=1, dash="dash"),
        name="Anchor",
        hoverinfo="skip",
        showlegend=False,
    ),
    row=1,
    col=1,
)

if not vol_view.empty:
    fig.add_trace(
        go.Scatter(
            x=vol_view["date"],
            y=vol_view["lstm_hi"],
            mode="lines",
            line=dict(width=0),
            name="vol-hi",
            showlegend=False,
            hoverinfo="skip",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=vol_view["date"],
            y=vol_view["lstm_lo"],
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(255, 127, 14, 0.18)",
            name="LSTM 80% vol band",
            hovertemplate="%{x|%b %d, %Y}<br>Low: %{y:.4f}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=vol_view["date"],
            y=vol_view["y_true"],
            mode="lines",
            name="Actual vol",
            line=dict(color="#2ca02c", width=1.4, dash="dot"),
            hovertemplate="%{x|%b %d, %Y}<br>Actual: %{y:.4f}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=vol_view["date"],
            y=vol_view["y_pred"],
            mode="lines",
            name="LSTM predicted",
            line=dict(color="#ff7f0e", width=2.0),
            hovertemplate="%{x|%b %d, %Y}<br>Predicted: %{y:.4f}<extra></extra>",
        ),
        row=2,
        col=1,
    )

fig.update_layout(
    height=720,
    template="plotly_white",
    hovermode="x unified",
    plot_bgcolor="#ffffff",
    paper_bgcolor="#ffffff",
    margin=dict(l=10, r=10, t=60, b=10),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.06,
        xanchor="right",
        x=1,
        bgcolor="rgba(0,0,0,0)",
    ),
)
fig.update_yaxes(title_text="Price (EGP)", row=1, col=1)
fig.update_yaxes(title_text="5-day fwd σ", row=2, col=1, tickformat=".3f")
fig.update_xaxes(showgrid=True, gridcolor="#eef0f3")
fig.update_yaxes(showgrid=True, gridcolor="#eef0f3")

st.plotly_chart(fig, use_container_width=True)

st.caption(
    "Cone formula: anchor × exp(±1.2816 · σ̂ · √k) for k = 1..N trading days, where "
    "σ̂ is the LSTM's predicted 5-day forward volatility on the anchor date. Test "
    "window: 2024-01-02 → 2026-04-30."
)
