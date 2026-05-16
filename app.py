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


lstm_df = load_lstm_predictions()
intervals_df = load_intervals()
tickers = sorted(lstm_df["ticker"].unique())

with st.sidebar:
    st.title("EGX Volatility Forecast")
    st.caption(
        "Pick a stock and a window to see the LSTM's 5-day forward "
        "volatility forecast against the realized series."
    )
    ticker = st.selectbox("Stock", tickers, index=tickers.index("COMI") if "COMI" in tickers else 0)

    prices_df = load_prices(ticker)
    pred_slice = lstm_df[lstm_df["ticker"] == ticker]

    overlap_min = max(prices_df["Date"].min(), pred_slice["date"].min()).date()
    overlap_max = min(prices_df["Date"].max(), pred_slice["date"].max()).date()

    date_range = st.date_input(
        "Date range",
        value=(overlap_min, overlap_max),
        min_value=overlap_min,
        max_value=overlap_max,
    )

    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date, end_date = overlap_min, overlap_max

start_ts = pd.Timestamp(start_date)
end_ts = pd.Timestamp(end_date)

vol = (
    pred_slice.merge(
        intervals_df[intervals_df["ticker"] == ticker],
        on=["ticker", "date"],
        how="left",
    )
    .loc[lambda d: (d["date"] >= start_ts) & (d["date"] <= end_ts)]
    .sort_values("date")
    .reset_index(drop=True)
)

prices = prices_df[
    (prices_df["Date"] >= start_ts) & (prices_df["Date"] <= end_ts)
].reset_index(drop=True)

st.markdown(f"### {ticker} — {start_date:%b %d, %Y} → {end_date:%b %d, %Y}")

if vol.empty:
    st.warning("No predictions available in the selected window.")
    st.stop()

rmse = float(np.sqrt(np.mean((vol["y_true"] - vol["y_pred"]) ** 2)))
mae = float(np.mean(np.abs(vol["y_true"] - vol["y_pred"])))

band_mask = vol[["lstm_lo", "lstm_hi"]].notna().all(axis=1)
if band_mask.any():
    inside = (
        (vol.loc[band_mask, "y_true"] >= vol.loc[band_mask, "lstm_lo"])
        & (vol.loc[band_mask, "y_true"] <= vol.loc[band_mask, "lstm_hi"])
    )
    coverage = float(inside.mean() * 100)
    coverage_str = f"{coverage:.1f}%"
else:
    coverage_str = "—"

c1, c2, c3, c4 = st.columns(4)
c1.metric("Days", f"{len(vol):,}")
c2.metric("RMSE", f"{rmse:.4f}")
c3.metric("MAE", f"{mae:.4f}")
c4.metric("Inside 80% band", coverage_str)

fig = make_subplots(
    rows=2,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.06,
    row_heights=[0.55, 0.45],
    subplot_titles=("Price (EGP)", "5-day Forward Volatility"),
)

fig.add_trace(
    go.Scatter(
        x=prices["Date"],
        y=prices["Price"],
        mode="lines",
        name="Price",
        line=dict(color="#1f77b4", width=2),
        hovertemplate="%{x|%b %d, %Y}<br>Price: %{y:.2f}<extra></extra>",
    ),
    row=1,
    col=1,
)

fig.add_trace(
    go.Scatter(
        x=vol["date"],
        y=vol["lstm_hi"],
        mode="lines",
        line=dict(width=0),
        name="80% band (upper)",
        showlegend=False,
        hoverinfo="skip",
    ),
    row=2,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=vol["date"],
        y=vol["lstm_lo"],
        mode="lines",
        line=dict(width=0),
        fill="tonexty",
        fillcolor="rgba(255, 127, 14, 0.18)",
        name="LSTM 80% band",
        hovertemplate="%{x|%b %d, %Y}<br>Low: %{y:.4f}<extra></extra>",
    ),
    row=2,
    col=1,
)

fig.add_trace(
    go.Scatter(
        x=vol["date"],
        y=vol["y_true"],
        mode="lines",
        name="Actual vol",
        line=dict(color="#2ca02c", width=1.6, dash="dot"),
        hovertemplate="%{x|%b %d, %Y}<br>Actual: %{y:.4f}<extra></extra>",
    ),
    row=2,
    col=1,
)

fig.add_trace(
    go.Scatter(
        x=vol["date"],
        y=vol["y_pred"],
        mode="lines",
        name="LSTM predicted",
        line=dict(color="#ff7f0e", width=2.2),
        hovertemplate="%{x|%b %d, %Y}<br>Predicted: %{y:.4f}<extra></extra>",
    ),
    row=2,
    col=1,
)

fig.update_layout(
    height=720,
    template="plotly_white",
    hovermode="x unified",
    margin=dict(l=10, r=10, t=50, b=10),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.04,
        xanchor="right",
        x=1,
        bgcolor="rgba(0,0,0,0)",
    ),
)
fig.update_yaxes(title_text="Price (EGP)", row=1, col=1)
fig.update_yaxes(title_text="5-day fwd σ", row=2, col=1, tickformat=".3f")
fig.update_xaxes(showgrid=True, row=2, col=1)

st.plotly_chart(fig, use_container_width=True)

st.caption(
    "Test window: 2024-01-02 → 2026-04-30. Shaded ribbon is the LSTM 80% "
    "confidence cone (z = 1.2816 × residual σ). Volatility is annualized-free "
    "realized standard deviation of daily log returns over the next 5 trading days."
)
