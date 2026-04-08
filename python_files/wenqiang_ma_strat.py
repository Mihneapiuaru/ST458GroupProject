
"""
Exploratory analysis of the training data (2010–2013) tested six signals for
predictive power using the Spearman Information Coefficient (IC) — the rank
correlation between today's signal and tomorrow's cross-sectional stock return.

The 60-day momentum signal stood out as the strongest and most robust predictor,
but in the *opposite* direction to what classic momentum theory would suggest:

    Mean IC  = -0.037    (negative: past 60-day winners tend to underperform)
    t-stat   = -7.2      (highly significant; |t| > 2 is the usual threshold)
    IC IR    = -3.74     (annualised information ratio of the IC time series)

This means the market exhibits strong **mean reversion** at the 60-day horizon:
stocks that rallied over the past quarter systematically give back returns, and
vice versa for stocks that fell.

## Signal

For each stock i on day t:

    signal_i,t = -sign( close_{i,t} - close_{i,t-60} )

    +1  if the stock's price is *lower* than 60 trading days ago  → BUY  (past loser)
    -1  if the stock's price is *higher* than 60 trading days ago → SELL (past winner)
     0  if the price is unchanged (rare)

The sign transformation deliberately discards the *magnitude* of the 60-day
move and trades only on direction. This reduces sensitivity to outliers
(e.g. a single stock that moved 200%), keeps turnover low, and produces a
stable, binary rebalancing rule.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd


NUM_SYMBOLS = 100
LOOKBACK = 60  # 60-day mean-reversion window


@dataclass
class State:
    symbols: list[str]
    price_buffer: np.ndarray  # shape (LOOKBACK, NUM_SYMBOLS), row 0 = most recent
    wealth: float
    positions: np.ndarray     # current dollar allocations, shape (NUM_SYMBOLS,)


def initialise_state(data: pd.DataFrame) -> State:
    df = data.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date

    dates = np.sort(df["date"].unique())
    symbols = sorted(df["symbol"].unique().tolist())

    sym_to_idx = {s: i for i, s in enumerate(symbols)}
    price_buffer = np.full((LOOKBACK, NUM_SYMBOLS), np.nan, dtype=float)

    # Fill buffer with the last LOOKBACK closes from training (newest at row 0)
    for i in range(min(LOOKBACK, len(dates))):
        dt = dates[len(dates) - 1 - i]
        sub = df.loc[df["date"] == dt, ["symbol", "close"]]
        for _, row in sub.iterrows():
            price_buffer[i, sym_to_idx[row["symbol"]]] = float(row["close"])

    return State(
        symbols=symbols,
        price_buffer=price_buffer,
        wealth=1.0,
        positions=np.zeros(NUM_SYMBOLS, dtype=float),
    )


def trading_algorithm(new_data: pd.DataFrame, state: State) -> Tuple[np.ndarray, State]:
    state.price_buffer[1:, :] = state.price_buffer[:-1, :]

    new_data = new_data.copy()
    new_data["symbol"] = new_data["symbol"].astype(str)
    sym_rank = {s: i for i, s in enumerate(state.symbols)}
    new_data["_r"] = new_data["symbol"].map(sym_rank)
    new_data = new_data.sort_values("_r").drop(columns=["_r"]).reset_index(drop=True)

    closes_today = new_data["close"].to_numpy(dtype=float)
    state.price_buffer[0, :] = closes_today

    r1d = state.price_buffer[0, :] / state.price_buffer[1, :] - 1.0
    r1d = np.where(np.isfinite(r1d), r1d, 0.0)
    state.wealth = float(state.wealth + np.sum(state.positions * r1d))
    state.positions = state.positions * (1.0 + r1d)

    price_60d_ago = state.price_buffer[LOOKBACK - 1, :]
    signal = -np.sign(closes_today - price_60d_ago)
    signal = np.where(np.isfinite(signal), signal, 0.0)

    n_active = int(np.sum(signal != 0))
    if n_active == 0 or state.wealth <= 0:
        new_positions = np.zeros(NUM_SYMBOLS, dtype=float)
    else:
        new_positions = (state.wealth / n_active) * signal

    trades = new_positions - state.positions
    state.positions = new_positions

    return trades.astype(float), state
