from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg

NUM_SYMBOLS = 100
FITTING_WINDOW = 252
K_STEP_RETURNS = 1
NUMBER_POSITIONS = 2
K_RANK = NUMBER_POSITIONS//2
REG_MODEL = AutoReg
P_LAG = [1]

@dataclass
class State:
    symbols: list[str]
    lagged_prices: np.ndarray
    lagged_returns: np.ndarray
    wealth: float
    positions: np.ndarray


def simple_returns(df: pd.DataFrame, 
                   k_step: int=1, 
                   forward: bool=False) -> pd.DataFrame:
    if forward:
        df_returns = df.shift(k_step)/df - 1
    else:
        df_returns = df/df.shift(k_step) - 1
    df_returns = df_returns.dropna()
    return df_returns

def initialise_state(data: pd.DataFrame) -> State:
    df = data.copy()
    df['date'] = pd.to_datetime(df['date']).dt.date

    dates = np.sort(df['date'].unique())
    symbols = sorted(df['symbol'].unique().tolist())

    if len(symbols) != NUM_SYMBOLS:
        # Not strictly required, but keeps behavior aligned with the R template
        raise ValueError(f"Expected {NUM_SYMBOLS} symbols, got {len(symbols)}")
    
    # Pivot to wide format for return calculation
    df_wide = (
        df.pivot(index='date', 
                 values='close', 
                 columns='symbol')
         .sort_index() # Safety check to make sure the data is in order
    )

    # Keep the most recent dates for fitting
    n = df_wide.shape[0]
    df_prices_recent = df_wide.iloc[(n-FITTING_WINDOW-1): n, :][symbols].copy() # Reorder columns to match symbols order

    # Calculate returns
    df_simple_returns = (
        simple_returns(df_prices_recent, 
                       K_STEP_RETURNS, 
                       forward=False)
    )


    # Construct lagged prices - note that it is backward looking, most recent date is the last entry
    lagged_prices = np.array(df_prices_recent)

    # Construct lagged returns
    lagged_returns = np.array(df_simple_returns)

    # Initiate positions
    positions = np.zeros(NUM_SYMBOLS, dtype=float)

    return State(symbols=symbols, 
                 lagged_prices=lagged_prices, 
                 lagged_returns=lagged_returns, 
                 wealth=1.0, 
                 positions=positions)


def ar_forecast(data: np.ndarray, 
                reg_model, 
                k_step: int, 
                **kwargs) -> float:
    
    # Fit ar_model
    ar_model = reg_model(data, **kwargs).fit()

    # Forecast k-step ahead
    forecasts = ar_model.forecast(k_step)

    # Return the desired forecasted value
    return forecasts[k_step-1]

def k_ranks(forecasts: np.ndarray, 
            k: int) -> Tuple[np.ndarray, np.ndarray]:
    
    n = len(forecasts)
    indx_order = np.argsort(forecasts)
    # Find the indices corresponding to the k-smalles values, k-largest values
    k_smallest, k_largest = indx_order[:k], indx_order[(n-k):]
    return (k_smallest, k_largest)


def ar_trading_algorithm(new_data: pd.DataFrame, 
                         state: State) -> Tuple[np.ndarray, State]:
    
    # Update lagged price matrix - last entries reflect most recent prices
    state.lagged_prices[:FITTING_WINDOW, :] = state.lagged_prices[1:(FITTING_WINDOW+1), :]

    # Update return matrix
    state.lagged_returns[:(FITTING_WINDOW-1), :] = state.lagged_returns[1: FITTING_WINDOW, :]

    # Enforce symbols to be in correct order
    new_data = new_data.copy()
    new_data["symbol"] = new_data["symbol"].astype(str)
    sym_rank = {s: i for i, s in enumerate(state.symbols)}
    new_data["_r"] = new_data["symbol"].map(sym_rank)
    new_data = new_data.sort_values("_r").drop(columns=["_r"]).reset_index(drop=True)

    closes_today = new_data["close"].to_numpy(dtype=float)
    if closes_today.shape[0] != NUM_SYMBOLS:
        raise ValueError(f"Expected {NUM_SYMBOLS} rows in new_data, got {closes_today.shape[0]}")
    
    # Update the prices to contain the most recent close
    state.lagged_prices[FITTING_WINDOW, :] = closes_today

    # Update wealth and positions to reflect 1-day PnL (same as R example)
    r1d = state.lagged_prices[-1, :] / state.lagged_prices[-2, :] - 1.0
    #r1d = np.where(np.isfinite(r1d), r1d, 0.0)  # safety if any NA slipped through


    state.wealth = float(state.wealth + np.sum(state.positions * r1d))
    state.positions = state.positions * (1.0 + r1d)
    
    # Insert today's returns at the last entry in lagged_returns
    state.lagged_returns[FITTING_WINDOW-1, :] = r1d

    # Find AR forecasts
    return_forecasts = np.zeros(NUM_SYMBOLS, dtype=float)
    for i in range(NUM_SYMBOLS):
        curr_data = state.lagged_returns[:, i]
        # Find 1 day forecast returns
        return_forecasts[i] += ar_forecast(curr_data,
                                           REG_MODEL,
                                           K_STEP_RETURNS,
                                           lags=P_LAG)

    # Find the indices corresponding to the k-th smallest, k-th largest forecasts
    k_smallest, k_largest = k_ranks(return_forecasts, K_RANK)

    # Initiate empty positions
    new_positions = np.zeros(NUM_SYMBOLS, dtype=float)
    # Short the instruments with the lowest forecasts
    new_positions[k_smallest] -= (state.wealth/NUMBER_POSITIONS)
    # Long the largest k forecasts
    new_positions[k_largest] += (state.wealth/NUMBER_POSITIONS)

    # Trades are adjustment from current positions
    trades = new_positions - state.positions

    # Update state positions
    state.positions = new_positions

    return trades.astype(float), state


        








   




    



