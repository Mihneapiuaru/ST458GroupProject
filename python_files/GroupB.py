#!/usr/bin/env python3
"""
GroupB.py: AR(1) Trading Algorithm

Strategy Description:
- Save the close prices for the last W + 1 days (lagged_prices)
- Compute 1 day close-close returns for the last W days (lagged_returns)
- Fit AR(1) model on returns for each symbol using last W days and generate one day ahead forecasts
- Go long the K largest forecasted returns and the K lowest forecasted returns
- Target position = (wealth/num_symbols) * sign (sign is 1 for K largest, -1 for K lowest)
- Trades = target position - current position


This matches walk_forward.py’s interface:
  initialise_state(df_train) -> state
  trading_algorithm(new_data, state) -> (trades, new_state)

Trades are in "wealth units" (dollar allocation).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg

NUM_SYMBOLS = 100
FITTING_WINDOW = 126
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


def simple_returns(
        df: pd.DataFrame, 
        k_step: int=1, 
        forward: bool=False
 ) -> pd.DataFrame:
    """Calculates simple returns from a wide dataframe.

    Args:
        df (pd.DataFrame): Dataframe with symbols as columns.
        k_step (int, optional): The horizon of returns, i.e. which lag or lead are we taking returns on. Defaults to 1 (previous day return).
        forward (bool, optional): Whether we are taking forward returns or backward returns. Defauls to False.

    Returns:
        pd.DataFrame: Dataframe with symbols as column, and the entries are the (close-close) returns on each date.
    """
    if forward:
        df_returns = df.shift(k_step)/df - 1 # If taking future returns, we divide the future value by the current value
    else:
        df_returns = df/df.shift(k_step) - 1
    df_returns = df_returns.dropna() # Missing values were introduced by differencing
    return df_returns

def initialise_state(data: pd.DataFrame) -> State:
    """Initiliases the state from the training data, 
    which keeps track of the variables we need for the trading algorithm.

    Args:
        data (pd.DataFrame): training data, in ohlcv format, with a symbol column and a date column.

    Raises:
        ValueError: Error if not all symbols are present on a trading day.

    Returns:
        State: The current state with the variables required to initiate the trading algorithm.
    """
    df = data.copy()

    # Convert to datetime and extract the date
    df['date'] = pd.to_datetime(df['date']).dt.date

    # Sort symbols to maintain order
    symbols = sorted(df['symbol'].unique().tolist())

    if len(symbols) != NUM_SYMBOLS:
        # Not strictly required, but keeps behavior aligned with the R template
        raise ValueError(f"Expected {NUM_SYMBOLS} symbols, got {len(symbols)}")
    
    # Pivot to wide format for return calculation
    df_wide = (
        df.pivot(
            index='date', 
            values='close', 
            columns='symbol'
        )
         .sort_index() # Safety check to make sure the data is in order
    )

    # Keep the most recent dates for fitting
    n = df_wide.shape[0]
    df_prices_recent = df_wide.iloc[(n-FITTING_WINDOW-1):, :][symbols].copy() # Reorder columns to match symbols order

    # Calculate lagged returns
    df_simple_returns = (
        simple_returns(
            df_prices_recent,  
            K_STEP_RETURNS, 
            forward=False
        )# Returns are lagged, so forward=False
    )

    # Construct lagged prices, most recent date at the last entry
    lagged_prices = np.array(df_prices_recent)

    # Construct lagged returns, most recent date at the last entry
    lagged_returns = np.array(df_simple_returns)

    # Initiate positions
    positions = np.zeros(NUM_SYMBOLS, dtype=float)

    return State(symbols=symbols, 
                 lagged_prices=lagged_prices, 
                 lagged_returns=lagged_returns, 
                 wealth=1.0, 
                 positions=positions)


def ar_forecast(
        data: np.ndarray, 
        reg_model, 
        k_step: int, 
        **kwargs
 ) -> float:
    """Produces univariate autoregressive modelling forecasts. Works with any univariate model satisfying statsmodels API.

    Args:
        data (np.ndarray): Series we forecast.
        reg_model (_type_): The autoregressive model we want to use, e.g. AutoReg or SARIMAX.
        k_step (int): Forecast horizon.

    Returns:
        float: Forecasted value.
    """
    
    # Fit ar_model
    ar_model = reg_model(data, **kwargs).fit()

    # Forecast k-step ahead
    forecasts = ar_model.forecast(k_step)

    # Return the desired forecasted value
    return forecasts[k_step-1]

    
def k_ranks(arr: np.ndarray, k: int) -> \
        Tuple[np.ndarray, np.ndarray]:
    """Returns the indices of the k-smallest entries and k-largest entries of an array .

    Args:
        arr (np.ndarray): Array from which we want to identify the indices of the k-smallest values and k-largest values.
        k (int): Rank number.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Arrays with indices of k-smallest values and k-largest values.
    """
    
    n = len(arr)
    # np.argsort returns the indices of the entries in sorted order
    indx_order = np.argsort(arr)
    # Find the indices corresponding to the k-smalles values, k-largest values
    k_smallest, k_largest = indx_order[:k], indx_order[(n-k):]
    return (k_smallest, k_largest)


def trading_algorithm(new_data: pd.DataFrame, state: State) -> \
        Tuple[np.ndarray, State]:
    """Implements trading algorithm based on AutoRegressive forecasts.

    Args:
        new_data (pd.DataFrame): New data with close prices.
        state (State): Current state.

    Raises:
        ValueError: If not all symbols are present.

    Returns:
        Tuple[np.ndarray, State]: Trades, update state.
    """
    
    # Shift lagged prices one date down
    state.lagged_prices[:FITTING_WINDOW, :] = state.lagged_prices[1:(FITTING_WINDOW+1), :]

    # Shift lagged returns one day down
    state.lagged_returns[:(FITTING_WINDOW-1), :] = state.lagged_returns[1:FITTING_WINDOW, :]

    new_data = new_data.copy()
    new_symbols = new_data["symbol"].unique().tolist()
    if state.symbols != new_symbols: # Checks whether the symbols are in sorted order
        # Enforce symbols to be in correct order
        new_data["symbol"] = new_data["symbol"].astype(str)
        sym_rank = {s: i for i, s in enumerate(state.symbols)}
        new_data["_r"] = new_data["symbol"].map(sym_rank)
        new_data = new_data.sort_values("_r").drop(columns=["_r"]).reset_index(drop=True)

    # Extract closes in current order
    closes_today = new_data["close"].to_numpy(dtype=float)

    # Check all symbols are present
    if closes_today.shape[0] != NUM_SYMBOLS:
        raise ValueError(f"Expected {NUM_SYMBOLS} rows in new_data, got {closes_today.shape[0]}")
    
    # Update the prices to contain the most recent close
    state.lagged_prices[FITTING_WINDOW, :] = closes_today

    # Update wealth and positions to reflect 1-day PnL
    r1d = state.lagged_prices[-1, :] / state.lagged_prices[-2, :] - 1.0
    state.wealth = float(state.wealth + np.sum(state.positions * r1d))
    state.positions = state.positions * (1.0 + r1d)
    
    # Insert today's returns at the last entry in lagged_returns
    state.lagged_returns[FITTING_WINDOW-1, :] = r1d

    # Initiate forecasts to 0's
    return_forecasts = np.zeros(NUM_SYMBOLS, dtype=float)
    for i in range(NUM_SYMBOLS):
        # Current symbol FITTING_WINDOW lagged returns
        curr_data = state.lagged_returns[:, i]
        # Find k-step forecast returns for the current symbol
        return_forecasts[i] += ar_forecast(curr_data,
                                           REG_MODEL,
                                           K_STEP_RETURNS,
                                           lags=P_LAG)


    # Find the indices corresponding to the k-th smallest, k-th largest forecasts
    k_smallest, k_largest = k_ranks(return_forecasts, K_RANK)

    # Initiate 0's positions
    new_positions = np.zeros(NUM_SYMBOLS, dtype=float)
    # Short the instruments with the lowest forecasts
    new_positions[k_smallest] -= (state.wealth/NUMBER_POSITIONS)
    # Long the instruments with the largest forecasts
    new_positions[k_largest] += (state.wealth/NUMBER_POSITIONS)

    # Trades are adjustment from current positions
    trades = new_positions - state.positions

    # Update state positions
    state.positions = new_positions

    return trades.astype(float), state


        








   




    



