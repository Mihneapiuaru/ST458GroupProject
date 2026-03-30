# Standard library imports
import datetime as dt

# Third party imports
import pandas as pd
import numpy as np

def simple_returns(df: pd.DataFrame, 
                   k_step: int=1, 
                   forward: bool=False) -> pd.DataFrame:
    if forward:
        df_returns = df.shift(k_step)/df - 1
    else:
        df_returns = df/df.shift(k_step) - 1
    df_returns = df_returns.dropna()
    return df_returns

def log_returns(df: pd.DataFrame,
                k_step: int=1,
                forward: bool=False) -> pd.DataFrame:
    if forward:
        df_returns = np.log(df.shift(k_step)) - np.log(df)
    else:
        df_returns = np.log(df) - np.log(df.shift(k_step))
    df_returns = df_returns.dropna()
    return df_returns

def directional_positions(df: pd.DataFrame,
                          col_position: str,
                          num_instruments: int=100) -> pd.DataFrame:
    df_positions = df.copy()
    df_positions['weight'] = np.where(df_positions[col_position] > 0, 1, -1)
    df_positions['weight'] = df_positions['weight']/num_instruments
    return df_positions

def cross_sectional_positions(df: pd.DataFrame,
                              col_position: str,
                              number_positions: int=10) -> pd.DataFrame:
    df_positions = df.copy()
    # Rank over the date
    df_positions[f'{col_position}_rank'] = (
        df_positions
        .groupby(level='date')[col_position]
        .rank(method='max')
    )
    # Long-short portfolio
    max_rank = df_positions[f'{col_position}_rank'].max()
    nb_short = number_positions/2
    nb_long = number_positions - nb_short
    rank_long = max_rank - nb_long + 1
    # Entry conditions
    conditions = [
        df_positions[f'{col_position}_rank'] <= nb_short,
        df_positions[f'{col_position}_rank'] >= rank_long
    ]
    choices = [-1, 1]
    df_positions['weight'] = np.select(conditions, choices, default=0)
    df_positions['weight'] = df_positions['weight']/number_positions

    return df_positions
    

def position_returns(df: pd.DataFrame,
                    weight: str,
                    returns: str) -> pd.DataFrame:
    df_returns = df.copy()
    df_returns['positions_return'] = df_returns[weight] * df_returns[returns]
    return df_returns

def date_returns(df: pd.DataFrame,
                 pos_return: str) -> pd.DataFrame:
    
    df_date_returns = (
        df[[pos_return]] # Pass as list to maintain dataframe structure instead of series
        .copy()
        .groupby(level='date')
        .sum()
    )
    return df_date_returns

def calculate_sr(df: pd.DataFrame, 
                 return_col: str,
                 rf: float=0.0) -> float:
    returns_arr = np.array(df[return_col])
    avg_returns = np.mean(returns_arr)
    vol_returns = np.std(returns_arr)
    sr = (avg_returns-rf)/vol_returns
    return sr