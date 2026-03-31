# Standard library imports
import datetime as dt

# Third party imports
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller

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

def adf_fuller_test(df: pd.DataFrame,
                    series: str,
                    significance: str) -> pd.DataFrame:
    series_arr = np.array(df[series])
    adf_res = adfuller(series_arr)
    adf_res_df = pd.DataFrame({'symbol': [series],
                               'test_statistic': [adf_res[0]],
                               'p_value': [adf_res[1]],
                               'lags_used': [adf_res[2]],
                               'number_observations': [adf_res[3]],
                               'critical_value': [adf_res[4][significance]]})
    adf_res_df['stationary'] = adf_res_df['test_statistic'] < adf_res_df['critical_value']
    return adf_res_df

def multiple_adf_tests(df: pd.DataFrame,
                       symbols: list[str],
                       significance: str) -> pd.DataFrame:
    results = []
    for symbol in symbols:
        curr_adf_res = adf_fuller_test(df, symbol, significance)
        results.append(curr_adf_res)
    results_all = pd.concat(results, axis=0).reset_index(drop=True)
    return results_all
    


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
    df_positions['weight'] = df_positions['weight']/number_positions # Equal weight between active positions

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

def calculate_wealth(df: pd.DataFrame,
                     return_col: str) -> pd.DataFrame:
    df_with_wealth = df.copy()
    df_with_wealth['wealth'] = 1 + df_with_wealth[return_col]
    return df_with_wealth

def cum_simple_wealth(df: pd.DataFrame,
                      wealth_col) -> pd.DataFrame:
    df_cum_wealth = df.copy()
    df_cum_wealth['cum_wealth'] = df_cum_wealth[wealth_col].cumprod()
    return df_cum_wealth

def cum_log_wealth(df: pd.DataFrame,
                   wealth_col) -> pd.DataFrame:
    df_log_w = df.copy()
    df_log_w['log_cum_wealth'] = (np.log(df[wealth_col])).cumsum()
    return df_log_w

