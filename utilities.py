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