# Standard library imports
import datetime as dt
import sys
from typing import Callable
from pathlib import Path

# Third party applications imports
import pandas as pd
import numpy as np
from statsmodels.tsa.ar_model import AutoReg


def rolling_ar_forecast(data: pd.DataFrame, 
                        endog: str,
                        reg_model,
                        w: int=252,
                        k: int=1,
                        **kwargs) -> pd.DataFrame:
    # Find the dependent variable
    endog_data = data[endog]
    # Convert to numpy.ndarr
    endog_arr = np.array(endog_data)
    n = len(endog_arr)

    # Empty array to hold forecasted values
    forecasts = np.zeros(n-k-w+1)
    # Walk-forward autoregression
    for t in range(w, n-k+1):
        # Find the fitting sample
        train_sample = endog_arr[(t-w):t]
        ar_p = reg_model(train_sample, **kwargs).fit()
        # Find forecast
        curr_forecast = ar_p.forecast(k)
        # Add the k-step forecats
        forecasts[t-w] = curr_forecast[k-1]

    df_forecasts = pd.DataFrame({'forecast': forecasts,
                                endog: endog_arr[(w+k-1):]},
                                index=data.iloc[(w+k-1):, :].index)
    
    return df_forecasts

def multiple_forecasts(data: pd.DataFrame,
                       symbols: list[str],
                       reg_model,
                       w: int=252,
                       k: int=1,
                       **kwargs) -> list[pd.DataFrame]:
    reg_forecasts = []
    for symbol in symbols:
        # Find the rolling AR(1) forecats
        curr_forecast = rolling_ar_forecast(data, symbol, reg_model, w, k, **kwargs)
        # Append results to the list
        reg_forecasts.append(curr_forecast)
    return reg_forecasts


def multiple_scores(forecasts: list[pd.DataFrame],
                    symbols: list[str],
                    metric: Callable,
                    metric_name: str):
    forecast_scores = []
    for i, symbol in enumerate(symbols):
        curr_forecast = forecasts[i]
        curr_score = metric(curr_forecast[symbol], curr_forecast['forecast'])
        forecast_scores.append(curr_score)

    return pd.DataFrame({'symbol': symbols,
                         metric_name: forecast_scores})


def rolling_regression(data: pd.DataFrame,
                       dep: str,
                       indep: list[str],
                       model, w: int=252, k: int=1):
    n = data.shape[0]
    X = data[indep]
    y = data[dep]
    prediction_ls = [] # List to hold predictions
    for t in range(w+k, n-k+1):
        X_train = X.iloc[(t-w-k):(t-k+1), :]
        X_test = X.iloc[t:(t+1), :]
        y_train = y.iloc[(t-w-k):(t-k+1)]
        model.fit(X_train, y_train)
        curr_pred = model.predict(X_test)
        if curr_pred.ndim == 2:
            prediction_ls.append(curr_pred[0][0])
        else:
            prediction_ls.append(curr_pred[0])
    res = pd.DataFrame({'forecast': prediction_ls,
                        dep: y.iloc[(w+k):]},
                        index=data.iloc[(w+k):, :].index)
    return res








