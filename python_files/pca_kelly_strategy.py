#!/usr/bin/env python3
"""
pca_kelly_strategy.py

Walk-forward compatible strategy implementing:
- Monthly expanding-window refits for PCA (4 factors) and symbol betas
- Daily AR forecasts for factor realizations with fixed lag sets
- One-day-ahead expected return forecasts per symbol
- Factor-model covariance construction
- Constrained Kelly-Markowitz allocation solved with SciPy SLSQP

Public interface expected by walk_forward.py:
  initialise_state(df_train) -> State
  trading_algorithm(new_data, state) -> (trades, new_state)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.ar_model import AutoReg


NUM_FACTORS = 3 # 3 factors has the highest PnL in 2012 validation (way above the rest) -> strange peak
AR_LAG_SPEC: Tuple[Tuple[int, ...], ...] = ((2,), (1,), (1,2), (30,))

# Allocation controls
GROSS_CAP = 1.0 # add leverage if needed (GROSS_CAP = 1.0 means no short-selling)
MAX_ABS_WEIGHT = 1.0 # maximal weight put on one asset (long or short)
KELLY_FRACTION = 0.25 # What is a Kelly Fraction?

# Match project description transaction cost (5 bps) as a turnover penalty
# we include the turnover penalty directly in the optimization
TURNOVER_PENALTY = 0.0005
TURNOVER_SMOOTH_EPS = 1e-8

# Numerical safeguards
IDIO_VAR_FLOOR = 1e-8
FACTOR_VAR_FLOOR = 1e-8
COV_JITTER = 1e-8
WEALTH_FLOOR = 1e-12


@dataclass
class State:
    symbols: list[str]
    symbol_to_idx: Dict[str, int]
    wealth: float
    positions: np.ndarray
    target_weights: np.ndarray
    last_close: np.ndarray
    returns_history: np.ndarray
    factor_history: np.ndarray
    pca_model: PCA
    alpha: np.ndarray
    beta: np.ndarray
    idio_var: np.ndarray
    current_year_month: tuple[int, int]


# TODO: Force the given constraints on the weights
def _project_weights(weights: np.ndarray) -> np.ndarray:
    """Project to the net-zero and gross-cap feasible set approximately."""

    w = np.asarray(weights, dtype=float).copy()
    if w.ndim != 1:
        w = w.ravel()
    if not np.all(np.isfinite(w)):
        return np.zeros_like(w)

    # Net exposure to zero
    w -= np.mean(w)

    # Hard per-name bound for stability
    w = np.clip(w, -MAX_ABS_WEIGHT, MAX_ABS_WEIGHT)

    # Gross exposure cap
    gross = float(np.sum(np.abs(w)))
    if gross > GROSS_CAP and gross > 0.0:
        w *= GROSS_CAP / gross

    # Re-center after scaling/clipping
    w -= np.mean(w)
    gross = float(np.sum(np.abs(w)))
    if gross > GROSS_CAP and gross > 0.0:
        w *= GROSS_CAP / gross

    return w

def _ar_forecast(series: np.ndarray, lags: Tuple[int, ...]) -> tuple[float, float]:
    """Fit no-intercept AR with statsmodels AutoReg and return (forecast, residual_var)."""

    x = np.asarray(series, dtype=float).ravel()
    n = x.shape[0]

    model = AutoReg(x, lags=list(lags), trend="n")
    res = model.fit()
    pred = res.predict(start=n, end=n, dynamic=False)
    forecast = float(np.asarray(pred).ravel()[-1])

    # residual (idiosyncratic) variance of the fitted AR model -> serves as variance estimate of the factors
    resid = np.asarray(res.resid, dtype=float).ravel()
    dof = max(1, resid.size - int(np.asarray(res.params).size))
    resid_var = float((resid @ resid) / dof)

    # numerical stability safeguard in case where variance estimate is close to zero
    resid_var = max(resid_var, FACTOR_VAR_FLOOR)

    return forecast, resid_var

def _fit_factor_model(returns_history: np.ndarray) -> tuple[PCA, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract factors with PCA and fit a regression on each symbol returns series to extract factor betas and alphas.

    Input:
    returns_history: time series numpy array of logarithmic (or simple) returns of shape num_days x num_symbols

    Returns:
      pca_model, factor_history, alpha, beta, idio_var
      (scikit-learn pca model object, time series of the principcal components, symbol alphas and sensitivities,
      idiosyncratic variance estimate for each symbol)

    """

    # construct factor series (our independent variables)
    pca_model = PCA(n_components=NUM_FACTORS)
    factor_history = pca_model.fit_transform(returns_history)

    t, k = factor_history.shape
    reg = LinearRegression(fit_intercept=False)

    # fits the same regression for each column of returns_history (i.e. each ticker symbol)
    reg.fit(factor_history, returns_history)
    fitted = reg.predict(factor_history)

    alpha = np.asarray(reg.intercept_, dtype=float)  # (nun_symbols,)
    beta = np.asarray(reg.coef_, dtype=float)  # (nun_symbols, num_factors)

    # standard idiosyncratic variance estimator (sum_of_squared_residuals / #degrees_of_freedom)
    resid = returns_history - fitted
    dof = t - k - 1
    idio_var = np.sum(resid * resid, axis=0) / dof

    # numerical stability safeguard (for each symbol)
    idio_var = np.maximum(idio_var, IDIO_VAR_FLOOR)

    return pca_model, factor_history, alpha, beta, idio_var

def _forecast_factors(factor_history: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Forecast each factor one step ahead and estimate variances.
    
    Inputs:
    - factor_history: numpy array of fitted PCA factors time series of dimensions num_periods x num_factors
    
    Output:
    - forecasts: numpy array of one-step-ahead forecasts of all factors
    - variances: factor variance estimated  (see _ar_forecast for formula)"""
    
    forecasts = np.zeros(NUM_FACTORS, dtype=float)
    variances = np.zeros(NUM_FACTORS, dtype=float)

    for i in range(NUM_FACTORS):
        forecasts[i], variances[i] = _ar_forecast(factor_history[:, i], AR_LAG_SPEC[i])

    return forecasts, variances


# TODO: Should we regularize the covariance matrix for more stable numerical optimization?
def _solve_kelly_markowitz(
    mu: np.ndarray, cov: np.ndarray, prev_weights: np.ndarray
) -> np.ndarray:
    """Solve constrained fractional Kelly-Markowitz allocation with robust fallback."""
    n = mu.shape[0]
    mu_eff = KELLY_FRACTION * mu  # fractional Kelly inside the objective

    w0 = _project_weights(prev_weights)

    # objective function: optimize the profit, considering the traded volume and risk penalties
    def objective(w: np.ndarray) -> float:

        # Turnover L_2 style penalty for the gradient-based optimizer to handle it better
        # include a small smoothing constant as numeric safeguard
        turnover = np.sqrt((w - prev_weights) ** 2 + TURNOVER_SMOOTH_EPS)
        return float(
            0.5 * w @ cov @ w
            - mu_eff @ w
            + TURNOVER_PENALTY * np.sum(turnover)
        )

    # gradient function of the objective explicitly given to speed-up optimization
    def objective_jac(w: np.ndarray) -> np.ndarray:
        turnover_grad = (w - prev_weights) / np.sqrt(
            (w - prev_weights) ** 2 + TURNOVER_SMOOTH_EPS
        )
        return cov @ w - mu_eff + TURNOVER_PENALTY * turnover_grad


    # 1. Sum of the weights is equal to 0 -> i.e. short exposure = long exposure
    # 2. Leverage constraint: How much in total (as % of total current wealth)
    constraints = (
        {
            "type": "eq",
            "fun": lambda w: float(np.sum(w)),
            "jac": lambda w: np.ones_like(w),
        },
        {
            "type": "ineq",
            "fun": lambda w: float(GROSS_CAP - np.sum(np.abs(w))),
            "jac": lambda w: -np.sign(w),
        },
    )
    bounds = [(-MAX_ABS_WEIGHT, MAX_ABS_WEIGHT)] * n

    try:
        res = minimize(
            objective,
            w0,
            method="SLSQP",
            jac=objective_jac,
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 100, "ftol": 1e-6, "disp": False},
        )
    except Exception:
        return w0

    # Error handling: infinite weights
    if not np.all(np.isfinite(res.x)):
        print("Some weights are infinite,  returning baseline weights (previous period)")
        return w0

    w = _project_weights(res.x)
    if not np.all(np.isfinite(w)):
        print("Some weights are infinite,  returning baseline weights (previous period)")
        return w0

    # Accept non-success solver status if projected candidate improves objective.
    if objective(w) > objective(w0):
        print("Solver has not improved the portfolio profit, returning baseline weights (previous period)")
        return w0

    return w
# TODO: Refit should be more flexible and I want to collect the parameters to examine stability
def _monthly_refit(state: State, dt: pd.Timestamp) -> None:
    """Refit PCA/betas on the expanding history on month boundary."""
    ym = (int(dt.year), int(dt.month))
    if ym == state.current_year_month:
        return

    pca_model, factor_history, alpha, beta, idio_var = _fit_factor_model(state.returns_history)
    state.pca_model = pca_model
    state.factor_history = factor_history
    state.alpha = alpha
    state.beta = beta
    state.idio_var = idio_var
    state.current_year_month = ym


def initialise_state(data: pd.DataFrame) -> State:
    """
    Initialise state from in-sample training data.

    Expects long-format columns: date, symbol, close.
    """
    df = data[["date", "symbol", "close"]].copy()
    df["date"] = pd.to_datetime(df["date"])
    df["symbol"] = df["symbol"].astype(str)

    symbols = sorted(df["symbol"].unique().tolist())
    symbol_to_idx = {s: i for i, s in enumerate(symbols)}

    close_wide = (
        df.pivot(index="date", columns="symbol", values="close")
        .sort_index()
        .reindex(columns=symbols)
    )

    close_mat = close_wide.to_numpy(dtype=float)
    returns_history = np.log(close_mat[1:] / close_mat[:-1])
    returns_history = np.where(np.isfinite(returns_history), returns_history, 0.0)

    pca_model, factor_history, alpha, beta, idio_var = _fit_factor_model(returns_history)

    last_close = close_mat[-1].astype(float)
    n = len(symbols)
    positions = np.zeros(n, dtype=float)
    target_weights = np.zeros(n, dtype=float)

    last_dt = pd.Timestamp(close_wide.index[-1])
    current_year_month = (int(last_dt.year), int(last_dt.month))

    return State(
        symbols=symbols,
        symbol_to_idx=symbol_to_idx,
        wealth=1.0,
        positions=positions,
        target_weights=target_weights,
        last_close=last_close,
        returns_history=returns_history,
        factor_history=factor_history,
        pca_model=pca_model,
        alpha=alpha,
        beta=beta,
        idio_var=idio_var,
        current_year_month=current_year_month,
    )


def trading_algorithm(new_data: pd.DataFrame, state: State) -> tuple[np.ndarray, State]:
    """
    Core function that implements the daily Factor-based trading strategy.

    Inputs:
    - state: an instance of the State dataclass which represents a snapshot of the portfolio
    - new_data: a single-day pandas dataframe with OHLCV data for all 100 symbols

    Outputs:
    - trades: position adjustments for the input date, based on forecasted expected factor values

    Daily trading strategy steps:
    1) monthly refit on month change using info through prior day
    2) update internal wealth/positions with realized close-to-close move
    3) append today's log returns
    4) compute today's factors and AR-based t+1 factor forecast
    5) map to symbol expected returns + covariance
    6) solve constrained Kelly-Markowitz and return trades
    """

    # extract day and close price information
    df = new_data[["date", "symbol", "close"]].copy()
    df["symbol"] = df["symbol"].astype(str)
    df["_idx"] = df["symbol"].map(state.symbol_to_idx)
    closes_today = df["close"].to_numpy(dtype=float)
    dt = pd.to_datetime(df["date"].iloc[0])

    # Refit at first trading day of each month with history available through prior day.
    # modifies the mutable state class instance
    _monthly_refit(state, dt)

    # Update internal PnL state with realized simple returns from prior close -> current close.
    simple_r = closes_today / state.last_close - 1.0
    simple_r = np.where(np.isfinite(simple_r), simple_r, 0.0)

    state.wealth = float(state.wealth + np.sum(state.positions * simple_r))
    state.positions = state.positions * (1.0 + simple_r)

    # Append today's log returns to expanding history.
    log_r = np.log(closes_today / state.last_close)
    log_r = np.where(np.isfinite(log_r), log_r, 0.0) # Why do we do this?

    state.returns_history = np.vstack([state.returns_history, log_r[None, :]])

    # Today's factor realization under current active PCA basis. (Is this applying the factor loadings?)
    factor_today = state.pca_model.transform(log_r.reshape(1, -1))[0]
    state.factor_history = np.vstack([state.factor_history, factor_today[None, :]])

    # Daily AR factor updates and one-step factor forecast.
    f_hat, f_var = _forecast_factors(state.factor_history)

    mu = state.alpha + state.beta @ f_hat # forecasted expected return

    # Diagonal covariance matrix since the extracted Principal Components are orthogonal
    sigma_f = np.diag(np.maximum(f_var, FACTOR_VAR_FLOOR))
    
    # covariance matrix breaks down as a sum of factor-explained variance and idiosyncratic variance
    cov = state.beta @ sigma_f @ state.beta.T + np.diag(np.maximum(state.idio_var, IDIO_VAR_FLOOR))

    # Degenerate case: if wealth is practically zero, flatten all positions.
    if state.wealth <= WEALTH_FLOOR:
        target_positions = np.zeros_like(state.positions)
        trades = target_positions - state.positions
        state.positions = target_positions
        state.target_weights = np.zeros_like(state.target_weights)
        state.last_close = closes_today
        return trades.astype(float), state
    # Convert optimization weights to target dollar positions.
    else:
        current_weights = state.positions / state.wealth
        current_weights = np.where(np.isfinite(current_weights), current_weights, 0.0)

    # Kelly-optimal trades construction
    opt_weights = _solve_kelly_markowitz(mu=mu, cov=cov, prev_weights=current_weights)
    target_positions = opt_weights * state.wealth
    trades = target_positions - state.positions

    state.positions = target_positions
    state.target_weights = opt_weights
    state.last_close = closes_today

    return trades.astype(float), state
