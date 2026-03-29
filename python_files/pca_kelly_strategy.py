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


NUM_FACTORS = 4
AR_LAG_SPEC: Tuple[Tuple[int, ...], ...] = ((1, 5), (2,), (1,), (30,))

# Allocation controls
GROSS_CAP = 1.0
MAX_ABS_WEIGHT = 1.0 # add leverage
KELLY_FRACTION = 0.25
# Match walk_forward.py default transaction cost (5 bps) as a turnover penalty.
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


# TODO: What's this?
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
    p_max = max(lags)
    n = x.shape[0]

    if n <= p_max + 1:
        if n == 0:
            return 0.0, FACTOR_VAR_FLOOR
        fallback_var = float(np.var(x, ddof=1)) if n > 1 else FACTOR_VAR_FLOOR
        return float(x[-1]), max(fallback_var, FACTOR_VAR_FLOOR)

    try:
        model = AutoReg(x, lags=list(lags), trend="n")
        res = model.fit()

        pred = res.predict(start=n, end=n, dynamic=False)
        forecast = float(np.asarray(pred).ravel()[-1])

        resid = np.asarray(res.resid, dtype=float).ravel()
        if resid.size == 0:
            resid_var = FACTOR_VAR_FLOOR
        else:
            dof = max(1, resid.size - int(np.asarray(res.params).size))
            resid_var = float((resid @ resid) / dof)
            resid_var = max(resid_var, FACTOR_VAR_FLOOR)
    except Exception:
        fallback_var = float(np.var(x, ddof=1)) if n > 1 else FACTOR_VAR_FLOOR
        return float(x[-1]), max(fallback_var, FACTOR_VAR_FLOOR)

    return forecast, resid_var

def _fit_factor_model(returns_history: np.ndarray) -> tuple[PCA, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit PCA + cross-sectional regressions on the expanding return matrix.

    Returns:
      pca_model, factor_history, alpha, beta, idio_var
    """
    pca_model = PCA(n_components=NUM_FACTORS)
    factor_history = pca_model.fit_transform(returns_history)

    t, k = factor_history.shape
    if k != NUM_FACTORS:
        raise ValueError(f"Expected {NUM_FACTORS} factors, got {k}")

    reg = LinearRegression(fit_intercept=True)
    reg.fit(factor_history, returns_history)
    fitted = reg.predict(factor_history)

    alpha = np.asarray(reg.intercept_, dtype=float)  # (n,)
    beta = np.asarray(reg.coef_, dtype=float)  # (n, k)

    resid = returns_history - fitted
    dof = max(1, t - k - 1)
    idio_var = np.sum(resid * resid, axis=0) / dof
    idio_var = np.maximum(idio_var, IDIO_VAR_FLOOR)

    return pca_model, factor_history, alpha, beta, idio_var

def _forecast_factors(factor_history: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Forecast each factor one step ahead and estimate innovation variances."""
    forecasts = np.zeros(NUM_FACTORS, dtype=float)
    variances = np.zeros(NUM_FACTORS, dtype=float)

    for i in range(NUM_FACTORS):
        forecasts[i], variances[i] = _ar_forecast(factor_history[:, i], AR_LAG_SPEC[i])

    return forecasts, variances


# TODO: What does this do?
def _regularize_covariance(cov: np.ndarray) -> np.ndarray:
    cov = 0.5 * (cov + cov.T)

    diag = np.diag(cov).copy()
    floor_add = np.maximum(IDIO_VAR_FLOOR - diag, 0.0)
    if np.any(floor_add > 0):
        cov[np.diag_indices_from(cov)] += floor_add

    cov[np.diag_indices_from(cov)] += COV_JITTER

    return cov


def _solve_kelly_markowitz(
    mu: np.ndarray, cov: np.ndarray, prev_weights: np.ndarray
) -> np.ndarray:
    """Solve constrained fractional Kelly-Markowitz allocation with robust fallback."""
    n = mu.shape[0]
    mu_eff = KELLY_FRACTION * mu  # fractional Kelly inside the objective

    cov = _regularize_covariance(cov)
    w0 = _project_weights(prev_weights)

    def objective(w: np.ndarray) -> float:
        turnover = np.sqrt((w - prev_weights) ** 2 + TURNOVER_SMOOTH_EPS)
        return float(
            0.5 * w @ cov @ w
            - mu_eff @ w
            + TURNOVER_PENALTY * np.sum(turnover)
        )

    def objective_jac(w: np.ndarray) -> np.ndarray:
        turnover_grad = (w - prev_weights) / np.sqrt(
            (w - prev_weights) ** 2 + TURNOVER_SMOOTH_EPS
        )
        return cov @ w - mu_eff + TURNOVER_PENALTY * turnover_grad

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

    if not np.all(np.isfinite(res.x)):
        return w0

    w = _project_weights(res.x)
    if not np.all(np.isfinite(w)):
        return w0

    # Accept non-success solver status if projected candidate improves objective.
    if objective(w) > objective(w0):
        return w0

    return w


# TODO: Is this a necessary function?
def _align_new_data(new_data: pd.DataFrame, state: State) -> tuple[pd.Timestamp, np.ndarray]:
    """Return (date, closes in state symbol order) for a single-day input frame."""
    if new_data["date"].nunique() != 1:
        raise ValueError("new_data must contain exactly one unique date")

    df = new_data[["date", "symbol", "close"]].copy()
    df["symbol"] = df["symbol"].astype(str)
    df["_idx"] = df["symbol"].map(state.symbol_to_idx)

    if df["_idx"].isna().any():
        missing = sorted(df.loc[df["_idx"].isna(), "symbol"].unique().tolist())
        raise ValueError(f"Unknown symbols in new_data: {missing}")

    df = df.sort_values("_idx")
    closes = df["close"].to_numpy(dtype=float)

    if closes.shape[0] != len(state.symbols):
        raise ValueError(
            f"Expected {len(state.symbols)} symbols for new_data, got {closes.shape[0]}"
        )

    dt = pd.to_datetime(df["date"].iloc[0])
    return dt, closes

# TODO: Refit should be more flexible and I want to collect the parameters to examine stability
def _monthly_refit_if_needed(state: State, dt: pd.Timestamp) -> None:
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

    if close_wide.isna().any().any():
        raise ValueError("Missing close prices found while constructing training matrix")

    close_mat = close_wide.to_numpy(dtype=float)
    if close_mat.shape[0] < 2:
        raise ValueError("Need at least two training dates to compute returns")

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
    Daily strategy step:
    1) monthly refit on month change using info through prior day
    2) update internal wealth/positions with realized close-to-close move
    3) append today's log returns
    4) compute today's factors and AR-based t+1 factor forecast
    5) map to symbol expected returns + covariance
    6) solve constrained Kelly-Markowitz and return trades
    """
    dt, closes_today = _align_new_data(new_data, state)

    # Refit at first trading day of each month with history available through prior day.
    _monthly_refit_if_needed(state, dt)

    # Update internal PnL state with realized simple returns from prior close -> current close.
    simple_r = closes_today / state.last_close - 1.0
    simple_r = np.where(np.isfinite(simple_r), simple_r, 0.0)

    state.wealth = float(state.wealth + np.sum(state.positions * simple_r))
    state.positions = state.positions * (1.0 + simple_r)

    # Append today's log returns to expanding history.
    log_r = np.log(closes_today / state.last_close)
    log_r = np.where(np.isfinite(log_r), log_r, 0.0)
    state.returns_history = np.vstack([state.returns_history, log_r[None, :]])

    # Today's factor realization under current active PCA basis.
    factor_today = state.pca_model.transform(log_r.reshape(1, -1))[0]
    state.factor_history = np.vstack([state.factor_history, factor_today[None, :]])

    # Daily AR factor updates and one-step factor forecast.
    f_hat, f_var = _forecast_factors(state.factor_history)

    mu = state.alpha + state.beta @ f_hat
    sigma_f = np.diag(np.maximum(f_var, FACTOR_VAR_FLOOR))
    cov = state.beta @ sigma_f @ state.beta.T + np.diag(np.maximum(state.idio_var, IDIO_VAR_FLOOR))

    # Degenerate case: if wealth is depleted, flatten.
    if state.wealth <= WEALTH_FLOOR:
        target_positions = np.zeros_like(state.positions)
        trades = target_positions - state.positions
        state.positions = target_positions
        state.target_weights = np.zeros_like(state.target_weights)
        state.last_close = closes_today
        return trades.astype(float), state

    # Convert optimization weights to target dollar positions.
    current_weights = np.zeros_like(state.positions)
    if state.wealth > WEALTH_FLOOR:
        current_weights = state.positions / state.wealth
        current_weights = np.where(np.isfinite(current_weights), current_weights, 0.0)

    opt_weights = _solve_kelly_markowitz(mu=mu, cov=cov, prev_weights=current_weights)
    target_positions = opt_weights * state.wealth

    trades = target_positions - state.positions

    state.positions = target_positions
    state.target_weights = opt_weights
    state.last_close = closes_today

    return trades.astype(float), state
