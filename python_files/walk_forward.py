# walk_forward.py
# Python version of the R script

import numpy as np
import pandas as pd
import pickle

def walk_forward(strategy, initialiser, df_train, df_test, cost_rate=0.0005):
    """strategy and initialiser is what we create based on developed strategy,
    - df_train we have at our disposal
    - df_test is locked away and provided only upon test time (but we can split train and test locally for validation)
    - cost_rate is 5 basis points"""
    # Initialise state

    state = initialiser(df_train)

    test_dates = sorted(df_test["date"].unique())
    n_test_dates = len(test_dates)

    num_symbols = len(df_test[df_test["date"] == test_dates[0]])
    positions = np.zeros(num_symbols)
    daily_pnl = np.zeros(n_test_dates)

    # collector variables for betas
    betas = []

    # loop through the test dates day by day
    for i, dt in enumerate(test_dates):

        # one daily dataframe (long format)
        new_data = df_test[df_test["date"] == dt]
        trades, state = strategy(new_data, state)

        betas.append(state.beta) #TODO: Remove before submission

        # first oos trading day we just lock our positions and pay the fee
        if i == 0:
            price = new_data["close"].to_numpy()
            positions = trades
            daily_pnl[i] = -cost_rate * np.sum(np.abs(trades))

        else:
            # price from previous day (i.e. what is currently stored as price before locking in new positions)
            price_lag1 = price

            # new price
            price = new_data["close"].to_numpy()

            # percentage price difference compared to yesterday
            r1d = price / price_lag1 - 1.0

            # TODO: We are using simple returns -> they should not sum up like this no?
            daily_pnl[i] = np.sum(positions * r1d) - cost_rate * np.sum(np.abs(trades))
            print(f"PnL on day {dt} is {daily_pnl[i]}")

            # update our positions for each ticker symbol
            positions = positions * (1.0 + r1d) + trades

    # cummulative PnL
    wealth_seq = 1.0 + np.cumsum(daily_pnl)

    # Ensure wealth does not go negative
    if np.any(wealth_seq <= 0):
        first_idx = np.where(wealth_seq <= 0)[0][0]
        wealth_seq[first_idx:] = 0.0

    return wealth_seq, betas, daily_pnl

# Mihnea's version
def walk_forward2(strategy, initialiser, df_train, df_test, cost_rate=0.0005):
    # Initialise state
    state = initialiser(df_train)
    
    test_dates = sorted(df_test["date"].unique())
    n_test_dates = len(test_dates)

    num_symbols = len(df_test[df_test["date"] == test_dates[0]])
    positions = np.zeros(num_symbols)
    daily_pnl = np.zeros(n_test_dates)

    for i, dt in enumerate(test_dates):
        new_data = df_test[df_test["date"] == dt]

        trades, state = strategy(new_data, state)

        if i == 0:
            price = new_data["close"].to_numpy()
            positions = trades
            daily_pnl[i] = -cost_rate * np.sum(np.abs(trades))
        else:
            price_lag1 = price
            price = new_data["close"].to_numpy()
            r1d = price / price_lag1 - 1.0

            daily_pnl[i] = np.sum(positions * r1d) - cost_rate * np.sum(np.abs(trades))
            positions = positions * (1.0 + r1d) + trades

    wealth_seq = 1.0 + np.cumsum(daily_pnl)

    # Ensure wealth does not go negative
    if np.any(wealth_seq <= 0):
        first_idx = np.where(wealth_seq <= 0)[0][0]
        wealth_seq[first_idx:] = 0.0

    return wealth_seq


if __name__ == "__main__":
    df = pd.read_csv("df_train.csv")
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df[df["date"] < pd.to_datetime("2013-01-01").date()]

    # train test split configuration
    train_idx = df["date"] < pd.to_datetime("2012-01-01").date()
    df_train = df[train_idx].copy()
    df_test = df[~train_idx].copy()

    import example_script_PCA as example_script_PCA
    import example_script

    wealth_seq, betas, daily_pnl = walk_forward(
        example_script_PCA.trading_algorithm,
        example_script_PCA.initialise_state,
        df_train,
        df_test,
        cost_rate=0.0005,
    )

    # Sharpe ratio from backtest wealth path
    daily_returns = np.diff(np.insert(wealth_seq, 0, 1.0))  # recovers daily_pnl
    rf_annual = 0.04  # set to e.g. 0.03 for 3% annual risk-free rate
    rf_daily = (1.0 + rf_annual) ** (1.0 / 252.0) - 1.0

    excess_returns = daily_returns - rf_daily
    vol = excess_returns.std(ddof=1)
    print(f"return volatility {vol}")

    sharpe_annualized = np.sqrt(252.0) * excess_returns.mean() / vol if vol > 0 else np.nan
    print(f"Annualized Sharpe ratio: {sharpe_annualized:.3f}")

    with open('betas.pkl', 'wb') as f:
        pickle.dump(betas, f)

    with open('daily_pnl.pkl', 'wb') as f:
        pickle.dump(daily_pnl, f)

    print("log wealth =", np.log(wealth_seq[-1]) if wealth_seq[-1] > 0 else -np.inf)
    print(f"ending wealth of original scale is {np.round(wealth_seq[-1], 3)} with total PnL {np.round((wealth_seq[-1] - 1)*100,2)}%")