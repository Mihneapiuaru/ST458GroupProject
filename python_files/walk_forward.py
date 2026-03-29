# walk_forward.py
# Python version of the R script

import numpy as np
import pandas as pd

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

    # loop through the test dates day by day
    for i, dt in enumerate(test_dates):

        # one daily dataframe (long format)
        new_data = df_test[df_test["date"] == dt]
        trades, state = strategy(new_data, state)

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

            daily_pnl[i] = np.sum(positions * r1d) - cost_rate * np.sum(np.abs(trades))

            # update our positions for each ticker symbol
            positions = positions * (1.0 + r1d) + trades

    # cummulative PnL
    wealth_seq = 1.0 + np.cumsum(daily_pnl)

    # Ensure wealth does not go negative
    if np.any(wealth_seq <= 0):
        first_idx = np.where(wealth_seq <= 0)[0][0]
        wealth_seq[first_idx:] = 0.0

    return wealth_seq


if __name__ == "__main__":
    df = pd.read_csv("df_train.csv")
    df["date"] = pd.to_datetime(df["date"]).dt.date

    # train test split configuration
    train_idx = df["date"] < pd.to_datetime("2012-01-01").date()
    df_train = df[train_idx].copy()
    df_test = df[~train_idx].copy()

    import python_files.example_script_PCA as example_script_PCA  # your strategy file

    wealth_seq = walk_forward(
        example_script_PCA.trading_algorithm,
        example_script_PCA.initialise_state,
        df_train,
        df_test,
        cost_rate=0.0005,
    )

    print("log wealth =", np.log(wealth_seq[-1]) if wealth_seq[-1] > 0 else -np.inf)