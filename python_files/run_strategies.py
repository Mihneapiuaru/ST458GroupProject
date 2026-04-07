"""
run_strategies.py

Run the 60-day mean-reversion strategy through walk_forward.py and report performance.
Usage (from the project root or python_files/):
    python python_files/run_strategies.py
"""

import os, sys
import numpy as np
import pandas as pd

# Resolve paths robustly regardless of where script is called from
HERE     = os.path.dirname(os.path.abspath(__file__))   
ROOT     = os.path.dirname(HERE)                        
CSV_PATH = os.path.join(ROOT, "df_train.csv")

if HERE not in sys.path:
    sys.path.insert(0, HERE)

from walk_forward import walk_forward
import strategy_simple

# Load & split data
df = pd.read_csv(CSV_PATH)
df["date"] = pd.to_datetime(df["date"]).dt.date

TRAIN_END = pd.to_datetime("2013-01-01").date()
TEST_END  = pd.to_datetime("2014-01-01").date()
df_train  = df[df["date"] < TRAIN_END].copy()
df_test   = df[(df["date"] >= TRAIN_END) & (df["date"] < TEST_END)].copy()
COST_RATE = 0.0005   # 5 basis points, as per the assignment

print(f"Train : {df_train['date'].min()} → {df_train['date'].max()}  ({df_train['date'].nunique()} days)")
print(f"Test  : {df_test['date'].min()}  → {df_test['date'].max()}  ({df_test['date'].nunique()} days)\n")

# Run strategy
print("Running: Simple 60-day Mean-Reversion ...")
wealth_seq = walk_forward(
    strategy_simple.trading_algorithm,
    strategy_simple.initialise_state,
    df_train, df_test,
    cost_rate=COST_RATE,
)

log_w = float(np.log(wealth_seq[-1])) if wealth_seq[-1] > 0 else -10.0
print(f"  Final wealth : {wealth_seq[-1]:.4f}")
print(f"  Log wealth   : {log_w:.4f}")
