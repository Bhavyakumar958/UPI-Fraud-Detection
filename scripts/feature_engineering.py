"""
feature_engineering.py
-----------------------
Reads raw UPI transactions and engineers 15 behavioural features.

Features engineered:
  1.  txn_hour                  - hour of day (0-23)
  2.  txn_day_of_week           - day of week (0=Mon)
  3.  is_night_txn              - 1 if txn between 23:00-05:00
  4.  is_weekend                - 1 if Sat/Sun
  5.  amount_log                - log1p(amount) — normalises skew
  6.  amount_zscore             - z-score within sender
  7.  sender_txn_velocity_1h    - txns by sender in prior 1 hour
  8.  sender_txn_velocity_24h   - txns by sender in prior 24 hours
  9.  sender_avg_amount_30d     - rolling 30-day avg amount per sender
  10. sender_txn_count_30d      - rolling 30-day txn count per sender
  11. device_mismatch           - device different from sender's usual (raw col)
  12. geo_anomaly               - state different from sender's usual (raw col)
  13. is_new_receiver           - receiver never seen before from this sender
  14. amount_round_number       - amount is a round number (mod 500 == 0)
  15. txn_status_encoded        - SUCCESS=1, FAILED=0

Input : data/upi_transactions_raw.csv
Output: data/upi_transactions_features.csv
"""

import pandas as pd
import numpy as np
import os

RAW_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'upi_transactions_raw.csv')
OUT_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'upi_transactions_features.csv')

# ─── Load ───────────────────────────────────────────────────────────────────
print("[1/5] Loading raw data …")
df = pd.read_csv(RAW_PATH, parse_dates=['timestamp'])
df = df.sort_values('timestamp').reset_index(drop=True)
print(f"      {len(df):,} rows loaded.")

# ─── Time features ──────────────────────────────────────────────────────────
print("[2/5] Engineering time features …")
df['txn_hour'] = df['timestamp'].dt.hour
df['txn_day_of_week'] = df['timestamp'].dt.dayofweek
df['is_night_txn'] = df['txn_hour'].apply(lambda h: 1 if (h >= 23 or h <= 5) else 0)
df['is_weekend'] = df['txn_day_of_week'].apply(lambda d: 1 if d >= 5 else 0)

# ─── Amount features ────────────────────────────────────────────────────────
print("[3/5] Engineering amount features …")
df['amount_log'] = np.log1p(df['amount'])

# Z-score per sender (how unusual is this amount for this sender?)
sender_stats = df.groupby('sender_vpa')['amount'].agg(['mean', 'std']).rename(
    columns={'mean': '_smean', 'std': '_sstd'})
df = df.join(sender_stats, on='sender_vpa')
df['_sstd'] = df['_sstd'].fillna(1).replace(0, 1)
df['amount_zscore'] = (df['amount'] - df['_smean']) / df['_sstd']
df['amount_zscore'] = df['amount_zscore'].clip(-10, 10)
df.drop(columns=['_smean', '_sstd'], inplace=True)

df['amount_round_number'] = (df['amount'] % 500 == 0).astype(int)

# ─── Velocity features (rolling window) ─────────────────────────────────────
print("[4/5] Engineering velocity & behavioural features (this may take ~30s) …")
df = df.set_index('timestamp')

def compute_velocity(df_indexed, window: str):
    """Count prior txns per sender within `window` (e.g. '1h', '24h')."""
    counts = []
    for sender, grp in df_indexed.groupby('sender_vpa'):
        # Rolling count excluding current row
        rolling = grp.rolling(window, closed='left')['amount'].count()
        counts.append(rolling)
    result = pd.concat(counts).reindex(df_indexed.index)
    return result.fillna(0).astype(int)

df['sender_txn_velocity_1h'] = compute_velocity(df, '1h')
df['sender_txn_velocity_24h'] = compute_velocity(df, '24h')

# Rolling 30-day average amount & count per sender
def rolling_30d_stats(df_indexed):
    avgs, counts = [], []
    for sender, grp in df_indexed.groupby('sender_vpa'):
        roll = grp.rolling('30D', closed='left')['amount']
        avgs.append(roll.mean())
        counts.append(roll.count())
    avg_series = pd.concat(avgs).reindex(df_indexed.index).fillna(df_indexed['amount'])
    cnt_series = pd.concat(counts).reindex(df_indexed.index).fillna(0).astype(int)
    return avg_series, cnt_series

df['sender_avg_amount_30d'], df['sender_txn_count_30d'] = rolling_30d_stats(df)

df = df.reset_index()

# Seen-before receiver per sender
df['sender_receiver_pair'] = df['sender_vpa'] + '|' + df['receiver_vpa']
seen_pairs = set()
is_new_receiver = []
for pair in df['sender_receiver_pair']:
    is_new_receiver.append(0 if pair in seen_pairs else 1)
    seen_pairs.add(pair)
df['is_new_receiver'] = is_new_receiver
df.drop(columns=['sender_receiver_pair'], inplace=True)

# txn status encoding
df['txn_status_encoded'] = (df['txn_status'] == 'SUCCESS').astype(int)

# ─── Save ───────────────────────────────────────────────────────────────────
print("[5/5] Saving feature-engineered data …")
df.to_csv(OUT_PATH, index=False)

feature_cols = [
    'txn_hour', 'txn_day_of_week', 'is_night_txn', 'is_weekend',
    'amount_log', 'amount_zscore', 'sender_txn_velocity_1h',
    'sender_txn_velocity_24h', 'sender_avg_amount_30d', 'sender_txn_count_30d',
    'device_mismatch', 'geo_anomaly', 'is_new_receiver',
    'amount_round_number', 'txn_status_encoded'
]

print(f"\n{'='*55}")
print(f"  Output        : {OUT_PATH}")
print(f"  Total rows    : {len(df):,}")
print(f"  Features      : {len(feature_cols)} behavioural features")
for i, f in enumerate(feature_cols, 1):
    print(f"    {i:2d}. {f}")
print(f"{'='*55}\n")
