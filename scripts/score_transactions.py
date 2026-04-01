"""
score_transactions.py
---------------------
Loads a trained XGBoost model and scores new/unseen transactions.
Outputs a risk-scored CSV with fraud probability and risk tier.

Usage:
    python scripts/score_transactions.py \
        --input data/upi_transactions_features.csv \
        --output outputs/scored_transactions.csv

Risk tiers:
  HIGH   : probability >= 0.70
  MEDIUM : probability >= 0.40
  LOW    : probability <  0.40
"""

import argparse
import pandas as pd
import numpy as np
import os
import joblib

BASE = os.path.dirname(__file__)
DEFAULT_INPUT  = os.path.join(BASE, '..', 'data', 'upi_transactions_features.csv')
DEFAULT_OUTPUT = os.path.join(BASE, '..', 'outputs', 'scored_transactions.csv')
MODEL_PATH     = os.path.join(BASE, '..', 'models', 'xgb_fraud_model.joblib')

FEATURE_COLS = [
    'txn_hour', 'txn_day_of_week', 'is_night_txn', 'is_weekend',
    'amount_log', 'amount_zscore', 'sender_txn_velocity_1h',
    'sender_txn_velocity_24h', 'sender_avg_amount_30d', 'sender_txn_count_30d',
    'device_mismatch', 'geo_anomaly', 'is_new_receiver',
    'amount_round_number', 'txn_status_encoded'
]

OPTIMAL_THRESHOLD = 0.50   # update to match train_model.py output

def risk_tier(prob: float) -> str:
    if prob >= 0.70:
        return 'HIGH'
    elif prob >= 0.40:
        return 'MEDIUM'
    return 'LOW'

def main(input_path: str, output_path: str):
    print(f"Loading model from {MODEL_PATH} …")
    model = joblib.load(MODEL_PATH)

    print(f"Loading transactions from {input_path} …")
    df = pd.read_csv(input_path)

    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    X = df[FEATURE_COLS]

    print("Scoring …")
    df['fraud_probability'] = model.predict_proba(X)[:, 1]
    df['fraud_flag']        = (df['fraud_probability'] >= OPTIMAL_THRESHOLD).astype(int)
    df['risk_tier']         = df['fraud_probability'].apply(risk_tier)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    total   = len(df)
    flagged = df['fraud_flag'].sum()
    print(f"\n{'='*50}")
    print(f"  Records scored  : {total:,}")
    print(f"  Flagged as fraud: {flagged:,}  ({flagged/total*100:.2f}%)")
    print(f"  HIGH risk       : {(df['risk_tier']=='HIGH').sum():,}")
    print(f"  MEDIUM risk     : {(df['risk_tier']=='MEDIUM').sum():,}")
    print(f"  LOW risk        : {(df['risk_tier']=='LOW').sum():,}")
    print(f"  Output saved to : {output_path}")
    print(f"{'='*50}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Score UPI transactions for fraud.')
    parser.add_argument('--input',  default=DEFAULT_INPUT)
    parser.add_argument('--output', default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    main(args.input, args.output)
