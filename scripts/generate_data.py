"""
generate_data.py
----------------
Generates 200,000+ synthetic UPI transaction records with realistic fraud patterns.
Run this first before any other script.

Output: data/upi_transactions_raw.csv
"""

import pandas as pd
import numpy as np
import random
import os
from datetime import datetime, timedelta

# ─── Config ────────────────────────────────────────────────────────────────────
SEED = 42
NUM_RECORDS = 210_000
FRAUD_RATE = 0.035          # 3.5% fraud (realistic for UPI ecosystem)
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'upi_transactions_raw.csv')

np.random.seed(SEED)
random.seed(SEED)

# ─── Reference pools ───────────────────────────────────────────────────────────
UPI_SUFFIXES = ['@oksbi', '@okaxis', '@okicici', '@ybl', '@paytm', '@upi', '@ibl', '@axl']
BANKS = ['SBI', 'HDFC', 'ICICI', 'Axis', 'Kotak', 'PNB', 'BOI', 'Canara', 'Yes Bank', 'IDFC']
DEVICE_TYPES = ['Android', 'iOS', 'HarmonyOS', 'Android']          # Android overweighted
APP_NAMES = ['PhonePe', 'GPay', 'Paytm', 'BHIM', 'Amazon Pay', 'WhatsApp Pay']
MERCHANTS = ['Flipkart', 'Amazon', 'Swiggy', 'Zomato', 'BigBasket',
             'Myntra', 'MakeMyTrip', 'IRCTC', 'BookMyShow', 'Nykaa',
             'PVR', 'Decathlon', 'Reliance Digital', 'Croma', None]   # None = P2P
STATES = ['Maharashtra', 'Delhi', 'Karnataka', 'Tamil Nadu', 'Uttar Pradesh',
          'Gujarat', 'West Bengal', 'Telangana', 'Rajasthan', 'Madhya Pradesh']
FAILURE_REASONS = [None, None, None, None,        # mostly success
                   'Insufficient funds', 'Bank server error',
                   'Daily limit exceeded', 'Wrong PIN', 'Network timeout']

# ─── Helpers ───────────────────────────────────────────────────────────────────
def random_vpa(prefix_pool=None):
    prefix = prefix_pool or f"user{random.randint(1000, 9_999_999)}"
    return f"{prefix}{random.choice(UPI_SUFFIXES)}"

def random_amount(is_fraud: bool) -> float:
    if is_fraud:
        # Fraudsters cluster near round numbers and high values
        bucket = np.random.choice(['small', 'medium', 'large'], p=[0.25, 0.45, 0.30])
        if bucket == 'small':
            amt = round(np.random.uniform(1, 999), 2)
        elif bucket == 'medium':
            amt = round(np.random.choice([4999, 9999, 14999, 19999, 24999])
                        + np.random.uniform(-5, 5), 2)
        else:
            amt = round(np.random.uniform(50_000, 200_000), 2)
    else:
        amt = round(np.random.exponential(scale=2500), 2)
        amt = min(amt, 100_000)
        amt = max(amt, 1)
    return amt

def random_timestamp(start: datetime, days: int = 365) -> datetime:
    delta = timedelta(
        days=random.randint(0, days),
        hours=random.randint(0, 23),
        minutes=random.randint(0, 59),
        seconds=random.randint(0, 59)
    )
    return start + delta

# ─── Main generation ───────────────────────────────────────────────────────────
print(f"[1/3] Generating {NUM_RECORDS:,} transaction records …")

start_date = datetime(2023, 1, 1)

# Pre-generate a pool of ~15K "real" users (senders)
real_users = [random_vpa() for _ in range(15_000)]
real_devices = {u: f"DEV-{random.randint(100000, 999999)}" for u in real_users}
real_states = {u: random.choice(STATES) for u in real_users}

rows = []
for i in range(NUM_RECORDS):
    is_fraud = np.random.random() < FRAUD_RATE
    txn_id = f"TXN{i+1:08d}"
    ts = random_timestamp(start_date, 365)
    sender = random.choice(real_users)

    # Fraud patterns: new/hijacked sender accounts
    if is_fraud and np.random.random() < 0.6:
        sender = random_vpa(f"new{random.randint(1,999999)}")

    receiver = random_vpa()
    amount = random_amount(is_fraud)
    bank = random.choice(BANKS)
    device = real_devices.get(sender, f"DEV-{random.randint(100000, 999999)}")
    app = random.choice(APP_NAMES)
    merchant = random.choice(MERCHANTS)
    state = real_states.get(sender, random.choice(STATES))

    # Device mismatch: fraud txns sometimes use a different device
    device_mismatch = 0
    if is_fraud and np.random.random() < 0.55:
        device = f"DEV-{random.randint(100000, 999999)}"
        device_mismatch = 1

    # Geographic anomaly
    geo_anomaly = 0
    if is_fraud and np.random.random() < 0.40:
        state = random.choice([s for s in STATES if s != state])
        geo_anomaly = 1

    txn_status = 'SUCCESS'
    failure_reason = None
    if is_fraud and np.random.random() < 0.12:
        txn_status = 'FAILED'
        failure_reason = random.choice(FAILURE_REASONS[4:])
    elif not is_fraud and np.random.random() < 0.06:
        txn_status = 'FAILED'
        failure_reason = random.choice(FAILURE_REASONS[4:])

    rows.append({
        'txn_id': txn_id,
        'timestamp': ts.strftime('%Y-%m-%d %H:%M:%S'),
        'sender_vpa': sender,
        'receiver_vpa': receiver,
        'amount': amount,
        'bank': bank,
        'device_id': device,
        'app_name': app,
        'merchant_name': merchant if merchant else 'P2P',
        'sender_state': state,
        'txn_status': txn_status,
        'failure_reason': failure_reason if failure_reason else '',
        'device_mismatch': device_mismatch,
        'geo_anomaly': geo_anomaly,
        'is_fraud': int(is_fraud),
    })

df = pd.DataFrame(rows)

print(f"[2/3] Saving to {OUTPUT_PATH} …")
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False)

print(f"[3/3] Done.")
print(f"\n{'='*50}")
print(f"  Total records : {len(df):,}")
print(f"  Fraud records : {df['is_fraud'].sum():,}  ({df['is_fraud'].mean()*100:.2f}%)")
print(f"  Saved to      : {OUTPUT_PATH}")
print(f"{'='*50}\n")
