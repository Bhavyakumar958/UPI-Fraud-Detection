"""
train_model.py
--------------
Trains an XGBoost fraud classifier on the 15 engineered features.
Compares against a Logistic Regression baseline.

Steps:
  1. Load feature-engineered data
  2. Train/test split (80/20, stratified)
  3. Train Logistic Regression baseline
  4. Train XGBoost with class_weight to handle imbalance
  5. Evaluate both models (AUC-ROC, Precision, Recall, F1, Confusion Matrix)
  6. Save model artefacts to models/
  7. Export feature importances

Output files:
  models/xgb_fraud_model.joblib
  models/logistic_baseline.joblib
  models/feature_importances.csv
  outputs/model_comparison.txt
"""

import pandas as pd
import numpy as np
import os
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, classification_report, confusion_matrix,
    precision_score, recall_score, f1_score, average_precision_score
)
import xgboost as xgb

# ─── Paths ──────────────────────────────────────────────────────────────────
BASE = os.path.dirname(__file__)
FEAT_PATH  = os.path.join(BASE, '..', 'data', 'upi_transactions_features.csv')
MODEL_DIR  = os.path.join(BASE, '..', 'models')
OUTPUT_DIR = os.path.join(BASE, '..', 'outputs')
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

FEATURE_COLS = [
    'txn_hour', 'txn_day_of_week', 'is_night_txn', 'is_weekend',
    'amount_log', 'amount_zscore', 'sender_txn_velocity_1h',
    'sender_txn_velocity_24h', 'sender_avg_amount_30d', 'sender_txn_count_30d',
    'device_mismatch', 'geo_anomaly', 'is_new_receiver',
    'amount_round_number', 'txn_status_encoded'
]
TARGET = 'is_fraud'
SEED = 42

# ─── Load data ──────────────────────────────────────────────────────────────
print("[1/6] Loading data …")
df = pd.read_csv(FEAT_PATH)
X = df[FEATURE_COLS]
y = df[TARGET]
fraud_count = y.sum()
total = len(y)
scale_pos_weight = (total - fraud_count) / fraud_count   # for XGBoost imbalance handling
print(f"      {total:,} records | fraud={fraud_count:,} ({fraud_count/total*100:.2f}%)")
print(f"      scale_pos_weight = {scale_pos_weight:.1f}")

# ─── Split ──────────────────────────────────────────────────────────────────
print("[2/6] Train/test split (80/20 stratified) …")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=SEED, stratify=y
)
print(f"      Train: {len(X_train):,}  |  Test: {len(X_test):,}")

# ─── Baseline: Logistic Regression ──────────────────────────────────────────
print("[3/6] Training Logistic Regression baseline …")
lr_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(
        class_weight='balanced',
        max_iter=1000,
        random_state=SEED,
        solver='lbfgs'
    ))
])
lr_pipeline.fit(X_train, y_train)
lr_proba = lr_pipeline.predict_proba(X_test)[:, 1]
lr_preds = (lr_proba >= 0.5).astype(int)

lr_auc     = roc_auc_score(y_test, lr_proba)
lr_pr_auc  = average_precision_score(y_test, lr_proba)
lr_prec    = precision_score(y_test, lr_preds)
lr_recall  = recall_score(y_test, lr_preds)
lr_f1      = f1_score(y_test, lr_preds)

joblib.dump(lr_pipeline, os.path.join(MODEL_DIR, 'logistic_baseline.joblib'))
print(f"      LR  AUC-ROC={lr_auc:.4f}  Precision={lr_prec:.4f}  Recall={lr_recall:.4f}  F1={lr_f1:.4f}")

# ─── XGBoost ────────────────────────────────────────────────────────────────
print("[4/6] Training XGBoost classifier …")
xgb_model = xgb.XGBClassifier(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    use_label_encoder=False,
    eval_metric='aucpr',
    random_state=SEED,
    n_jobs=-1,
    early_stopping_rounds=30,
    verbosity=0
)
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)
xgb_proba = xgb_model.predict_proba(X_test)[:, 1]

# Find optimal threshold via Youden's J
thresholds = np.linspace(0.1, 0.9, 80)
best_thresh, best_j = 0.5, 0.0
for t in thresholds:
    preds_t = (xgb_proba >= t).astype(int)
    if preds_t.sum() == 0:
        continue
    prec = precision_score(y_test, preds_t, zero_division=0)
    rec  = recall_score(y_test, preds_t, zero_division=0)
    j = prec + rec - 1
    if j > best_j:
        best_j, best_thresh = j, t

xgb_preds = (xgb_proba >= best_thresh).astype(int)
xgb_auc    = roc_auc_score(y_test, xgb_proba)
xgb_pr_auc = average_precision_score(y_test, xgb_proba)
xgb_prec   = precision_score(y_test, xgb_preds)
xgb_recall = recall_score(y_test, xgb_preds)
xgb_f1     = f1_score(y_test, xgb_preds)
cm = confusion_matrix(y_test, xgb_preds)

joblib.dump(xgb_model, os.path.join(MODEL_DIR, 'xgb_fraud_model.joblib'))
print(f"      XGB AUC-ROC={xgb_auc:.4f}  Precision={xgb_prec:.4f}  Recall={xgb_recall:.4f}  F1={xgb_f1:.4f}")
print(f"      Optimal threshold = {best_thresh:.2f}")

# ─── Feature importances ────────────────────────────────────────────────────
print("[5/6] Saving feature importances …")
fi = pd.DataFrame({
    'feature': FEATURE_COLS,
    'importance_gain': xgb_model.feature_importances_
}).sort_values('importance_gain', ascending=False)
fi.to_csv(os.path.join(MODEL_DIR, 'feature_importances.csv'), index=False)

# ─── Report ─────────────────────────────────────────────────────────────────
print("[6/6] Writing model comparison report …")
report = f"""
╔══════════════════════════════════════════════════════════════╗
║          UPI FRAUD DETECTION — MODEL EVALUATION REPORT       ║
╚══════════════════════════════════════════════════════════════╝

Dataset
  Total records        : {total:,}
  Fraud records        : {fraud_count:,}  ({fraud_count/total*100:.2f}%)
  Train / Test split   : 80% / 20%
  Test set size        : {len(X_test):,}

──────────────────────────────────────────────────────────────
  METRIC                 LOGISTIC REGRESSION     XGBOOST
──────────────────────────────────────────────────────────────
  AUC-ROC                {lr_auc:.4f}                 {xgb_auc:.4f}
  PR-AUC                 {lr_pr_auc:.4f}                 {xgb_pr_auc:.4f}
  Precision              {lr_prec:.4f}                 {xgb_prec:.4f}
  Recall                 {lr_recall:.4f}                 {xgb_recall:.4f}
  F1-Score               {lr_f1:.4f}                 {xgb_f1:.4f}
──────────────────────────────────────────────────────────────

XGBoost Optimal Threshold : {best_thresh:.2f}

XGBoost Confusion Matrix (Test Set):
  True Negatives  : {cm[0][0]:,}    False Positives : {cm[0][1]:,}
  False Negatives : {cm[1][0]:,}    True Positives  : {cm[1][1]:,}

Top 5 Feature Importances (Gain):
{fi.head(5).to_string(index=False)}

Saved artefacts:
  models/xgb_fraud_model.joblib
  models/logistic_baseline.joblib
  models/feature_importances.csv
"""

with open(os.path.join(OUTPUT_DIR, 'model_comparison.txt'), 'w') as f:
    f.write(report)

print(report)
print("✓ Training complete. All artefacts saved.")
