-- =============================================================
-- UPI FRAUD DETECTION — SQL ANALYTICS SCRIPTS
-- =============================================================
-- Compatible with: PostgreSQL 13+, MySQL 8+, SQLite 3.35+
-- Table: upi_transactions  (import from scored_transactions.csv)
-- =============================================================


-- ───────────────────────────────────────────────────────────────
-- 1. CREATE TABLE SCHEMA
-- ───────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS upi_transactions (
    txn_id                    VARCHAR(20)   PRIMARY KEY,
    timestamp                 TIMESTAMP     NOT NULL,
    sender_vpa                VARCHAR(100),
    receiver_vpa              VARCHAR(100),
    amount                    DECIMAL(15,2),
    bank                      VARCHAR(50),
    device_id                 VARCHAR(50),
    app_name                  VARCHAR(50),
    merchant_name             VARCHAR(100),
    sender_state              VARCHAR(50),
    txn_status                VARCHAR(20),
    failure_reason            VARCHAR(100),
    device_mismatch           SMALLINT      DEFAULT 0,
    geo_anomaly               SMALLINT      DEFAULT 0,
    is_fraud                  SMALLINT      DEFAULT 0,
    fraud_probability         FLOAT,
    fraud_flag                SMALLINT      DEFAULT 0,
    risk_tier                 VARCHAR(10),
    -- engineered features
    txn_hour                  SMALLINT,
    txn_day_of_week           SMALLINT,
    is_night_txn              SMALLINT,
    is_weekend                SMALLINT,
    amount_log                FLOAT,
    amount_zscore             FLOAT,
    sender_txn_velocity_1h    INT,
    sender_txn_velocity_24h   INT,
    sender_avg_amount_30d     FLOAT,
    sender_txn_count_30d      INT,
    is_new_receiver           SMALLINT,
    amount_round_number       SMALLINT,
    txn_status_encoded        SMALLINT
);

CREATE INDEX idx_timestamp    ON upi_transactions(timestamp);
CREATE INDEX idx_sender_vpa   ON upi_transactions(sender_vpa);
CREATE INDEX idx_risk_tier    ON upi_transactions(risk_tier);
CREATE INDEX idx_fraud_flag   ON upi_transactions(fraud_flag);


-- ───────────────────────────────────────────────────────────────
-- 2. KPI DASHBOARD QUERIES (Power BI / reporting)
-- ───────────────────────────────────────────────────────────────

-- 2a. Overall fraud KPIs
SELECT
    COUNT(*)                                                            AS total_transactions,
    SUM(fraud_flag)                                                     AS flagged_transactions,
    ROUND(100.0 * SUM(fraud_flag) / COUNT(*), 2)                       AS fraud_rate_pct,
    ROUND(SUM(CASE WHEN fraud_flag = 1 THEN amount ELSE 0 END), 2)     AS flagged_amount,
    ROUND(AVG(CASE WHEN fraud_flag = 1 THEN amount END), 2)            AS avg_fraud_ticket_size,
    ROUND(AVG(CASE WHEN fraud_flag = 0 THEN amount END), 2)            AS avg_legit_ticket_size
FROM upi_transactions;


-- 2b. Daily fraud trend (for time-series chart)
SELECT
    DATE(timestamp)                                 AS txn_date,
    COUNT(*)                                        AS total_txns,
    SUM(fraud_flag)                                 AS fraud_txns,
    ROUND(100.0 * SUM(fraud_flag) / COUNT(*), 3)   AS daily_fraud_rate_pct,
    ROUND(SUM(CASE WHEN fraud_flag=1 THEN amount ELSE 0 END), 2) AS fraud_amount
FROM upi_transactions
GROUP BY DATE(timestamp)
ORDER BY txn_date;


-- 2c. Fraud breakdown by risk tier
SELECT
    risk_tier,
    COUNT(*)                                                AS txn_count,
    SUM(fraud_flag)                                         AS confirmed_fraud,
    ROUND(AVG(fraud_probability) * 100, 2)                 AS avg_fraud_prob_pct,
    ROUND(SUM(amount), 2)                                   AS total_amount_at_risk
FROM upi_transactions
GROUP BY risk_tier
ORDER BY
    CASE risk_tier WHEN 'HIGH' THEN 1 WHEN 'MEDIUM' THEN 2 ELSE 3 END;


-- 2d. Top 10 fraudulent senders
SELECT
    sender_vpa,
    COUNT(*)                                            AS total_txns,
    SUM(fraud_flag)                                     AS fraud_txns,
    ROUND(100.0 * SUM(fraud_flag) / COUNT(*), 1)        AS fraud_rate_pct,
    ROUND(SUM(CASE WHEN fraud_flag=1 THEN amount END), 2) AS total_fraud_amount
FROM upi_transactions
GROUP BY sender_vpa
HAVING SUM(fraud_flag) > 0
ORDER BY fraud_txns DESC
LIMIT 10;


-- 2e. Fraud by bank
SELECT
    bank,
    COUNT(*)                                                    AS total_txns,
    SUM(fraud_flag)                                             AS fraud_txns,
    ROUND(100.0 * SUM(fraud_flag) / COUNT(*), 2)               AS fraud_rate_pct,
    ROUND(SUM(CASE WHEN fraud_flag=1 THEN amount ELSE 0 END), 2) AS fraud_exposure
FROM upi_transactions
GROUP BY bank
ORDER BY fraud_rate_pct DESC;


-- 2f. Fraud by hour of day (heatmap data)
SELECT
    txn_hour                                                AS hour_of_day,
    COUNT(*)                                                AS total_txns,
    SUM(fraud_flag)                                         AS fraud_txns,
    ROUND(100.0 * SUM(fraud_flag) / COUNT(*), 3)           AS fraud_rate_pct
FROM upi_transactions
GROUP BY txn_hour
ORDER BY txn_hour;


-- 2g. Fraud by app
SELECT
    app_name,
    COUNT(*)                                            AS total_txns,
    SUM(fraud_flag)                                     AS fraud_txns,
    ROUND(100.0 * SUM(fraud_flag) / COUNT(*), 2)        AS fraud_rate_pct
FROM upi_transactions
GROUP BY app_name
ORDER BY fraud_txns DESC;


-- 2h. Device mismatch analysis
SELECT
    device_mismatch,
    COUNT(*)                                                AS txn_count,
    SUM(fraud_flag)                                         AS fraud_count,
    ROUND(100.0 * SUM(fraud_flag) / COUNT(*), 2)           AS fraud_rate_pct
FROM upi_transactions
GROUP BY device_mismatch;


-- 2i. Geographic anomaly vs fraud rate
SELECT
    geo_anomaly,
    COUNT(*)                                            AS txn_count,
    SUM(fraud_flag)                                     AS fraud_count,
    ROUND(100.0 * SUM(fraud_flag) / COUNT(*), 2)       AS fraud_rate_pct,
    ROUND(AVG(amount), 2)                               AS avg_amount
FROM upi_transactions
GROUP BY geo_anomaly;


-- ───────────────────────────────────────────────────────────────
-- 3. OPERATIONAL QUERIES (fraud ops team)
-- ───────────────────────────────────────────────────────────────

-- 3a. All HIGH-risk transactions today (ops queue)
SELECT
    txn_id, timestamp, sender_vpa, receiver_vpa,
    amount, bank, app_name, fraud_probability, risk_tier,
    device_mismatch, geo_anomaly, is_new_receiver, sender_txn_velocity_1h
FROM upi_transactions
WHERE risk_tier = 'HIGH'
  AND DATE(timestamp) = CURRENT_DATE
ORDER BY fraud_probability DESC;


-- 3b. Velocity abuse detection (>10 txns in 1 hour)
SELECT
    sender_vpa,
    MAX(sender_txn_velocity_1h)     AS max_1h_velocity,
    COUNT(*)                        AS total_txns,
    SUM(fraud_flag)                 AS fraud_flags,
    ROUND(SUM(amount), 2)           AS total_amount
FROM upi_transactions
WHERE sender_txn_velocity_1h > 10
GROUP BY sender_vpa
ORDER BY max_1h_velocity DESC
LIMIT 25;


-- 3c. Anomaly: high amount + new receiver + device mismatch (compound risk)
SELECT
    txn_id, timestamp, sender_vpa, receiver_vpa,
    amount, fraud_probability, risk_tier
FROM upi_transactions
WHERE amount > 50000
  AND is_new_receiver = 1
  AND device_mismatch = 1
ORDER BY fraud_probability DESC;


-- 3d. Rolling 7-day fraud rate (trend monitoring)
SELECT
    week_start,
    SUM(total_txns)   AS weekly_txns,
    SUM(fraud_txns)   AS weekly_fraud,
    ROUND(100.0 * SUM(fraud_txns) / SUM(total_txns), 3) AS weekly_fraud_rate_pct
FROM (
    SELECT
        DATE_TRUNC('week', timestamp)  AS week_start,
        COUNT(*)                       AS total_txns,
        SUM(fraud_flag)                AS fraud_txns
    FROM upi_transactions
    GROUP BY DATE_TRUNC('week', timestamp)
) weekly
GROUP BY week_start
ORDER BY week_start;


-- 3e. Feature correlation with fraud (quick diagnostic)
SELECT
    ROUND(AVG(CASE WHEN is_fraud=1 THEN txn_hour END), 2)                AS fraud_avg_hour,
    ROUND(AVG(CASE WHEN is_fraud=0 THEN txn_hour END), 2)                AS legit_avg_hour,
    ROUND(AVG(CASE WHEN is_fraud=1 THEN amount END), 2)                  AS fraud_avg_amount,
    ROUND(AVG(CASE WHEN is_fraud=0 THEN amount END), 2)                  AS legit_avg_amount,
    ROUND(AVG(CASE WHEN is_fraud=1 THEN sender_txn_velocity_24h END), 2) AS fraud_avg_velocity_24h,
    ROUND(AVG(CASE WHEN is_fraud=0 THEN sender_txn_velocity_24h END), 2) AS legit_avg_velocity_24h,
    ROUND(AVG(CASE WHEN is_fraud=1 THEN CAST(device_mismatch AS FLOAT ) END)*100, 1) AS fraud_device_mismatch_pct,
    ROUND(AVG(CASE WHEN is_fraud=0 THEN CAST(device_mismatch AS FLOAT) END)*100, 1) AS legit_device_mismatch_pct
FROM upi_transactions;


