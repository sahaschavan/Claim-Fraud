-- name: q1_overview
SELECT COUNT(*)                          AS total_claims,
       SUM(fraud)                        AS fraud_claims,
       ROUND(100.0 * AVG(fraud), 1)      AS fraud_rate_pct,
       ROUND(AVG(total_claim_amount))    AS avg_claim,
       ROUND(SUM(total_claim_amount))    AS total_claimed
FROM claims;

-- name: q2_fraud_by_incident_type
SELECT incident_type,
       COUNT(*)                     AS claims,
       SUM(fraud)                   AS fraud_claims,
       ROUND(100.0 * AVG(fraud), 1) AS fraud_rate_pct,
       ROUND(AVG(total_claim_amount)) AS avg_claim
FROM claims
GROUP BY incident_type
ORDER BY fraud_rate_pct DESC;

-- name: q3_fraud_by_severity
SELECT incident_severity,
       COUNT(*)                     AS claims,
       ROUND(100.0 * AVG(fraud), 1) AS fraud_rate_pct,
       ROUND(AVG(total_claim_amount)) AS avg_claim
FROM claims
GROUP BY incident_severity
ORDER BY fraud_rate_pct DESC;

-- name: q4_fraud_by_age_band
SELECT CASE WHEN age < 30 THEN '1) under 30'
            WHEN age < 40 THEN '2) 30-39'
            WHEN age < 50 THEN '3) 40-49'
            ELSE '4) 50+' END        AS age_band,
       COUNT(*)                     AS claims,
       ROUND(100.0 * AVG(fraud), 1) AS fraud_rate_pct,
       ROUND(AVG(total_claim_amount)) AS avg_claim
FROM claims
GROUP BY age_band
ORDER BY age_band;

-- name: q5_hobbies_with_highest_fraud
SELECT insured_hobbies,
       COUNT(*)                     AS claims,
       ROUND(100.0 * AVG(fraud), 1) AS fraud_rate_pct
FROM claims
GROUP BY insured_hobbies
HAVING COUNT(*) >= 20
ORDER BY fraud_rate_pct DESC
LIMIT 8;

-- name: q6_claims_above_average_for_their_type
SELECT c.incident_type,
       COUNT(*)                     AS claims_above_type_avg,
       ROUND(100.0 * AVG(c.fraud), 1) AS fraud_rate_pct
FROM claims c
WHERE c.total_claim_amount > (SELECT AVG(total_claim_amount)
                              FROM claims
                              WHERE incident_type = c.incident_type)
GROUP BY c.incident_type
ORDER BY fraud_rate_pct DESC;

-- name: q7_top3_claims_per_state
WITH ranked AS (
    SELECT incident_state, claim_id, incident_type, total_claim_amount, fraud,
           RANK() OVER (PARTITION BY incident_state ORDER BY total_claim_amount DESC) AS rnk
    FROM claims
)
SELECT * FROM ranked
WHERE rnk <= 3
ORDER BY incident_state, rnk;

-- name: q8_weekly_claims_trend
WITH weekly AS (
    SELECT strftime('%Y-%W', incident_date) AS week,
           COUNT(*)                         AS claims,
           ROUND(SUM(total_claim_amount))   AS amount
    FROM claims
    GROUP BY week
)
SELECT week, claims, amount,
       claims - LAG(claims) OVER (ORDER BY week) AS change_vs_prev_week
FROM weekly
ORDER BY week;

-- name: q9_police_report_and_fraud
SELECT police_report_available,
       CASE WHEN witnesses = 0 THEN 'no witnesses' ELSE 'has witnesses' END AS witness_group,
       COUNT(*)                     AS claims,
       ROUND(100.0 * AVG(fraud), 1) AS fraud_rate_pct
FROM claims
GROUP BY police_report_available, witness_group
ORDER BY police_report_available, witness_group;

-- name: q10_review_list
SELECT claim_id, incident_type, incident_severity, police_report_available,
       total_claim_amount, fraud
FROM claims
WHERE incident_severity IN ('Major Damage', 'Total Loss')
  AND police_report_available != 'YES'
  AND total_claim_amount > (SELECT total_claim_amount FROM claims
                            ORDER BY total_claim_amount
                            LIMIT 1 OFFSET (SELECT CAST(COUNT(*) * 0.75 AS INT) FROM claims))
ORDER BY total_claim_amount DESC
LIMIT 20;
