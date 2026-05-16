-- ============================================================
-- Cross-Device Identity Resolution — Analytical Queries
-- ============================================================

-- 1. Session count and avg duration by device type
SELECT
    device_type,
    COUNT(*)                          AS n_sessions,
    ROUND(AVG(session_duration_s), 1) AS avg_duration_s,
    ROUND(AVG(scroll_depth_avg), 3)   AS avg_scroll_depth,
    ROUND(AVG(click_count), 1)        AS avg_clicks
FROM sessions
GROUP BY device_type
ORDER BY n_sessions DESC;


-- 2. Users with activity on 2+ device types (cross-device users)
SELECT
    user_id_hash,
    COUNT(DISTINCT device_type) AS n_device_types,
    COUNT(*)                    AS total_sessions
FROM sessions
GROUP BY user_id_hash
HAVING n_device_types > 1
ORDER BY n_device_types DESC
LIMIT 20;


-- 3. Top content categories by session volume
WITH cat_exploded AS (
    -- SQLite JSON1 workaround: parse category strings manually
    SELECT session_id, content_categories FROM sessions
)
SELECT content_categories, COUNT(*) AS n_sessions
FROM cat_exploded
GROUP BY content_categories
ORDER BY n_sessions DESC
LIMIT 20;


-- 4. Hourly session volume (to identify peak hours)
SELECT
    hour_of_day,
    COUNT(*) AS n_sessions,
    ROUND(AVG(scroll_depth_avg), 3) AS avg_scroll
FROM sessions
GROUP BY hour_of_day
ORDER BY hour_of_day;


-- 5. Audience segment overview (join sessions + segments)
SELECT
    s.segment_id,
    g.segment_label,
    COUNT(*) AS n_sessions,
    ROUND(AVG(s.scroll_depth_avg), 3)   AS avg_scroll,
    ROUND(AVG(s.session_duration_s), 1) AS avg_duration_s,
    SUM(CASE WHEN s.device_type = 'mobile' THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS pct_mobile
FROM sessions_enriched AS e
JOIN sessions AS s USING (session_id)
JOIN audience_segments AS g ON e.segment_id = g.segment_id
GROUP BY s.segment_id, g.segment_label
ORDER BY n_sessions DESC;


-- 6. Identity match rate by device pair
SELECT
    sa.device_type AS device_a,
    sb.device_type AS device_b,
    COUNT(*)                                              AS n_pairs,
    SUM(p.predicted_label)                                AS n_matched,
    ROUND(100.0 * SUM(p.predicted_label) / COUNT(*), 2)  AS match_rate_pct
FROM identity_pairs AS p
JOIN sessions AS sa ON p.session_id_a = sa.session_id
JOIN sessions AS sb ON p.session_id_b = sb.session_id
WHERE p.split = 'test'
GROUP BY device_a, device_b
ORDER BY n_pairs DESC;


-- 7. High-confidence same-user pairs (probability > 0.80)
SELECT
    session_id_a,
    session_id_b,
    ROUND(probability, 4) AS match_probability
FROM identity_pairs
WHERE predicted_label = 1
  AND probability > 0.80
ORDER BY probability DESC
LIMIT 50;


-- 8. Users with most diverse content interests
SELECT
    user_id_hash,
    COUNT(DISTINCT device_type)       AS devices,
    COUNT(*)                          AS sessions,
    ROUND(AVG(scroll_depth_avg), 3)   AS avg_scroll,
    ROUND(AVG(session_duration_s), 0) AS avg_duration
FROM sessions
GROUP BY user_id_hash
ORDER BY devices DESC, sessions DESC
LIMIT 20;
