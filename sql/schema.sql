-- ============================================================
-- Cross-Device Identity Resolution — Database Schema
-- ============================================================
-- Compatible with SQLite (default) and MySQL.
-- The pipeline creates these tables automatically via SQLAlchemy.
-- This file is provided for documentation and manual DB setup.
-- ============================================================

-- ── Raw session records ───────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS sessions (
    session_id              TEXT        PRIMARY KEY,
    real_user_id            TEXT,           -- Ground-truth user ID (eval only)
    user_id_hash            TEXT    NOT NULL,  -- SHA-256 of real_user_id
    device_type             TEXT    NOT NULL,  -- mobile | desktop | tablet
    os                      TEXT,
    browser                 TEXT,
    ip_prefix               TEXT,           -- Coarsened /16 subnet (pre-hash)
    ip_prefix_hash          TEXT,           -- SHA-256 of ip_prefix
    device_fingerprint_hash TEXT,           -- SHA-256 of device attributes
    user_agent_hash         TEXT,
    session_start           TEXT,           -- ISO-8601 timestamp
    session_duration_s      INTEGER,
    pages_visited           INTEGER,
    click_count             INTEGER,
    scroll_depth_avg        REAL,
    content_categories      TEXT,           -- JSON array
    interaction_text        TEXT,           -- Pipe-separated page titles
    active_hours_profile    TEXT,           -- JSON array length 24
    -- Engineered features
    click_rate              REAL,
    pages_per_minute        REAL,
    device_type_enc         INTEGER,
    is_mobile               INTEGER,
    hour_of_day             INTEGER,
    day_of_week             INTEGER,
    n_content_categories    INTEGER
);

CREATE INDEX IF NOT EXISTS idx_sessions_user   ON sessions (user_id_hash);
CREATE INDEX IF NOT EXISTS idx_sessions_device ON sessions (device_type);
CREATE INDEX IF NOT EXISTS idx_sessions_ip     ON sessions (ip_prefix_hash);

-- ── NLP-enriched sessions (written after NLP step) ───────────────────────
CREATE TABLE IF NOT EXISTS sessions_enriched (
    session_id          TEXT    PRIMARY KEY,
    clean_text          TEXT,
    keywords            TEXT,       -- Space-joined SpaCy keywords
    n_keywords          INTEGER,
    segment_id          INTEGER,
    embedding_dim       INTEGER     -- e.g. 384 for all-MiniLM-L6-v2
    -- (full embedding vectors are stored in MLflow artefacts, not the DB)
);

-- ── Identity pairs (sampled during training) ──────────────────────────────
CREATE TABLE IF NOT EXISTS identity_pairs (
    pair_id             INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id_a        TEXT    NOT NULL,
    session_id_b        TEXT    NOT NULL,
    label               INTEGER NOT NULL,   -- 1 = same user, 0 = different
    probability         REAL,               -- Model output
    predicted_label     INTEGER,
    split               TEXT                -- train | test
);

-- ── Audience segment profiles ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS audience_segments (
    segment_id              INTEGER PRIMARY KEY,
    segment_label           TEXT,
    n_sessions              INTEGER,
    pct_of_total            REAL,
    top_keywords            TEXT,
    top_categories          TEXT,
    device_distribution     TEXT,   -- JSON
    avg_scroll_depth        REAL,
    avg_session_duration_s  REAL,
    avg_click_count         REAL,
    avg_pages_visited       REAL,
    pct_mobile              REAL
);
