"""
Unit Tests — Identity Resolution
=================================
Run: pytest tests/ -v
     pytest tests/ -v --cov=src --cov-report=term-missing
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.data.generator import generate_sessions
from src.data.preprocessing import engineer_session_features, build_pair_features, FEATURE_COLS
from src.identity.hashing import sha256_hash, hash_column, anonymise_dataframe
from src.identity.matching import sample_pairs, build_feature_matrix


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def small_sessions() -> pd.DataFrame:
    """200 sessions from 50 users — fast fixture for unit tests."""
    df = generate_sessions(n_users=50, seed=0)
    return engineer_session_features(df)


# ── Data generation tests ─────────────────────────────────────────────────

class TestDataGenerator:
    def test_output_shape(self, small_sessions):
        assert len(small_sessions) >= 100, "Expected at least 100 sessions for 50 users"
        assert len(small_sessions) <= 400, "Expected at most 400 sessions for 50 users"

    def test_required_columns(self, small_sessions):
        required = [
            "session_id", "real_user_id", "user_id_hash",
            "device_type", "ip_prefix_hash", "device_fingerprint_hash",
            "session_duration_s", "click_count", "scroll_depth_avg",
            "content_categories", "interaction_text", "active_hours_profile",
        ]
        for col in required:
            assert col in small_sessions.columns, f"Missing column: {col}"

    def test_device_types(self, small_sessions):
        assert set(small_sessions["device_type"].unique()).issubset({"mobile", "desktop", "tablet"})

    def test_no_null_session_ids(self, small_sessions):
        assert small_sessions["session_id"].isna().sum() == 0

    def test_unique_session_ids(self, small_sessions):
        assert small_sessions["session_id"].nunique() == len(small_sessions)

    def test_scroll_depth_range(self, small_sessions):
        assert small_sessions["scroll_depth_avg"].between(0.0, 1.0).all()

    def test_content_categories_valid_json(self, small_sessions):
        for val in small_sessions["content_categories"].head(20):
            parsed = json.loads(val)
            assert isinstance(parsed, list)
            assert len(parsed) >= 1

    def test_active_hours_profile_sums_to_one(self, small_sessions):
        for val in small_sessions["active_hours_profile"].head(10):
            arr = json.loads(val)
            assert len(arr) == 24
            assert abs(sum(arr) - 1.0) < 1e-6, "Active hours profile should sum to 1.0"


# ── Feature engineering tests ─────────────────────────────────────────────

class TestPreprocessing:
    def test_engineered_columns_present(self, small_sessions):
        expected = ["click_rate", "pages_per_minute", "device_type_enc",
                    "is_mobile", "hour_of_day", "day_of_week", "n_content_categories"]
        for col in expected:
            assert col in small_sessions.columns

    def test_click_rate_non_negative(self, small_sessions):
        assert (small_sessions["click_rate"] >= 0).all()

    def test_is_mobile_binary(self, small_sessions):
        assert set(small_sessions["is_mobile"].unique()).issubset({0, 1})

    def test_hour_of_day_range(self, small_sessions):
        assert small_sessions["hour_of_day"].between(0, 23).all()

    def test_pair_features_shape(self, small_sessions):
        sids = small_sessions["session_id"].tolist()
        a_ids = sids[:5]
        b_ids = sids[5:10]
        feat_df = build_pair_features(small_sessions, a_ids, b_ids)
        assert len(feat_df) == 5
        for col in FEATURE_COLS:
            assert col in feat_df.columns, f"Pair feature missing: {col}"

    def test_pair_features_ip_match(self, small_sessions):
        """Sessions with same ip_prefix_hash should get ip_prefix_match=1."""
        df = small_sessions.copy()
        # Manually force two sessions to share the same IP hash
        sid_a = df.iloc[0]["session_id"]
        sid_b = df.iloc[1]["session_id"]
        df.at[df.index[1], "ip_prefix_hash"] = df.iloc[0]["ip_prefix_hash"]
        feat = build_pair_features(df, [sid_a], [sid_b])
        assert feat["ip_prefix_match"].iloc[0] == 1


# ── Hashing tests ─────────────────────────────────────────────────────────

class TestHashing:
    def test_sha256_deterministic(self):
        h1 = sha256_hash("test_value")
        h2 = sha256_hash("test_value")
        assert h1 == h2

    def test_sha256_different_inputs(self):
        assert sha256_hash("abc") != sha256_hash("def")

    def test_sha256_length(self):
        h = sha256_hash("any_value")
        assert len(h) == 64  # 32 bytes as hex

    def test_hash_column(self):
        s = pd.Series(["a", "b", "a"])
        hashed = hash_column(s)
        assert hashed.iloc[0] == hashed.iloc[2]  # same input → same hash
        assert hashed.iloc[0] != hashed.iloc[1]

    def test_anonymise_drops_real_user_id(self, small_sessions):
        anon = anonymise_dataframe(small_sessions, drop_real_user_id=True)
        assert "real_user_id" not in anon.columns

    def test_anonymise_keeps_user_id_hash(self, small_sessions):
        anon = anonymise_dataframe(small_sessions, drop_real_user_id=True)
        assert "user_id_hash" in anon.columns


# ── Pair sampling tests ───────────────────────────────────────────────────

class TestPairSampling:
    def test_pair_labels_balanced(self, small_sessions):
        pairs = sample_pairs(small_sessions, n_pos_per_user=2, neg_pos_ratio=1.0, seed=0)
        pos = (pairs["label"] == 1).sum()
        neg = (pairs["label"] == 0).sum()
        # Should be roughly balanced (within 10%)
        assert abs(pos - neg) / max(pos + neg, 1) < 0.15

    def test_positive_pairs_same_user(self, small_sessions):
        pairs = sample_pairs(small_sessions, seed=0)
        uid_map = small_sessions.set_index("session_id")["user_id_hash"].to_dict()
        pos_pairs = pairs[pairs["label"] == 1]
        for _, row in pos_pairs.iterrows():
            assert uid_map[row["session_id_a"]] == uid_map[row["session_id_b"]], \
                "Positive pair must belong to the same user"

    def test_negative_pairs_different_users(self, small_sessions):
        pairs = sample_pairs(small_sessions, seed=0)
        uid_map = small_sessions.set_index("session_id")["user_id_hash"].to_dict()
        neg_pairs = pairs[pairs["label"] == 0].head(50)
        for _, row in neg_pairs.iterrows():
            assert uid_map[row["session_id_a"]] != uid_map[row["session_id_b"]], \
                "Negative pair must belong to different users"

    def test_feature_matrix_shape(self, small_sessions):
        pairs = sample_pairs(small_sessions, n_pos_per_user=1, neg_pos_ratio=1.0, seed=0)
        X, y = build_feature_matrix(pairs, small_sessions)
        assert X.shape[0] == len(pairs)
        assert X.shape[1] == len(FEATURE_COLS)
        assert len(y) == len(pairs)
        assert set(y.unique()).issubset({0, 1})

    def test_no_nan_features(self, small_sessions):
        pairs = sample_pairs(small_sessions, n_pos_per_user=1, seed=0)
        X, _ = build_feature_matrix(pairs, small_sessions)
        assert not X.isna().any().any(), "Feature matrix should have no NaN values"
