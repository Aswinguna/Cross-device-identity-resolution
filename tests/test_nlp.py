"""
Unit Tests — NLP & Segmentation
==================================
Run: pytest tests/test_nlp.py -v

Note: NLP tests require SpaCy and sentence-transformers to be installed.
      Skip them with: pytest tests/ -v -k "not nlp"
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.generator import generate_sessions
from src.data.preprocessing import engineer_session_features
from src.nlp.text_processor import clean_text


# ── Text cleaning tests (no heavy models needed) ──────────────────────────

class TestTextCleaning:
    def test_clean_removes_pipes(self):
        raw = "AI Laptops | Gaming Mice | Best Deals"
        cleaned = clean_text(raw)
        assert "|" not in cleaned

    def test_clean_normalises_spaces(self):
        raw = "hello   world  |  foo"
        cleaned = clean_text(raw)
        assert "  " not in cleaned

    def test_clean_empty_string(self):
        assert clean_text("") == ""

    def test_clean_none_like(self):
        # Should not raise
        result = clean_text(None)  # type: ignore
        assert result == ""

    def test_clean_preserves_content(self):
        raw = "Best Laptops 2025 | iPhone Review"
        cleaned = clean_text(raw)
        assert "Best Laptops 2025" in cleaned
        assert "iPhone Review" in cleaned


# ── Segmentation tests (no heavy models needed) ───────────────────────────

class TestSegmentation:
    @pytest.fixture
    def dummy_embeddings(self):
        """400-d random unit vectors as stand-in embeddings."""
        rng = np.random.default_rng(42)
        emb = rng.standard_normal((500, 384)).astype(np.float32)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        return emb / norms

    @pytest.fixture
    def dummy_sessions(self):
        df = generate_sessions(n_users=100, seed=1)
        df = engineer_session_features(df)
        df["keywords"] = "technology laptop gaming"
        df["clean_text"] = df["interaction_text"]
        return df

    def test_kmeans_trains(self, dummy_embeddings):
        from src.segmentation.clustering import train_kmeans
        km = train_kmeans(dummy_embeddings, n_clusters=5, random_state=42)
        assert km.n_clusters == 5
        assert len(km.labels_) == len(dummy_embeddings)

    def test_labels_in_range(self, dummy_embeddings):
        from src.segmentation.clustering import train_kmeans
        km = train_kmeans(dummy_embeddings, n_clusters=8, random_state=42)
        assert set(km.labels_).issubset(set(range(8)))

    def test_segment_profiles_shape(self, dummy_sessions, dummy_embeddings):
        from src.segmentation.clustering import train_kmeans, label_segments
        km = train_kmeans(dummy_embeddings[:len(dummy_sessions)], n_clusters=5, random_state=42)
        profiles = label_segments(dummy_sessions, km.labels_, top_n_keywords=5)
        assert len(profiles) == 5
        assert "segment_label" in profiles.columns
        assert "n_sessions" in profiles.columns
        assert "top_categories" in profiles.columns

    def test_segment_sessions_sum_to_total(self, dummy_sessions, dummy_embeddings):
        from src.segmentation.clustering import train_kmeans, label_segments
        km = train_kmeans(dummy_embeddings[:len(dummy_sessions)], n_clusters=5, random_state=42)
        profiles = label_segments(dummy_sessions, km.labels_)
        assert profiles["n_sessions"].sum() == len(dummy_sessions)

    def test_pct_of_total_sums_to_100(self, dummy_sessions, dummy_embeddings):
        from src.segmentation.clustering import train_kmeans, label_segments
        km = train_kmeans(dummy_embeddings[:len(dummy_sessions)], n_clusters=5, random_state=42)
        profiles = label_segments(dummy_sessions, km.labels_)
        assert abs(profiles["pct_of_total"].sum() - 100.0) < 1.0

    def test_find_best_k(self, dummy_embeddings):
        from src.segmentation.clustering import find_best_k
        best_k, scores = find_best_k(dummy_embeddings, k_range=[4, 5, 6], random_state=42)
        assert best_k in {4, 5, 6}
        assert all(isinstance(v, float) for v in scores.values())
