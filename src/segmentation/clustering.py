"""
Audience Segmentation via K-Means
===================================
Clusters sessions into audience segments using sentence embeddings.
Automatically selects the best K by silhouette score (optional).
Labels each segment with its top keywords.

Design note
-----------
Moving from **user-level** to **audience-level** targeting means we:
  1. Cluster session embeddings → audience segments (K-Means).
  2. Describe each segment by its dominant keywords & content categories.
  3. Assign every session (and thus every cross-device identity) to a segment.
  4. This enables privacy-preserving, cohort-based ad targeting without
     persistent user-level profiles.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from typing import Optional

import mlflow
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

logger = logging.getLogger(__name__)


# ── K selection ───────────────────────────────────────────────────────────

def find_best_k(
    embeddings: np.ndarray,
    k_range: list[int],
    random_state: int = 42,
    sample_size: int = 10_000,
) -> tuple[int, dict[int, float]]:
    """
    Evaluate K-Means for each k in `k_range` using silhouette score.

    A random subsample is used for speed when the dataset is large.

    Returns
    -------
    best_k : int
    scores : {k: silhouette_score}
    """
    if len(embeddings) > sample_size:
        idx = np.random.default_rng(random_state).choice(len(embeddings), size=sample_size, replace=False)
        sample = embeddings[idx]
    else:
        sample = embeddings

    scores: dict[int, float] = {}
    for k in k_range:
        km = KMeans(n_clusters=k, init="k-means++", n_init=5, random_state=random_state)
        labels = km.fit_predict(sample)
        scores[k] = float(silhouette_score(sample, labels, sample_size=min(5000, len(sample))))
        logger.info("  k=%d → silhouette=%.4f", k, scores[k])

    best_k = max(scores, key=scores.__getitem__)
    logger.info("Best k = %d (silhouette = %.4f)", best_k, scores[best_k])
    return best_k, scores


# ── K-Means training ──────────────────────────────────────────────────────

def train_kmeans(
    embeddings: np.ndarray,
    n_clusters: int,
    init: str = "k-means++",
    max_iter: int = 500,
    random_state: int = 42,
) -> KMeans:
    """Fit and return a K-Means model on L2-normalised embeddings."""
    emb_norm = normalize(embeddings, norm="l2")
    km = KMeans(
        n_clusters=n_clusters,
        init=init,
        n_init=10,
        max_iter=max_iter,
        random_state=random_state,
    )
    km.fit(emb_norm)
    logger.info(
        "K-Means trained: %d clusters | inertia=%.2f",
        n_clusters, km.inertia_,
    )
    return km


# ── Segment labelling ─────────────────────────────────────────────────────

def label_segments(
    sessions: pd.DataFrame,
    cluster_labels: np.ndarray,
    top_n_keywords: int = 8,
    top_n_categories: int = 5,
) -> pd.DataFrame:
    """
    Describe each audience segment by its top keywords and content categories.

    Parameters
    ----------
    sessions : DataFrame with `keywords` (str) and `content_categories` (JSON list)
    cluster_labels : array-like of shape (N,)

    Returns
    -------
    segment_profiles : DataFrame indexed by segment_id
    """
    df = sessions.copy()
    df["segment_id"] = cluster_labels

    profiles: list[dict] = []
    for seg_id in sorted(df["segment_id"].unique()):
        subset = df[df["segment_id"] == seg_id]

        # Top keywords
        all_kws: list[str] = []
        for kws in subset["keywords"].dropna():
            all_kws.extend(str(kws).split())
        top_kws = [w for w, _ in Counter(all_kws).most_common(top_n_keywords)]

        # Top content categories
        all_cats: list[str] = []
        for cats_json in subset["content_categories"].dropna():
            try:
                all_cats.extend(json.loads(cats_json))
            except (json.JSONDecodeError, TypeError):
                pass
        top_cats = [c for c, _ in Counter(all_cats).most_common(top_n_categories)]

        # Device composition
        device_dist = subset["device_type"].value_counts(normalize=True).round(3).to_dict()

        # Behavioural averages
        label = _make_label(top_kws, top_cats, seg_id)

        profiles.append(
            {
                "segment_id": seg_id,
                "segment_label": label,
                "n_sessions": len(subset),
                "pct_of_total": round(100 * len(subset) / len(df), 2),
                "top_keywords": ", ".join(top_kws),
                "top_categories": ", ".join(top_cats),
                "device_distribution": json.dumps(device_dist),
                "avg_scroll_depth": round(float(subset["scroll_depth_avg"].mean()), 3),
                "avg_session_duration_s": round(float(subset["session_duration_s"].mean()), 1),
                "avg_click_count": round(float(subset["click_count"].mean()), 1),
                "avg_pages_visited": round(float(subset["pages_visited"].mean()), 1),
                "pct_mobile": round(100 * (subset["device_type"] == "mobile").mean(), 1),
            }
        )

    return pd.DataFrame(profiles).set_index("segment_id")


def _make_label(keywords: list[str], categories: list[str], seg_id: int) -> str:
    """Create a human-readable segment label."""
    label_parts: list[str] = []
    if categories:
        label_parts.append(categories[0].replace("_", " ").title())
    if len(categories) > 1:
        label_parts.append(categories[1].replace("_", " ").title())
    if keywords:
        label_parts.append(keywords[0].title())
    return " & ".join(label_parts) if label_parts else f"Segment {seg_id}"


# ── Full segmentation pipeline ────────────────────────────────────────────

def run_segmentation(
    sessions: pd.DataFrame,
    embeddings: np.ndarray,
    cfg: dict,
    mlflow_run: bool = True,
) -> tuple[np.ndarray, pd.DataFrame, KMeans]:
    """
    End-to-end audience segmentation.

    Returns
    -------
    cluster_labels : np.ndarray of shape (N,)
    segment_profiles : DataFrame of segment descriptions
    kmeans_model : fitted KMeans
    """
    seg_cfg = cfg.get("segmentation", {})
    k_search = seg_cfg.get("k_search_range", [8, 10, 12])
    n_clusters = seg_cfg.get("n_clusters", 10)
    random_state = seg_cfg.get("kmeans_random_state", 42)
    top_kws = seg_cfg.get("top_keywords_per_segment", 8)

    # Optionally search for best K
    if len(k_search) > 1:
        logger.info("Searching for best k in %s …", k_search)
        best_k, sil_scores = find_best_k(
            embeddings, k_range=k_search, random_state=random_state
        )
        n_clusters = best_k
        if mlflow_run:
            for k, s in sil_scores.items():
                mlflow.log_metric(f"silhouette_k{k}", round(s, 4))
    else:
        sil_scores = {}

    # Train final model
    km = train_kmeans(
        embeddings,
        n_clusters=n_clusters,
        init=seg_cfg.get("kmeans_init", "k-means++"),
        max_iter=seg_cfg.get("kmeans_max_iter", 500),
        random_state=random_state,
    )
    labels = km.labels_

    # Silhouette on full set (subsample for speed)
    sample_n = min(10_000, len(embeddings))
    idx = np.random.default_rng(random_state).choice(len(embeddings), size=sample_n, replace=False)
    sil_full = silhouette_score(normalize(embeddings[idx], norm="l2"), labels[idx])
    logger.info("Final silhouette score (n=%d): %.4f", sample_n, sil_full)

    # Label segments
    seg_profiles = label_segments(sessions, labels, top_n_keywords=top_kws)

    if mlflow_run:
        mlflow.log_params(
            {
                "n_clusters": n_clusters,
                "kmeans_init": seg_cfg.get("kmeans_init", "k-means++"),
                "kmeans_max_iter": seg_cfg.get("kmeans_max_iter", 500),
            }
        )
        mlflow.log_metrics(
            {
                "inertia": round(km.inertia_, 2),
                "silhouette_final": round(sil_full, 4),
                "n_sessions": len(sessions),
            }
        )

    logger.info("Segmentation complete → %d segments.", n_clusters)
    return labels, seg_profiles, km
