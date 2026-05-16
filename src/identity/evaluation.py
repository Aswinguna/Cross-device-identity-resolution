"""
Evaluation Utilities
====================
Metrics, plots, and identity-graph analytics for model evaluation.
"""

from __future__ import annotations

import logging
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    RocCurveDisplay,
    PrecisionRecallDisplay,
    classification_report,
    roc_curve,
    precision_recall_curve,
)

logger = logging.getLogger(__name__)


def print_report(y_true, y_pred, y_prob=None) -> None:
    """Print a full classification report + optional AUC."""
    print(classification_report(y_true, y_pred, target_names=["Different User", "Same User"]))
    if y_prob is not None:
        from sklearn.metrics import roc_auc_score
        print(f"ROC-AUC : {roc_auc_score(y_true, y_prob):.4f}")


def plot_roc(y_true, y_prob, title: str = "ROC Curve", save_path: Optional[str] = None):
    fig, ax = plt.subplots(figsize=(7, 5))
    RocCurveDisplay.from_predictions(y_true, y_prob, ax=ax, name="Identity Model")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        logger.info("ROC curve saved → %s", save_path)
    plt.close()
    return fig


def plot_precision_recall(y_true, y_prob, title: str = "Precision-Recall Curve", save_path: Optional[str] = None):
    fig, ax = plt.subplots(figsize=(7, 5))
    PrecisionRecallDisplay.from_predictions(y_true, y_prob, ax=ax, name="Identity Model")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()
    return fig


def plot_feature_importance(
    feat_imp: pd.Series,
    top_n: int = 15,
    title: str = "Feature Importance",
    save_path: Optional[str] = None,
):
    top = feat_imp.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top.index, top.values, color="#4C72B0")
    ax.set_title(title)
    ax.set_xlabel("Importance")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close()
    return fig


def threshold_sweep(y_true, y_prob, thresholds=None) -> pd.DataFrame:
    """
    Evaluate accuracy, precision, recall, F1 across probability thresholds.
    Useful for picking the operating point.
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    thresholds = thresholds or np.linspace(0.1, 0.9, 17)
    rows = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        rows.append(
            {
                "threshold": round(t, 2),
                "accuracy": round(accuracy_score(y_true, y_pred), 4),
                "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
                "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
                "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
            }
        )
    return pd.DataFrame(rows)


def identity_graph_stats(sessions: pd.DataFrame, predictions: pd.DataFrame) -> dict:
    """
    Compute high-level statistics about the predicted identity graph.

    Parameters
    ----------
    sessions : original sessions DataFrame
    predictions : DataFrame with columns [session_id_a, session_id_b, probability, predicted_label]
    """
    matched = predictions[predictions["predicted_label"] == 1]
    n_matched_pairs = len(matched)
    n_unique_sessions = len(
        set(matched["session_id_a"]) | set(matched["session_id_b"])
    )
    cross_device = matched.merge(
        sessions[["session_id", "device_type"]].rename(columns={"session_id": "session_id_a", "device_type": "dt_a"}),
        on="session_id_a",
    ).merge(
        sessions[["session_id", "device_type"]].rename(columns={"session_id": "session_id_b", "device_type": "dt_b"}),
        on="session_id_b",
    )
    n_cross_device = int((cross_device["dt_a"] != cross_device["dt_b"]).sum())

    return {
        "total_pairs_evaluated": len(predictions),
        "matched_pairs": n_matched_pairs,
        "match_rate_pct": round(100 * n_matched_pairs / max(len(predictions), 1), 2),
        "sessions_in_identity_graph": n_unique_sessions,
        "cross_device_matches": n_cross_device,
        "avg_match_probability": round(float(matched["probability"].mean()), 4),
    }
