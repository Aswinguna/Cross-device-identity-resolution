"""
Probabilistic Cross-Device Identity Resolution
"""

from __future__ import annotations

import logging
import random
from typing import Any

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

from src.data.preprocessing import FEATURE_COLS, build_pair_features

logger = logging.getLogger(__name__)


def sample_pairs(
    sessions: pd.DataFrame,
    n_pos_per_user: int = 3,
    neg_pos_ratio: float = 1.5,
    seed: int = 42,
) -> pd.DataFrame:
    rng = random.Random(seed)
    np.random.seed(seed)

    user_sessions: dict[str, list[str]] = (
        sessions.groupby("user_id_hash")["session_id"]
        .apply(list)
        .to_dict()
    )

    positives: list[tuple[str, str, int]] = []
    for uid, sids in user_sessions.items():
        if len(sids) < 2:
            continue
        for _ in range(n_pos_per_user):
            a, b = rng.sample(sids, 2)
            positives.append((a, b, 1))

    n_neg = int(len(positives) * neg_pos_ratio)
    all_sids = sessions["session_id"].tolist()
    sid_to_uid = sessions.set_index("session_id")["user_id_hash"].to_dict()

    negatives: list[tuple[str, str, int]] = []
    attempts = 0
    while len(negatives) < n_neg and attempts < n_neg * 10:
        a, b = rng.sample(all_sids, 2)
        if sid_to_uid[a] != sid_to_uid[b]:
            negatives.append((a, b, 0))
        attempts += 1

    all_pairs = positives + negatives
    rng.shuffle(all_pairs)

    df = pd.DataFrame(all_pairs, columns=["session_id_a", "session_id_b", "label"])
    logger.info("Sampled %s pairs | positives=%s negatives=%s",
        f"{len(df):,}", f"{len(positives):,}", f"{len(negatives):,}")
    return df


def build_feature_matrix(
    pairs: pd.DataFrame,
    sessions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    feat_df = build_pair_features(
        sessions,
        pairs["session_id_a"].tolist(),
        pairs["session_id_b"].tolist(),
    )
    X = feat_df[FEATURE_COLS].fillna(0)
    y = pairs["label"].values
    return X, pd.Series(y)


def _build_model(model_name: str, cfg: dict) -> Any:
    if model_name == "random_forest":
        params = cfg.get("rf_params", {})
        return RandomForestClassifier(**params)
    elif model_name == "xgboost":
        if not _HAS_XGB:
            raise ImportError("xgboost is not installed.")
        params = cfg.get("xgb_params", {})
        return XGBClassifier(**params)
    elif model_name == "logistic_regression":
        return LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    else:
        raise ValueError(f"Unknown model: {model_name!r}")


def train_identity_model(
    sessions: pd.DataFrame,
    cfg: dict,
    mlflow_run: bool = True,
) -> dict[str, Any]:
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, roc_auc_score, confusion_matrix,
    )

    id_cfg = cfg.get("identity", {})
    model_name: str = id_cfg.get("model", "random_forest")
    test_size: float = id_cfg.get("test_size", 0.20)
    threshold: float = id_cfg.get("threshold", 0.50)
    seed: int = cfg.get("data", {}).get("random_seed", 42)

    pairs = sample_pairs(
        sessions,
        n_pos_per_user=id_cfg.get("n_pairs_per_user", 3),
        neg_pos_ratio=id_cfg.get("neg_pos_ratio", 1.5),
        seed=seed,
    )

    X, y = build_feature_matrix(pairs, sessions)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    model = _build_model(model_name, id_cfg)
    logger.info("Training %s on %s pairs ...", model_name, f"{len(X_train):,}")
    model.fit(X_train_sc, y_train)

    y_prob = model.predict_proba(X_test_sc)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    # Only numeric values — MLflow cannot log strings as metrics
    numeric_metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_prob)), 4),
        "threshold": float(threshold),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }

    metrics = {**numeric_metrics, "model": model_name}

    cm = confusion_matrix(y_test, y_pred)
    logger.info("Identity model metrics: %s", metrics)
    logger.info("Confusion matrix:\n%s", cm)

    feat_imp = None
    if hasattr(model, "feature_importances_"):
        feat_imp = pd.Series(
            model.feature_importances_, index=FEATURE_COLS
        ).sort_values(ascending=False)

    if mlflow_run:
        mlflow.log_params({"model": model_name, "threshold": str(threshold)})
        mlflow.log_metrics(numeric_metrics)
        if feat_imp is not None:
            for feat, imp in feat_imp.items():
                mlflow.log_metric(f"fi_{feat}", round(float(imp), 6))

    return {
        "model": model,
        "scaler": scaler,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "metrics": metrics,
        "feature_importance": feat_imp,
        "confusion_matrix": cm,
    }


def save_model(model, scaler, path_prefix: str = "outputs/identity_model") -> None:
    joblib.dump(model, f"{path_prefix}_clf.joblib")
    joblib.dump(scaler, f"{path_prefix}_scaler.joblib")
    logger.info("Model saved -> %s_clf.joblib", path_prefix)


def load_model(path_prefix: str = "outputs/identity_model"):
    model = joblib.load(f"{path_prefix}_clf.joblib")
    scaler = joblib.load(f"{path_prefix}_scaler.joblib")
    return model, scaler


def predict_match(
    session_a: pd.Series,
    session_b: pd.Series,
    model,
    scaler,
    threshold: float = 0.50,
) -> dict[str, float]:
    sessions_mini = pd.DataFrame([session_a, session_b])
    feat_df = build_pair_features(
        sessions_mini,
        [session_a["session_id"]],
        [session_b["session_id"]],
    )
    X = feat_df[FEATURE_COLS].fillna(0)
    X_sc = scaler.transform(X)
    prob = model.predict_proba(X_sc)[0, 1]
    return {"probability": float(prob), "is_same_user": prob >= threshold}
