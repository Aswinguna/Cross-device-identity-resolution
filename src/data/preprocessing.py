"""
Preprocessing & Feature Engineering
=====================================
Derives session-level features used by the identity resolution model.

Features produced per session
-------------------------------
- click_rate          : clicks / duration (events/s)
- pages_per_minute    : browsing pace
- scroll_depth_avg    : 0–1 engagement depth
- session_duration_s  : raw duration
- device_type_enc     : label-encoded device
- hour_of_day         : session start hour (0–23)
- day_of_week         : 0=Mon … 6=Sun
- n_content_categories: count of unique categories
- is_mobile           : binary flag

Pair features (for identity matching)
---------------------------------------
- ip_prefix_match     : 1 if same /16 subnet
- device_fp_match     : 1 if identical device fingerprint
- device_type_same    : 1 if same device class
- device_pair_enc     : categorical (mobile-desktop, etc.)
- time_delta_hours    : |session_start_A − session_start_B| in hours
- active_hours_cosine : cosine similarity of active-hour histograms
- content_jaccard     : Jaccard similarity of content categories
- click_rate_diff     : |click_rate_A − click_rate_B|
- scroll_depth_diff   : |scroll_depth_A − scroll_depth_B|
- pages_diff          : |pages_per_minute_A − pages_per_minute_B|
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

# ── Encoding maps ──────────────────────────────────────────────────────────
_DEVICE_ENC = {"mobile": 0, "desktop": 1, "tablet": 2}

_DEVICE_PAIR_LABELS: dict[tuple[str, str], str] = {
    ("mobile", "desktop"): "mobile-desktop",
    ("desktop", "mobile"): "mobile-desktop",
    ("mobile", "tablet"): "mobile-tablet",
    ("tablet", "mobile"): "mobile-tablet",
    ("desktop", "tablet"): "desktop-tablet",
    ("tablet", "desktop"): "desktop-tablet",
    ("mobile", "mobile"): "same-mobile",
    ("desktop", "desktop"): "same-desktop",
    ("tablet", "tablet"): "same-tablet",
}

_DEVICE_PAIR_ENC: dict[str, int] = {
    "mobile-desktop": 0,
    "mobile-tablet": 1,
    "desktop-tablet": 2,
    "same-mobile": 3,
    "same-desktop": 4,
    "same-tablet": 5,
}


# ── Session-level features ─────────────────────────────────────────────────

def engineer_session_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features to a sessions DataFrame (in-place copy)."""
    df = df.copy()

    df["click_rate"] = (df["click_count"] / df["session_duration_s"].clip(lower=1)).round(6)
    df["pages_per_minute"] = (df["pages_visited"] / (df["session_duration_s"].clip(lower=1) / 60)).round(4)
    df["device_type_enc"] = df["device_type"].map(_DEVICE_ENC).fillna(-1).astype(int)
    df["is_mobile"] = (df["device_type"] == "mobile").astype(int)

    df["session_start"] = pd.to_datetime(df["session_start"])
    df["hour_of_day"] = df["session_start"].dt.hour
    df["day_of_week"] = df["session_start"].dt.dayofweek

    df["n_content_categories"] = df["content_categories"].apply(
        lambda x: len(json.loads(x)) if isinstance(x, str) else 0
    )

    logger.debug("Session features engineered: %s columns added", 6)
    return df


# ── Pair-level features ────────────────────────────────────────────────────

def _cosine_sim(a: list[float], b: list[float]) -> float:
    va, vb = np.array(a, dtype=float), np.array(b, dtype=float)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    return float(np.dot(va, vb) / denom) if denom > 1e-9 else 0.0


def _jaccard(a_json: str, b_json: str) -> float:
    try:
        sa = set(json.loads(a_json))
        sb = set(json.loads(b_json))
    except (json.JSONDecodeError, TypeError):
        return 0.0
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def build_pair_features(
    sessions: pd.DataFrame,
    session_ids_a: list[str],
    session_ids_b: list[str],
) -> pd.DataFrame:
    """
    Compute matching features for (session_A, session_B) pairs.

    Parameters
    ----------
    sessions : DataFrame
        Must include session-level features from `engineer_session_features`.
    session_ids_a / session_ids_b : lists of session IDs (same length).

    Returns
    -------
    DataFrame of pair features, one row per pair.
    """
    idx = sessions.set_index("session_id")
    rows: list[dict] = []

    for sid_a, sid_b in zip(session_ids_a, session_ids_b):
        a = idx.loc[sid_a]
        b = idx.loc[sid_b]

        dt_a = pd.to_datetime(a["session_start"])
        dt_b = pd.to_datetime(b["session_start"])
        time_delta_h = abs((dt_a - dt_b).total_seconds()) / 3600.0

        ah_a = json.loads(a["active_hours_profile"]) if isinstance(a["active_hours_profile"], str) else a["active_hours_profile"]
        ah_b = json.loads(b["active_hours_profile"]) if isinstance(b["active_hours_profile"], str) else b["active_hours_profile"]
        ah_cosine = _cosine_sim(ah_a, ah_b)

        dev_pair = (a["device_type"], b["device_type"])
        device_pair_label = _DEVICE_PAIR_LABELS.get(dev_pair, "unknown")

        rows.append(
            {
                "session_id_a": sid_a,
                "session_id_b": sid_b,
                # --- Network signals ---
                "ip_prefix_match": int(a["ip_prefix_hash"] == b["ip_prefix_hash"]),
                "device_fp_match": int(a["device_fingerprint_hash"] == b["device_fingerprint_hash"]),
                # --- Device signals ---
                "device_type_same": int(a["device_type"] == b["device_type"]),
                "device_pair_enc": _DEVICE_PAIR_ENC.get(device_pair_label, -1),
                "is_cross_device": int(a["device_type"] != b["device_type"]),
                # --- Temporal signals ---
                "time_delta_hours": round(time_delta_h, 2),
                "time_delta_days": round(time_delta_h / 24.0, 3),
                "same_hour_of_day": int(a["hour_of_day"] == b["hour_of_day"]),
                "same_day_of_week": int(a["day_of_week"] == b["day_of_week"]),
                # --- Behavioral similarity ---
                "active_hours_cosine": round(ah_cosine, 6),
                "click_rate_diff": abs(float(a["click_rate"]) - float(b["click_rate"])),
                "scroll_depth_diff": abs(float(a["scroll_depth_avg"]) - float(b["scroll_depth_avg"])),
                "pages_diff": abs(float(a["pages_per_minute"]) - float(b["pages_per_minute"])),
                "duration_ratio": (
                    min(float(a["session_duration_s"]), float(b["session_duration_s"])) /
                    max(float(a["session_duration_s"]), float(b["session_duration_s"]), 1)
                ),
                # --- Content signals ---
                "content_jaccard": round(_jaccard(a["content_categories"], b["content_categories"]), 6),
                "n_shared_categories": len(
                    set(json.loads(a["content_categories"])) & set(json.loads(b["content_categories"]))
                    if isinstance(a["content_categories"], str) else set()
                ),
            }
        )

    return pd.DataFrame(rows)


FEATURE_COLS: list[str] = [
    "ip_prefix_match",
    "device_fp_match",
    "device_type_same",
    "device_pair_enc",
    "is_cross_device",
    "time_delta_hours",
    "time_delta_days",
    "same_hour_of_day",
    "same_day_of_week",
    "active_hours_cosine",
    "click_rate_diff",
    "scroll_depth_diff",
    "pages_diff",
    "duration_ratio",
    "content_jaccard",
    "n_shared_categories",
]
