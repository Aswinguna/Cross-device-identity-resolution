"""
Privacy-Preserving Hashing
===========================
Utilities to anonymise PII fields before storage or model training.

Design principles
-----------------
- One-way SHA-256 hashing — originals are never stored in the model pipeline
- Deterministic: same input → same hash (enables join-based matching)
- Salt support: salt can be rotated to invalidate old hashes
- ``real_user_id`` is retained **only** in the evaluation split, then dropped
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

_DEFAULT_SALT = os.environ.get("HASH_SALT", "cross_device_pipeline_v1")


# ── Core hashing ───────────────────────────────────────────────────────────

def sha256_hash(value: str, salt: str = _DEFAULT_SALT) -> str:
    """Return a salted SHA-256 hex digest."""
    return hmac.new(
        salt.encode(), value.encode(), hashlib.sha256
    ).hexdigest()


def hash_column(series: pd.Series, salt: str = _DEFAULT_SALT) -> pd.Series:
    """Apply salted SHA-256 hashing to every element of a Series."""
    return series.astype(str).apply(lambda v: sha256_hash(v, salt))


# ── DataFrame-level helpers ────────────────────────────────────────────────

PII_COLUMNS = [
    "ip_prefix",      # Already a coarsened /16 — hash for extra privacy
    "user_agent_hash",
    "device_fingerprint_hash",
]


def anonymise_dataframe(
    df: pd.DataFrame,
    columns: Optional[list[str]] = None,
    salt: str = _DEFAULT_SALT,
    drop_real_user_id: bool = True,
) -> pd.DataFrame:
    """
    Hash specified columns and (optionally) drop the real_user_id.

    Parameters
    ----------
    df : DataFrame — the sessions table
    columns : list of column names to (re-)hash; defaults to PII_COLUMNS
    salt : HMAC salt
    drop_real_user_id : if True, remove the ground-truth column from output

    Returns
    -------
    Anonymised copy of the DataFrame.
    """
    df = df.copy()
    cols = columns or PII_COLUMNS

    for col in cols:
        if col in df.columns:
            df[col] = hash_column(df[col], salt=salt)
            logger.debug("Hashed column: %s", col)

    if drop_real_user_id and "real_user_id" in df.columns:
        df = df.drop(columns=["real_user_id"])
        logger.info("Dropped 'real_user_id' — production-safe DataFrame ready.")

    return df


def verify_hash(original: str, hashed: str, salt: str = _DEFAULT_SALT) -> bool:
    """Constant-time comparison (safe against timing attacks)."""
    expected = sha256_hash(original, salt)
    return hmac.compare_digest(expected, hashed)
