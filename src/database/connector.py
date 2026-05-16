"""
Database Connector
==================
Thin abstraction over SQLAlchemy for SQLite (default) and MySQL.
All data is stored / retrieved as pandas DataFrames.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)


def build_engine(cfg: dict) -> Engine:
    """
    Build a SQLAlchemy engine from the `database` section of config.yaml.

    Parameters
    ----------
    cfg : dict  (the full config dict, or just the `database` sub-dict)
    """
    db_cfg = cfg.get("database", cfg)
    backend = db_cfg.get("backend", "sqlite").lower()

    if backend == "sqlite":
        path = db_cfg.get("sqlite_path", "data/processed/sessions.db")
        url = f"sqlite:///{path}"
        logger.info("Using SQLite backend → %s", path)
    elif backend == "mysql":
        m = db_cfg["mysql"]
        url = (
            f"mysql+pymysql://{m['user']}:{m['password']}"
            f"@{m['host']}:{m['port']}/{m['database']}"
        )
        logger.info("Using MySQL backend → %s@%s/%s", m["user"], m["host"], m["database"])
    else:
        raise ValueError(f"Unsupported database backend: {backend!r}")

    return create_engine(url, echo=False)


def write_table(df: pd.DataFrame, table: str, engine: Engine, if_exists: str = "replace") -> None:
    """Write a DataFrame to a database table."""
    df.to_sql(table, engine, if_exists=if_exists, index=False, chunksize=10_000)
    logger.info("Wrote %s rows to table '%s'", f"{len(df):,}", table)


def read_table(table: str, engine: Engine, query: Optional[str] = None) -> pd.DataFrame:
    """Read an entire table (or a custom query) into a DataFrame."""
    sql = query or f"SELECT * FROM {table}"
    with engine.connect() as conn:
        df = pd.read_sql(text(sql), conn)
    logger.info("Read %s rows from '%s'", f"{len(df):,}", table)
    return df


def table_exists(table: str, engine: Engine) -> bool:
    with engine.connect() as conn:
        from sqlalchemy import inspect
        return inspect(engine).has_table(table)
