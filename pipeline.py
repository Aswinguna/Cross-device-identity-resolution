#!/usr/bin/env python3
"""
Cross-Device Identity Resolution & Contextual Targeting Pipeline
================================================================
Run this script to execute the full end-to-end pipeline:

    python pipeline.py                   # full run (150K sessions)
    python pipeline.py --fast            # quick smoke test (5K sessions)
    python pipeline.py --skip-nlp        # skip NLP / segmentation
    python pipeline.py --config path/to/config.yaml

Steps
-----
  1. Synthetic data generation (150 K session records)
  2. Feature engineering
  3. Persist to SQLite / MySQL
  4. Identity Resolution (MLflow run #1)
     - Pair sampling, pair feature engineering
     - Train Random Forest / XGBoost
     - Evaluate → targets ≥ 87 % accuracy
  5. NLP Processing (MLflow run #2)
     - SpaCy keyword extraction
     - HuggingFace sentence embeddings
  6. Audience Segmentation (K-Means, k chosen by silhouette)
  7. Persist results (identity graph + audience segments) to DB
  8. Print summary report
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

import numpy as np
import pandas as pd
import yaml

# ── Path fix (allows running from project root) ───────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from src.data.generator import generate_sessions, save_sample
from src.data.preprocessing import engineer_session_features
from src.database.connector import build_engine, write_table, read_table, table_exists
from src.identity.hashing import anonymise_dataframe
from src.identity.matching import train_identity_model, save_model
from src.identity.evaluation import print_report, plot_roc, plot_feature_importance, identity_graph_stats
from src.tracking.mlflow_utils import setup_mlflow, start_run, log_dataset_info

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")


# ── Config loader ─────────────────────────────────────────────────────────

def load_config(path: str = "config/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── Step helpers ──────────────────────────────────────────────────────────

def step(n: int, title: str) -> None:
    logger.info("")
    logger.info("=" * 60)
    logger.info("  STEP %d — %s", n, title)
    logger.info("=" * 60)


def banner(text: str) -> None:
    logger.info("")
    logger.info("  ✓  %s", text)


# ── Main pipeline ─────────────────────────────────────────────────────────

def run(cfg: dict, fast: bool = False, skip_nlp: bool = False) -> None:
    t0 = time.time()
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("data/sample", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    data_cfg = cfg.get("data", {})
    seed = data_cfg.get("random_seed", 42)
    n_users = 1000 if fast else data_cfg.get("n_users", 30_000)

    # ── 1. Data generation ────────────────────────────────────────────────
    step(1, "Synthetic Data Generation")
    sessions_raw = generate_sessions(n_users=n_users, seed=seed)
    save_sample(sessions_raw, data_cfg.get("sample_dir", "data/sample") + "/sessions_sample.csv")
    banner(f"Generated {len(sessions_raw):,} sessions for {n_users:,} users")

    # ── 2. Feature engineering ────────────────────────────────────────────
    step(2, "Feature Engineering")
    sessions = engineer_session_features(sessions_raw)
    banner("Session-level features added")

    # ── 3. Persist raw sessions to DB ─────────────────────────────────────
    step(3, "Persist Sessions to Database")
    setup_mlflow(cfg)
    engine = build_engine(cfg)
    write_table(sessions, "sessions", engine)
    banner("Sessions written to database")

    # ── 4. Identity Resolution ────────────────────────────────────────────
    step(4, "Cross-Device Identity Resolution")
    with start_run(cfg, "experiment_identity", run_name=f"identity_{cfg['identity']['model']}") as run:
        log_dataset_info(sessions)

        result = train_identity_model(sessions, cfg, mlflow_run=True)

        metrics = result["metrics"]
        logger.info("─" * 40)
        logger.info("  Accuracy  : %.4f", metrics["accuracy"])
        logger.info("  Precision : %.4f", metrics["precision"])
        logger.info("  Recall    : %.4f", metrics["recall"])
        logger.info("  F1        : %.4f", metrics["f1"])
        logger.info("  ROC-AUC   : %.4f", metrics["roc_auc"])
        logger.info("─" * 40)

        print_report(result["y_test"], result["y_pred"], result["y_prob"])

        # Save model
        save_model(result["model"], result["scaler"], "outputs/identity_model")

        # Save evaluation plots
        plot_roc(result["y_test"], result["y_prob"], save_path="outputs/roc_curve.png")
        if result["feature_importance"] is not None:
            plot_feature_importance(result["feature_importance"], save_path="outputs/feature_importance.png")

        banner(f"Identity model trained — accuracy={metrics['accuracy']:.4f}")

    # ── 5. NLP Processing ─────────────────────────────────────────────────
    if skip_nlp:
        logger.info("Skipping NLP & segmentation (--skip-nlp flag).")
        _print_final_summary(sessions, result["metrics"], None, t0)
        return

    step(5, "NLP Processing — SpaCy + HuggingFace Embeddings")
    with start_run(cfg, "experiment_segmentation", run_name="nlp_embeddings") as _:
        from src.nlp.text_processor import process_sessions_nlp
        from src.segmentation.clustering import run_segmentation

        nlp_cfg = cfg.get("nlp", {})
        sessions_nlp, embeddings = process_sessions_nlp(
            sessions,
            spacy_model=nlp_cfg.get("spacy_model", "en_core_web_sm"),
            embedding_model=nlp_cfg.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2"),
            batch_size=nlp_cfg.get("batch_size", 64),
        )
        banner(f"Embeddings: shape={embeddings.shape}")

        # ── 6. Audience Segmentation ──────────────────────────────────────
        step(6, "Audience Segmentation (K-Means)")
        cluster_labels, seg_profiles, km = run_segmentation(
            sessions_nlp, embeddings, cfg, mlflow_run=True
        )
        sessions_nlp["segment_id"] = cluster_labels

        banner(f"Segmented into {km.n_clusters} audience segments")

        # ── 7. Persist enriched sessions + segment profiles ───────────────
        step(7, "Persist Enriched Data")
        enriched = sessions_nlp.drop(columns=["keyword_list"], errors="ignore")
        enriched["embedding_dim"] = embeddings.shape[1]
        write_table(enriched, "sessions_enriched", engine)
        write_table(seg_profiles.reset_index(), "audience_segments", engine)
        banner("Enriched sessions and segment profiles written to DB")

        # Print segment summary
        logger.info("")
        logger.info("Audience Segment Profiles:")
        logger.info("-" * 60)
        for seg_id, row in seg_profiles.iterrows():
            logger.info(
                "  [%2d] %-35s | n=%6d (%4.1f%%) | top: %s",
                seg_id,
                row["segment_label"][:35],
                row["n_sessions"],
                row["pct_of_total"],
                row["top_categories"],
            )

    # ── 8. Final summary ──────────────────────────────────────────────────
    _print_final_summary(sessions, result["metrics"], seg_profiles, t0)


def _print_final_summary(sessions, id_metrics, seg_profiles, t0) -> None:
    elapsed = time.time() - t0
    logger.info("")
    logger.info("=" * 60)
    logger.info("  PIPELINE COMPLETE  (%.0fs)", elapsed)
    logger.info("=" * 60)
    logger.info("  Sessions generated       : %s", f"{len(sessions):,}")
    logger.info("  Unique users             : %s", f"{sessions['user_id_hash'].nunique():,}")
    logger.info("  Identity match accuracy  : %.4f", id_metrics["accuracy"])
    logger.info("  Identity ROC-AUC         : %.4f", id_metrics["roc_auc"])
    if seg_profiles is not None:
        logger.info("  Audience segments        : %d", len(seg_profiles))
    logger.info("")
    logger.info("  Outputs:")
    logger.info("    Database   → data/processed/sessions.db")
    logger.info("    Model      → outputs/identity_model_clf.joblib")
    logger.info("    MLflow UI  → run: mlflow ui --backend-store-uri outputs/mlflow")
    logger.info("    Dashboard  → run: python dashboard/app.py")
    logger.info("=" * 60)


# ── CLI entry point ───────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-Device Identity Resolution & Contextual Targeting Pipeline"
    )
    parser.add_argument("--config", default="config/config.yaml", help="Path to config.yaml")
    parser.add_argument("--fast", action="store_true", help="Quick test with 1K users instead of 30K")
    parser.add_argument("--skip-nlp", action="store_true", help="Skip NLP/segmentation steps")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run(cfg, fast=args.fast, skip_nlp=args.skip_nlp)


if __name__ == "__main__":
    main()
