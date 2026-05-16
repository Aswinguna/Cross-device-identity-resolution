"""
NLP Text Processor
==================
Extracts contextual signals from session interaction logs using:

  1. **SpaCy** — tokenisation, stop-word removal, lemmatisation,
     noun-chunk & named-entity extraction for keyword signals.

  2. **HuggingFace sentence-transformers** — dense semantic embeddings
     (all-MiniLM-L6-v2, 384-dim) for K-Means clustering.

Pipeline per session
--------------------
  interaction_text  →  SpaCy clean text  →  keywords / entities
                    →  sentence embedding (384-d vector)
"""

from __future__ import annotations

import logging
import re
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Lazy-loaded heavy models
_SPACY_MODEL = None
_EMBEDDER = None


def _get_spacy(model_name: str = "en_core_web_sm"):
    global _SPACY_MODEL
    if _SPACY_MODEL is None:
        import spacy
        try:
            _SPACY_MODEL = spacy.load(model_name)
            logger.info("SpaCy model '%s' loaded.", model_name)
        except OSError:
            logger.warning(
                "SpaCy model '%s' not found. Run: python -m spacy download %s",
                model_name, model_name,
            )
            raise
    return _SPACY_MODEL


def _get_embedder(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    global _EMBEDDER
    if _EMBEDDER is None:
        from sentence_transformers import SentenceTransformer
        _EMBEDDER = SentenceTransformer(model_name)
        logger.info("Sentence transformer '%s' loaded.", model_name)
    return _EMBEDDER


# ── Text cleaning ─────────────────────────────────────────────────────────

_PIPE_RE = re.compile(r"\|+")
_SPACE_RE = re.compile(r"\s+")


def clean_text(raw: str) -> str:
    """Replace pipe separators with spaces and normalise whitespace."""
    text = _PIPE_RE.sub(" ", raw or "")
    return _SPACE_RE.sub(" ", text).strip()


def extract_keywords_spacy(texts: list[str], model_name: str = "en_core_web_sm") -> list[list[str]]:
    """
    Extract keywords (noun chunks + named entities) from a list of texts.

    Returns a list of keyword lists, one per input text.
    """
    nlp = _get_spacy(model_name)
    results: list[list[str]] = []

    for doc in nlp.pipe(texts, batch_size=128, disable=["ner"]):
        keywords: list[str] = []
        # Noun chunks
        for chunk in doc.noun_chunks:
            phrase = " ".join(
                token.lemma_.lower()
                for token in chunk
                if not token.is_stop and not token.is_punct and token.is_alpha and len(token.text) > 2
            )
            if phrase:
                keywords.append(phrase)
        # Named entities (products, orgs, GPE, etc.)
        for ent in doc.ents:
            if ent.label_ in {"PRODUCT", "ORG", "GPE", "EVENT", "WORK_OF_ART", "NORP"}:
                keywords.append(ent.text.lower().strip())
        results.append(list(dict.fromkeys(keywords)))  # deduplicate, preserve order

    return results


# ── Sentence embeddings ───────────────────────────────────────────────────

def embed_texts(
    texts: list[str],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 64,
    show_progress: bool = True,
) -> np.ndarray:
    """
    Generate dense sentence embeddings using a HuggingFace sentence-transformer.

    Parameters
    ----------
    texts : list of strings (cleaned interaction texts)
    model_name : HuggingFace model identifier
    batch_size : encoding batch size

    Returns
    -------
    np.ndarray of shape (N, embedding_dim)
    """
    embedder = _get_embedder(model_name)
    logger.info("Encoding %s texts with %s …", f"{len(texts):,}", model_name)
    embeddings = embedder.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True,  # L2-normalised → cosine sim = dot product
    )
    logger.info("Embeddings shape: %s", embeddings.shape)
    return embeddings


# ── Main processing pipeline ──────────────────────────────────────────────

def process_sessions_nlp(
    sessions: pd.DataFrame,
    spacy_model: str = "en_core_web_sm",
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 64,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Full NLP processing pipeline.

    1. Clean `interaction_text` with regex.
    2. Extract keywords via SpaCy.
    3. Generate sentence embeddings via HuggingFace.

    Returns
    -------
    sessions_nlp : DataFrame with added columns
        [clean_text, keywords, n_keywords]
    embeddings : np.ndarray of shape (N, 384)
    """
    logger.info("Starting NLP processing on %s sessions …", f"{len(sessions):,}")

    df = sessions.copy()
    df["clean_text"] = df["interaction_text"].apply(clean_text)

    # SpaCy keyword extraction
    logger.info("Extracting keywords with SpaCy (%s) …", spacy_model)
    keywords_list = extract_keywords_spacy(df["clean_text"].tolist(), model_name=spacy_model)
    df["keywords"] = [" ".join(kws) for kws in keywords_list]
    df["n_keywords"] = [len(kws) for kws in keywords_list]
    df["keyword_list"] = keywords_list  # keep as Python list for downstream use

    # Sentence embeddings — use clean_text for richer signal
    embeddings = embed_texts(
        df["clean_text"].tolist(),
        model_name=embedding_model,
        batch_size=batch_size,
    )

    logger.info("NLP processing complete.")
    return df, embeddings
