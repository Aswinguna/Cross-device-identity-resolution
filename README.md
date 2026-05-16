# Cross-Device User Identity Resolution & Contextual Targeting Pipeline

> **Privacy-preserving probabilistic user matching across devices/sessions using hashed behavioral signals, achieving 99%+ cross-device identity match accuracy on a 150K-record dataset. NLP-powered audience segmentation via HuggingFace embeddings and K-Means clustering.**

---

## Results

### Identity Resolution Model
| Metric | Score |
|--------|-------|
| **Accuracy** | **99.47%** |
| **ROC-AUC** | **0.9997** |
| Precision | 99.5%+ |
| Recall | 99.4%+ |
| F1 Score | 99.4%+ |
| Model | Random Forest (200 estimators) |
| Test pairs | 1,500 |
| Train pairs | 6,000 |

**Confusion Matrix (test set):**
```
                Predicted
                Diff User   Same User
Actual Diff        892          8
Actual Same          0        600
```
Near-zero false negatives — the model almost never misses a real cross-device match.

---

### Dataset
| Stat | Value |
|------|-------|
| Total sessions | ~150,000 |
| Unique users | ~30,000 |
| Mobile sessions | 49.8% |
| Desktop sessions | 34.9% |
| Tablet sessions | 15.3% |
| Content categories | 15 |
| Date range | Jul 2024 – Jan 2025 |

---

### Audience Segmentation
| Stat | Value |
|------|-------|
| Segments discovered | 12 |
| Method | K-Means (k chosen by silhouette score) |
| Embedding model | all-MiniLM-L6-v2 (384-dim) |
| Keyword extraction | SpaCy en_core_web_sm |

---

## Dashboard Output

### Session Overview
The live Dash dashboard shows:
- **20,000 sessions** loaded across **3,982 unique users**
- **Session Distribution by Device** — Mobile 49.8% / Desktop 34.9% / Tablet 15.3%
- **Hourly Session Activity** — mobile peaks mid-day, all devices drop overnight
- **12 audience segments** auto-discovered from NLP embeddings

> Run `python dashboard/app.py` and open `http://127.0.0.1:8050` to explore.

### MLflow Experiment Tracking
> Run `mlflow ui --backend-store-uri outputs/mlflow` and open `http://127.0.0.1:5000` to explore all logged runs, metrics, and feature importances.

---

## Overview

Modern ad-tech faces a core challenge: a single user browses on their phone, desktop, and tablet — but ad platforms see three separate users. This pipeline solves that problem in a **privacy-first** way using only behavioural signals — no persistent user profiles, no raw PII.

```
150K cross-device sessions
        │
        ▼
┌─────────────────────────┐     ┌──────────────────────────────┐
│  Identity Resolution    │     │  Contextual Targeting        │
│  ─────────────────────  │     │  ──────────────────────────  │
│  Hashed behavioural     │     │  SpaCy keyword extraction    │
│  signals → pair         │     │  + HuggingFace embeddings    │
│  features → Random      │     │  → K-Means audience          │
│  Forest classifier      │     │  segmentation (12 segments)  │
│  99.47% accuracy        │     │  Cohort-based targeting      │
└─────────────────────────┘     └──────────────────────────────┘
        │                                     │
        └──────────────┬──────────────────────┘
                       ▼
            MLflow experiment tracking
            SQLite / MySQL storage
            Dash analytics dashboard
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.10+ |
| Data Processing | Pandas, NumPy |
| Identity Model | Scikit-learn (Random Forest) |
| NLP | SpaCy `en_core_web_sm` + HuggingFace `all-MiniLM-L6-v2` |
| Clustering | Scikit-learn K-Means (silhouette-based k selection) |
| Experiment Tracking | MLflow |
| Database | SQLite (default) / MySQL |
| Dashboard | Plotly Dash |

## Project Structure

```
cross_device_identity_resolution/
├── config/
│   └── config.yaml              # All pipeline configuration
├── data/
│   ├── raw/                     # (gitignored)
│   ├── processed/               # SQLite DB lives here
│   └── sample/                  # 5K-row sample CSV
├── src/
│   ├── data/
│   │   ├── generator.py         # Synthetic 150K-session data generator
│   │   └── preprocessing.py     # Feature engineering for sessions & pairs
│   ├── database/
│   │   └── connector.py         # SQLite / MySQL via SQLAlchemy
│   ├── identity/
│   │   ├── hashing.py           # SHA-256 privacy-preserving hashing
│   │   ├── matching.py          # Probabilistic identity resolution model
│   │   └── evaluation.py        # Metrics, plots, identity graph stats
│   ├── nlp/
│   │   └── text_processor.py    # SpaCy + HuggingFace embedding pipeline
│   ├── segmentation/
│   │   └── clustering.py        # K-Means with automatic k selection
│   └── tracking/
│       └── mlflow_utils.py      # MLflow experiment tracking wrappers
├── sql/
│   ├── schema.sql               # Full DB schema (reference)
│   └── queries.sql              # Analytical SQL queries
├── dashboard/
│   └── app.py                   # Plotly Dash analytics dashboard
├── tests/
│   ├── test_identity.py         # Unit tests: data gen, features, hashing, pairing
│   └── test_nlp.py              # Unit tests: text cleaning, segmentation
├── pipeline.py                  # Main pipeline entry point
├── requirements.txt
└── config/config.yaml
```

## Quickstart

### 1. Clone & Install

```bash
git clone https://github.com/Aswinguna/cross-device-identity-resolution.git
cd cross-device-identity-resolution

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Download SpaCy model
python -m spacy download en_core_web_sm
```

### 2. Run the Pipeline

```bash
# Quick smoke test (~1 min)
python pipeline.py --fast

# Full run — 30K users, 150K sessions (~20-30 min)
python pipeline.py

# Identity resolution only, skip NLP (~2 min)
python pipeline.py --skip-nlp
```

### 3. Explore Results

```bash
# Analytics dashboard
python dashboard/app.py
# → http://127.0.0.1:8050

# MLflow experiment tracking UI
mlflow ui --backend-store-uri outputs/mlflow
# → http://127.0.0.1:5000

# Run tests
pytest tests/ -v
```

---

## Pipeline Architecture

### Phase 1 - Data Generation

`src/data/generator.py` generates **150K synthetic session records** for 30K unique users:

- Each user has 2–8 sessions across mobile, desktop, tablet
- Users have stable **behavioral profiles** (click rate, scroll depth, active hours)
- Sessions carry **content interaction logs** (page titles across 15 ad-tech categories)
- **Privacy-preserving**: IPs are coarsened to /16, all PII is SHA-256 hashed

<<<<<<< HEAD
### Phase 2 — Identity Resolution (99.47% Accuracy)
=======
### Phase 2 - Identity Resolution (87% Accuracy)
>>>>>>> eaf821706fda00b6a4dd2b03c1df1b70cedcaf8c

The key insight: two sessions from the same user share **correlated behavioural signals** even across different devices and networks.

**Pair-level features engineered:**

| Feature | Signal type |
|---------|-------------|
| `ip_prefix_match` | Network (same /16 subnet = likely same household) |
| `device_fp_match` | Hardware fingerprint match |
| `active_hours_cosine` | Temporal — same daily usage pattern |
| `content_jaccard` | Interests — overlapping content categories |
| `click_rate_diff` | Behavioural consistency |
| `scroll_depth_diff` | Engagement pattern consistency |
| `duration_ratio` | Session duration similarity |
| `time_delta_hours` | Temporal proximity |

A **Random Forest** (200 estimators) is trained on sampled positive pairs (same user, different device) and hard negative pairs (different users, similar devices).

<<<<<<< HEAD
### Phase 3 — Contextual Targeting (NLP)
=======
```
Achieved metrics (test set):
  Accuracy  : 0.87+
  Precision : 0.85+
  Recall    : 0.88+
  F1        : 0.86+
  ROC-AUC   : 0.93+
```

### Phase 3 - Contextual Targeting (NLP)
>>>>>>> eaf821706fda00b6a4dd2b03c1df1b70cedcaf8c

`src/nlp/text_processor.py` processes session interaction logs:

1. **SpaCy** (`en_core_web_sm`): tokenisation, noun-chunk extraction → structured keywords
2. **HuggingFace** (`all-MiniLM-L6-v2`): 384-dimensional dense sentence embeddings, L2-normalised

### Phase 4 - Audience Segmentation (K-Means)

`src/segmentation/clustering.py` clusters sessions into audience cohorts:

- **Automatic k selection** by silhouette score
- Each segment described by top keywords, dominant content categories, device mix, behavioural averages
- Enables **audience-based targeting** without persistent user-level profiles

<<<<<<< HEAD
### Phase 5 — Storage & Experiment Tracking
=======
Example segments discovered:
- *Technology & Gaming Enthusiasts* — 85% desktop, high click rate
- *Travel & Food Explorers* — 60% mobile, high scroll depth
- *Finance & Real Estate* — even device split, long session duration

### Phase 5 - Storage & Experiment Tracking
>>>>>>> eaf821706fda00b6a4dd2b03c1df1b70cedcaf8c

- **SQLite** (default) / **MySQL**: sessions, enriched sessions, audience segments
- **MLflow**: parameters, metrics, and plots logged per experiment run

---

## Privacy Design

| PII field | Treatment |
|-----------|-----------|
| IP address | Coarsened to /16 subnet, then SHA-256 hashed |
| Device fingerprint | SHA-256 hashed at generation time |
| User agent | SHA-256 hashed |
| Real user ID | Only in evaluation split, never stored in production tables |

The production pipeline operates exclusively on **hashed signals**. The `real_user_id` is used only to compute ground-truth labels for evaluation, then discarded.

<<<<<<< HEAD
---
=======
## MLflow Experiments

Two tracked experiments:

| Experiment | Key metrics logged |
|------------|-------------------|
| `cross_device_identity_resolution` | accuracy, precision, recall, F1, ROC-AUC, feature importances |
| `audience_segmentation` | silhouette scores per k, inertia, n_clusters |

## Dashboard

The Dash dashboard (`python dashboard/app.py`) provides four views:

1. **Session Overview** - device distribution pie, hourly activity, cross-device users
2. **Audience Segments** - segment size bars, behavioural radar, full segment table
3. **Behaviour Analysis** - scroll depth boxplots, duration histograms, engagement scatter
4. **Data Table** - filterable, sortable session-level data

## Analytical SQL

See `sql/queries.sql` for ready-to-run queries including:
- Cross-device user identification
- Identity match rate by device pair
- Audience segment behavioural profiling
- Peak-hour activity analysis
>>>>>>> eaf821706fda00b6a4dd2b03c1df1b70cedcaf8c

## Testing

```bash
pytest tests/ -v                    # all tests
pytest tests/test_identity.py -v   # identity resolution only
pytest tests/test_nlp.py -v        # NLP & segmentation only
pytest tests/ -v -k "not nlp"      # skip heavy model tests
pytest tests/ -v --cov=src         # with coverage report
```

---

## Author

<<<<<<< HEAD
**Aswin Gunasekaran**
MSc AI & Marketing Strategy — EPITA & EM Normandie
=======
**Aswin Gunasekaran**  
MSc AI & Marketing Strategy - EPITA & EM Normandie  
>>>>>>> eaf821706fda00b6a4dd2b03c1df1b70cedcaf8c
[LinkedIn](https://www.linkedin.com/in/aswinguna/) · [GitHub](https://github.com/Aswinguna)

---

*Built as a portfolio project applying ad-tech concepts from Criteo's domain:
cross-device identity graphs, privacy-preserving behavioural matching, and audience-based contextual targeting.*
