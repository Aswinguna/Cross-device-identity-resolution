"""
Synthetic Data Generator
========================
Generates 150K realistic cross-device session records for 30K unique users.

Each user has:
  - 2–8 sessions across mobile / desktop / tablet
  - A consistent behavioral profile (click-rate, scroll depth, active hours)
  - Preferred content categories (2–4 out of 15)
  - Home / work / mobile network assignments

Privacy design:
  - Real user_id is retained ONLY for ground-truth evaluation
  - All PII fields (IP, device FP, user-agent) are SHA-256 hashed
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import uuid
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Content universe
# ---------------------------------------------------------------------------

CONTENT_CATEGORIES: list[str] = [
    "technology", "fashion", "sports", "travel", "food_beverage",
    "automotive", "finance", "health_wellness", "entertainment", "home_decor",
    "gaming", "beauty", "education", "real_estate", "fitness",
]

PAGE_TITLES: dict[str, list[str]] = {
    "technology": [
        "Best Laptops for Programming 2025", "iPhone 16 Pro Review",
        "How to Build a Gaming PC", "Top AI Tools for Developers",
        "MacBook Air M3 vs Dell XPS 13", "Best Mechanical Keyboards",
        "Cloud Storage Comparison 2025", "Smart Home Devices Worth Buying",
        "GPU Prices Finally Dropping", "Best Budget Smartphones 2025",
        "Open-Source LLM Models Compared", "Docker for Beginners",
    ],
    "fashion": [
        "Paris Fashion Week 2025 Highlights", "Best Sustainable Clothing Brands",
        "Summer Wardrobe Essentials", "How to Style Oversized Blazers",
        "Top Designer Collaborations 2025", "Vintage Shopping Guide Paris",
        "Capsule Wardrobe for Minimalists", "Best Online Fashion Retailers",
        "Streetwear Trends This Season", "Luxury Handbag Investment Guide",
    ],
    "sports": [
        "Champions League Final Preview", "Tour de France Route 2025",
        "NBA Playoffs Analysis", "Best Running Shoes for Marathons",
        "Football Transfer Window Updates", "Tennis Grand Slam Schedule",
        "CrossFit vs Traditional Gym", "Fantasy Football Tips",
        "Olympics 2028 Preparation", "Best Sports Nutrition Supplements",
    ],
    "travel": [
        "Best Hidden Gems in Southeast Asia", "Budget Travel Europe 2025",
        "Japan Cherry Blossom Season Guide", "Top Beach Destinations",
        "Digital Nomad Friendly Cities", "Luxury Resorts in Maldives",
        "Road Trip Planning Guide USA", "Best Travel Credit Cards",
        "Backpacking Southeast Asia Tips", "Weekend Getaways from Paris",
    ],
    "food_beverage": [
        "Best Restaurants in Paris 2025", "Easy Meal Prep for Busy Professionals",
        "Plant-Based Diet Beginner Guide", "Wine Pairing for Beginners",
        "Best Coffee Shops in Lyon", "French Pastry Making at Home",
        "Michelin Star Restaurants Worth Visiting", "Trending TikTok Recipes",
        "Best Meal Kit Delivery Services", "Artisan Cheese Selection Guide",
    ],
    "automotive": [
        "Best Electric Cars Under 30000", "Tesla Model Y Long-Term Review",
        "BMW vs Mercedes Comparison", "Car Maintenance Tips for Beginners",
        "Best Used Cars to Buy in 2025", "EV Charging Infrastructure France",
        "Formula 1 Technical Breakdown", "Best Budget SUVs 2025",
        "Hybrid vs Full Electric Which is Better", "Car Insurance Comparison",
    ],
    "finance": [
        "Best ETFs for Long-Term Investing", "Crypto Market Analysis 2025",
        "How to Start Investing in Your 20s", "Best Savings Accounts France",
        "Stock Market Outlook Q2 2025", "Passive Income Ideas 2025",
        "Budget Planning Apps Reviewed", "Real Estate Investment Guide",
        "Retirement Planning for Millennials", "Best Fintech Apps in Europe",
    ],
    "health_wellness": [
        "Best Mental Health Apps 2025", "Intermittent Fasting Complete Guide",
        "Yoga for Beginners Full Routine", "Sleep Optimization Tips",
        "Best Protein Sources for Vegans", "Managing Work Stress Effectively",
        "Gut Health Improvement Guide", "Best Fitness Trackers 2025",
        "Mindfulness Meditation for Anxiety", "Vitamin D Deficiency Signs",
    ],
    "entertainment": [
        "Best Movies to Watch This Weekend", "Netflix Shows Worth Binging",
        "Spotify Wrapped Analysis 2025", "Best Video Games of 2025",
        "Concert Tickets Paris Summer 2025", "Best Podcasts for Learning",
        "Streaming Service Comparison 2025", "Best Books of the Year",
        "Music Festival Guide Europe 2025", "Documentary Recommendations",
    ],
    "home_decor": [
        "Scandinavian Interior Design Guide", "Best Budget Home Makeover Tips",
        "Small Apartment Organization Hacks", "Trending Home Colors 2025",
        "IKEA Hack Ideas 2025", "Best Indoor Plants for Apartments",
        "Smart Home Integration Guide", "Vintage Furniture Finds Online",
        "Minimalist Living Room Ideas", "Best DIY Home Improvement Projects",
    ],
    "gaming": [
        "Best RPGs to Play in 2025", "PS5 vs Xbox Series X Comparison",
        "PC Gaming Setup for Beginners", "Most Anticipated Games 2025",
        "Best Gaming Headsets Under 100", "Indie Games Hidden Gems",
        "Game Pass vs PlayStation Plus", "Streaming Your Gaming on Twitch",
        "Best Gaming Chairs Reviewed", "Esports Tournament Schedule 2025",
    ],
    "beauty": [
        "Best K-Beauty Products 2025", "Natural Skincare Routine for Beginners",
        "Best Drugstore Makeup Dupes", "Anti-Aging Skincare Ingredients",
        "Hair Care Routine for Damaged Hair", "Best Perfumes for Women 2025",
        "Sustainable Beauty Brands Worth It", "SPF Products for Daily Use",
        "Nail Art Trends 2025", "Best Foundation for Different Skin Types",
    ],
    "education": [
        "Best Online Learning Platforms 2025", "Learn Python in 30 Days",
        "Data Science Career Path Guide", "Best MBA Programs in Europe",
        "Free Certification Courses Online", "How to Learn a New Language Fast",
        "Best Books for Self-Development", "PhD Application Tips",
        "Career Change into Tech at 30", "Best Productivity Methods",
    ],
    "real_estate": [
        "First-Time Homebuyer Guide France", "Paris Real Estate Market 2025",
        "Best Neighborhoods in Lyon", "How to Negotiate a Home Price",
        "Rental Property Investment Tips", "Real Estate vs Stock Market",
        "Mortgage Calculator and Advice", "Renovation ROI Which Projects Pay Off",
        "Furnished Apartment vs Unfurnished", "Co-Living Spaces in Major Cities",
    ],
    "fitness": [
        "Best Home Workout Equipment 2025", "30-Day Fitness Challenge",
        "Powerlifting Beginners Guide", "Best Running Apps Reviewed",
        "HIIT vs Steady-State Cardio", "Gym Etiquette for Beginners",
        "Building Muscle After 40", "Nutrition for Athletic Performance",
        "Best Yoga Retreats in Europe", "Marathon Training 16-Week Plan",
    ],
}

DEVICE_TYPES = ["mobile", "desktop", "tablet"]
DEVICE_WEIGHTS = [0.50, 0.35, 0.15]

OS_BY_DEVICE: dict[str, list[str]] = {
    "mobile": ["iOS", "Android"],
    "desktop": ["Windows", "macOS", "Linux"],
    "tablet": ["iOS", "Android", "Windows"],
}

BROWSERS_BY_DEVICE: dict[str, list[str]] = {
    "mobile": ["Chrome Mobile", "Safari Mobile", "Firefox Mobile", "Samsung Browser"],
    "desktop": ["Chrome", "Firefox", "Safari", "Edge", "Opera"],
    "tablet": ["Chrome", "Safari", "Firefox"],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _random_ipv4(rng: np.random.Generator) -> str:
    return (
        f"{rng.integers(1, 255)}.{rng.integers(0, 256)}"
        f".{rng.integers(0, 256)}.{rng.integers(1, 255)}"
    )


def _ip_prefix(ip: str, octets: int = 2) -> str:
    """Return the first `octets` of an IPv4 address (simulates /16 subnet)."""
    return ".".join(ip.split(".")[:octets])


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def generate_sessions(
    n_users: int = 30_000,
    seed: int = 42,
    start_date: datetime = datetime(2024, 7, 1),
    end_date: datetime = datetime(2025, 1, 1),
) -> pd.DataFrame:
    """
    Generate synthetic cross-device session data.

    Parameters
    ----------
    n_users : int
        Number of unique users to simulate. Default 30 000 → ~150 K sessions.
    seed : int
        Random seed for reproducibility.
    start_date / end_date : datetime
        Date range for session timestamps.

    Returns
    -------
    pd.DataFrame
        One row per session (~150 K rows).
    """
    rng = np.random.default_rng(seed)
    random.seed(seed)

    date_range_days = (end_date - start_date).days
    records: list[dict] = []

    logger.info("Generating %s users × sessions …", f"{n_users:,}")

    for user_idx in tqdm(range(n_users), desc="Generating sessions", unit="user"):
        real_user_id = f"user_{user_idx:06d}"

        # ── User profile ────────────────────────────────────────────────────
        n_sessions = int(rng.integers(2, 9))  # 2–8 sessions per user

        n_pref = int(rng.integers(2, 5))
        pref_cats = list(rng.choice(CONTENT_CATEGORIES, size=n_pref, replace=False))

        base_click_rate = float(rng.uniform(0.02, 0.15))     # clicks/second
        base_scroll_depth = float(rng.uniform(0.3, 0.9))
        base_pages = int(rng.integers(2, 12))
        base_duration = int(rng.integers(60, 900))            # seconds

        # Active-hours histogram (bimodal or unimodal)
        n_peaks = int(rng.integers(1, 3))
        peaks = rng.integers(6, 23, size=n_peaks)
        h_hist = np.zeros(24)
        for p in peaks:
            for h in range(24):
                h_hist[h] += np.exp(-0.5 * ((h - p) / 3) ** 2)
        h_hist /= h_hist.sum()

        # Networks
        home_ip = _random_ipv4(rng)
        mobile_ip = _random_ipv4(rng)
        work_ip = _random_ipv4(rng)

        for sess_idx in range(n_sessions):
            session_id = str(uuid.uuid4())

            # Device & browser
            device = str(rng.choice(DEVICE_TYPES, p=DEVICE_WEIGHTS))
            os_name = str(rng.choice(OS_BY_DEVICE[device]))
            browser = str(rng.choice(BROWSERS_BY_DEVICE[device]))

            # Network assignment
            if device == "mobile":
                net_probs = [0.60, 0.30, 0.10]
            else:
                net_probs = [0.10, 0.70, 0.20]
            ip = str(rng.choice([mobile_ip, home_ip, work_ip], p=net_probs))
            ip_pfx = _ip_prefix(ip)

            # Device fingerprint (stable per user+device; rotates occasionally)
            fp_seed = f"{real_user_id}|{device}|{os_name}|{sess_idx // 3}"
            device_fp_hash = _sha256(fp_seed)

            # Timestamp
            day_off = int(rng.integers(0, date_range_days))
            hour = int(rng.choice(24, p=h_hist))
            minute = int(rng.integers(0, 60))
            ts = start_date + timedelta(days=day_off, hours=hour, minutes=minute)

            # Behavioral signals (base + Gaussian noise)
            click_rate = max(0.001, base_click_rate + rng.normal(0, base_click_rate * 0.3))
            scroll_depth = float(np.clip(base_scroll_depth + rng.normal(0, 0.15), 0.05, 1.0))
            pages_n = max(1, int(base_pages + rng.normal(0, 2)))
            duration = max(10, int(base_duration + rng.normal(0, base_duration * 0.4)))
            click_count = max(0, int(click_rate * duration))

            # Content for this session
            n_cats = int(rng.integers(1, min(4, len(pref_cats)) + 1))
            sess_cats = list(rng.choice(pref_cats, size=min(n_cats, len(pref_cats)), replace=False))
            if rng.random() < 0.3:
                extras = [c for c in CONTENT_CATEGORIES if c not in sess_cats]
                if extras:
                    sess_cats.append(random.choice(extras))

            page_texts: list[str] = []
            for cat in sess_cats:
                pool = PAGE_TITLES.get(cat, [])
                k = max(1, pages_n // len(sess_cats))
                page_texts.extend(random.sample(pool, min(k, len(pool))))

            if not page_texts:
                page_texts = ["Homepage visit"]

            records.append(
                {
                    "session_id": session_id,
                    "real_user_id": real_user_id,
                    "user_id_hash": _sha256(real_user_id),  # privacy-safe eval key
                    "device_type": device,
                    "os": os_name,
                    "browser": browser,
                    "ip_prefix": ip_pfx,
                    "ip_prefix_hash": _sha256(ip_pfx),
                    "device_fingerprint_hash": device_fp_hash,
                    "user_agent_hash": _sha256(f"{browser} {os_name}"),
                    "session_start": ts,
                    "session_duration_s": duration,
                    "pages_visited": pages_n,
                    "click_count": click_count,
                    "scroll_depth_avg": round(scroll_depth, 4),
                    "content_categories": json.dumps(sess_cats),
                    "interaction_text": " | ".join(page_texts),
                    "active_hours_profile": json.dumps(h_hist.tolist()),
                }
            )

    df = pd.DataFrame(records)
    df["session_start"] = pd.to_datetime(df["session_start"])

    logger.info(
        "Generated %s sessions | device split: %s",
        f"{len(df):,}",
        df["device_type"].value_counts().to_dict(),
    )
    return df


def save_sample(df: pd.DataFrame, path: str, n: int = 5000) -> None:
    """Save a small sample CSV for quick inspection / testing."""
    sample = df.sample(n=min(n, len(df)), random_state=42)
    sample.to_csv(path, index=False)
    logger.info("Sample saved → %s (%s rows)", path, len(sample))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df = generate_sessions(n_users=30_000)
    print(df.shape)
    print(df.head(3).to_string())
