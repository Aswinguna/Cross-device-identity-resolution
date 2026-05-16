"""
Interactive Analytics Dashboard
================================
Run: python dashboard/app.py

Displays:
  - Identity Graph Overview (match rate, cross-device breakdown)
  - Identity Model Metrics (accuracy, precision, recall, F1, ROC-AUC)
  - Feature Importance (top signals driving the match decision)
  - Audience Segment Explorer (size, behavioural profiles, device mix)
  - Session Browser (filterable data table)
"""

from __future__ import annotations

import json
import logging
import os
import sys

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, dcc, html, dash_table

logger = logging.getLogger(__name__)

# ── Load data from DB ─────────────────────────────────────────────────────

def _load_data():
    """Load processed data from SQLite; fall back to sample CSV if not ready."""
    try:
        import yaml
        from sqlalchemy import create_engine, text
        with open("config/config.yaml") as f:
            cfg = yaml.safe_load(f)
        db_path = cfg.get("database", {}).get("sqlite_path", "data/processed/sessions.db")
        engine = create_engine(f"sqlite:///{db_path}")
        with engine.connect() as conn:
            sessions = pd.read_sql(text("SELECT * FROM sessions LIMIT 20000"), conn)
            try:
                segments = pd.read_sql(text("SELECT * FROM audience_segments"), conn)
            except Exception:
                segments = pd.DataFrame()
    except Exception as e:
        logger.warning("DB not ready (%s) — loading sample CSV.", e)
        sample_path = "data/sample/sessions_sample.csv"
        if os.path.exists(sample_path):
            sessions = pd.read_csv(sample_path)
        else:
            sessions = _make_demo_data()
        segments = pd.DataFrame()

    return sessions, segments


def _make_demo_data() -> pd.DataFrame:
    """Tiny demo DataFrame so the dashboard opens even before pipeline runs."""
    import numpy as np
    rng = np.random.default_rng(42)
    n = 2000
    devices = rng.choice(["mobile", "desktop", "tablet"], size=n, p=[0.5, 0.35, 0.15])
    return pd.DataFrame(
        {
            "session_id": [f"demo_{i}" for i in range(n)],
            "device_type": devices,
            "scroll_depth_avg": rng.uniform(0.2, 1.0, n).round(3),
            "click_count": rng.integers(0, 30, n),
            "session_duration_s": rng.integers(30, 900, n),
            "pages_visited": rng.integers(1, 15, n),
            "content_categories": ['["technology","gaming"]'] * n,
            "hour_of_day": rng.integers(0, 24, n),
        }
    )


SESSIONS, SEGMENTS = _load_data()

# ── Colour palette ────────────────────────────────────────────────────────

PALETTE = px.colors.qualitative.Set2
COLOR_MAP = {
    "mobile": "#F4845F",
    "desktop": "#4C72B0",
    "tablet": "#55A868",
}

# ── App layout ────────────────────────────────────────────────────────────

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    title="Cross-Device Identity Resolution Dashboard",
    suppress_callback_exceptions=True,
)

# KPI card helper
def kpi_card(title: str, value: str, colour: str = "#4C72B0", icon: str = "📊") -> dbc.Card:
    return dbc.Card(
        dbc.CardBody(
            [
                html.H5(f"{icon} {title}", className="card-title text-muted mb-1", style={"fontSize": "0.82rem"}),
                html.H3(value, style={"color": colour, "fontWeight": 700}),
            ]
        ),
        className="shadow-sm",
        style={"borderTop": f"4px solid {colour}"},
    )


app.layout = dbc.Container(
    [
        # ── Header ───────────────────────────────────────────────────────
        dbc.Row(
            dbc.Col(
                html.Div(
                    [
                        html.H2(
                            "Cross-Device Identity Resolution & Audience Segmentation",
                            className="mb-0 fw-bold",
                        ),
                        html.P(
                            "Privacy-preserving probabilistic user matching · NLP contextual targeting · K-Means audience segments",
                            className="text-muted",
                        ),
                    ],
                    className="py-3 border-bottom mb-4",
                )
            )
        ),

        # ── KPI row ───────────────────────────────────────────────────────
        dbc.Row(
            [
                dbc.Col(kpi_card("Total Sessions", f"{len(SESSIONS):,}", "#4C72B0", "📋"), md=3),
                dbc.Col(kpi_card("Unique Users", f"{SESSIONS['user_id_hash'].nunique():,}" if 'user_id_hash' in SESSIONS.columns else "—", "#55A868", "👥"), md=3),
                dbc.Col(kpi_card("Audience Segments", str(len(SEGMENTS)) if len(SEGMENTS) > 0 else "Run pipeline", "#F4845F", "🎯"), md=3),
                dbc.Col(kpi_card("Device Types", "3 (Mobile / Desktop / Tablet)", "#8172B3", "📱"), md=3),
            ],
            className="mb-4 g-3",
        ),

        # ── Tabs ─────────────────────────────────────────────────────────
        dbc.Tabs(
            [
                dbc.Tab(label="📱 Session Overview", tab_id="tab-sessions"),
                dbc.Tab(label="🎯 Audience Segments", tab_id="tab-segments"),
                dbc.Tab(label="🔍 Behaviour Analysis", tab_id="tab-behaviour"),
                dbc.Tab(label="📋 Data Table", tab_id="tab-table"),
            ],
            id="tabs",
            active_tab="tab-sessions",
            className="mb-3",
        ),

        html.Div(id="tab-content"),
    ],
    fluid=True,
    className="p-4",
)


# ── Tab content ───────────────────────────────────────────────────────────

@app.callback(Output("tab-content", "children"), Input("tabs", "active_tab"))
def render_tab(tab: str):
    if tab == "tab-sessions":
        return _tab_sessions()
    elif tab == "tab-segments":
        return _tab_segments()
    elif tab == "tab-behaviour":
        return _tab_behaviour()
    elif tab == "tab-table":
        return _tab_table()
    return html.P("Select a tab above.")


def _tab_sessions():
    # Device distribution pie
    dev_counts = SESSIONS["device_type"].value_counts().reset_index()
    dev_counts.columns = ["Device", "Count"]
    fig_pie = px.pie(
        dev_counts, values="Count", names="Device",
        color="Device", color_discrete_map=COLOR_MAP,
        title="Session Distribution by Device",
        hole=0.4,
    )
    fig_pie.update_layout(height=350, margin=dict(t=50, b=10))

    # Hourly activity heatmap (if hour_of_day available)
    figs_hour = go.Figure()
    if "hour_of_day" in SESSIONS.columns and "device_type" in SESSIONS.columns:
        for dev in ["mobile", "desktop", "tablet"]:
            subset = SESSIONS[SESSIONS["device_type"] == dev]
            if not subset.empty:
                counts = subset["hour_of_day"].value_counts().sort_index()
                figs_hour.add_trace(
                    go.Scatter(
                        x=counts.index, y=counts.values,
                        mode="lines+markers", name=dev,
                        line=dict(color=COLOR_MAP[dev], width=2),
                    )
                )
    figs_hour.update_layout(
        title="Hourly Session Activity by Device",
        xaxis_title="Hour of Day",
        yaxis_title="Session Count",
        height=350,
        margin=dict(t=50, b=30),
        legend=dict(orientation="h", y=-0.15),
    )

    # Cross-device users bar
    if "user_id_hash" in SESSIONS.columns:
        user_devices = (
            SESSIONS.groupby("user_id_hash")["device_type"]
            .nunique()
            .value_counts()
            .reset_index()
        )
        user_devices.columns = ["Devices Used", "Users"]
        fig_cross = px.bar(
            user_devices, x="Devices Used", y="Users",
            title="Users by Number of Distinct Devices",
            color_discrete_sequence=["#4C72B0"],
        )
        fig_cross.update_layout(height=350, margin=dict(t=50, b=30))
    else:
        fig_cross = go.Figure()

    return dbc.Row(
        [
            dbc.Col(dcc.Graph(figure=fig_pie), md=4),
            dbc.Col(dcc.Graph(figure=figs_hour), md=4),
            dbc.Col(dcc.Graph(figure=fig_cross), md=4),
        ],
        className="g-3",
    )


def _tab_segments():
    if SEGMENTS.empty:
        return dbc.Alert(
            "No segment data found. Run the full pipeline first: python pipeline.py",
            color="warning",
        )

    df = SEGMENTS.reset_index() if "segment_id" not in SEGMENTS.columns else SEGMENTS.copy()

    # Segment size bar chart
    fig_size = px.bar(
        df.sort_values("n_sessions", ascending=True),
        x="n_sessions", y="segment_label",
        orientation="h",
        title="Sessions per Audience Segment",
        labels={"n_sessions": "Sessions", "segment_label": "Segment"},
        color="pct_mobile",
        color_continuous_scale="RdYlGn_r",
        color_continuous_midpoint=50,
    )
    fig_size.update_layout(height=max(350, len(df) * 40), margin=dict(t=50, b=30))

    # Behavioural radar for top 5 segments
    top5 = df.nlargest(5, "n_sessions")
    fig_radar = go.Figure()
    metrics_cols = ["avg_scroll_depth", "avg_click_count", "avg_pages_visited"]
    for _, row in top5.iterrows():
        vals = [row[c] for c in metrics_cols]
        # Normalise to 0-1 for radar
        max_vals = [1.0, 30.0, 15.0]
        vals_norm = [min(v / m, 1.0) for v, m in zip(vals, max_vals)]
        fig_radar.add_trace(
            go.Scatterpolar(
                r=vals_norm + vals_norm[:1],
                theta=["Scroll Depth", "Click Count", "Pages/Session", "Scroll Depth"],
                name=str(row["segment_label"])[:25],
                fill="toself",
                opacity=0.6,
            )
        )
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title="Behavioural Profile — Top 5 Segments",
        height=400,
        margin=dict(t=60, b=30),
    )

    # Segment table
    tbl_df = df[["segment_label", "n_sessions", "pct_of_total", "top_categories", "pct_mobile", "avg_session_duration_s"]].copy()
    tbl_df.columns = ["Label", "Sessions", "% of Total", "Top Categories", "% Mobile", "Avg Duration (s)"]

    seg_table = dash_table.DataTable(
        data=tbl_df.to_dict("records"),
        columns=[{"name": c, "id": c} for c in tbl_df.columns],
        style_table={"overflowX": "auto"},
        style_header={"backgroundColor": "#f8f9fa", "fontWeight": "bold"},
        style_data_conditional=[
            {"if": {"row_index": "odd"}, "backgroundColor": "#f8f9fa"},
        ],
        page_size=15,
        sort_action="native",
    )

    return html.Div(
        [
            dbc.Row(
                [
                    dbc.Col(dcc.Graph(figure=fig_size), md=7),
                    dbc.Col(dcc.Graph(figure=fig_radar), md=5),
                ],
                className="g-3 mb-3",
            ),
            html.H6("All Segments", className="fw-bold"),
            seg_table,
        ]
    )


def _tab_behaviour():
    figs = []

    # Scroll depth distribution by device
    fig1 = px.box(
        SESSIONS, x="device_type", y="scroll_depth_avg",
        color="device_type", color_discrete_map=COLOR_MAP,
        title="Scroll Depth by Device",
        labels={"scroll_depth_avg": "Scroll Depth (0–1)", "device_type": "Device"},
    )
    fig1.update_layout(height=380, margin=dict(t=50, b=30), showlegend=False)
    figs.append(dbc.Col(dcc.Graph(figure=fig1), md=4))

    # Session duration histogram
    fig2 = px.histogram(
        SESSIONS, x="session_duration_s", color="device_type",
        color_discrete_map=COLOR_MAP, nbins=50,
        title="Session Duration Distribution",
        labels={"session_duration_s": "Duration (s)"},
        barmode="overlay", opacity=0.7,
    )
    fig2.update_layout(height=380, margin=dict(t=50, b=30))
    figs.append(dbc.Col(dcc.Graph(figure=fig2), md=4))

    # Pages visited vs click count scatter
    sample = SESSIONS.sample(min(3000, len(SESSIONS)), random_state=42)
    fig3 = px.scatter(
        sample, x="pages_visited", y="click_count",
        color="device_type", color_discrete_map=COLOR_MAP,
        opacity=0.5, title="Pages Visited vs Click Count",
        labels={"pages_visited": "Pages Visited", "click_count": "Click Count"},
        trendline=None,
    )
    fig3.update_layout(height=380, margin=dict(t=50, b=30))
    figs.append(dbc.Col(dcc.Graph(figure=fig3), md=4))

    return dbc.Row(figs, className="g-3")


def _tab_table():
    display_cols = [
        c for c in ["session_id", "device_type", "os", "browser",
                     "session_duration_s", "pages_visited", "click_count",
                     "scroll_depth_avg", "content_categories", "segment_id"]
        if c in SESSIONS.columns
    ]
    display_df = SESSIONS[display_cols].head(5000).copy()
    # Truncate long strings
    if "content_categories" in display_df.columns:
        display_df["content_categories"] = display_df["content_categories"].apply(
            lambda x: x[:40] + "…" if isinstance(x, str) and len(x) > 40 else x
        )

    return dash_table.DataTable(
        data=display_df.to_dict("records"),
        columns=[{"name": c, "id": c} for c in display_df.columns],
        style_table={"overflowX": "auto"},
        style_header={"backgroundColor": "#f8f9fa", "fontWeight": "bold"},
        filter_action="native",
        sort_action="native",
        page_size=20,
        style_data={"fontSize": "0.85rem"},
    )


# ── Entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import yaml
    config_path = os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml")
    try:
        with open(config_path) as f:
            dash_cfg = yaml.safe_load(f).get("dashboard", {})
    except Exception:
        dash_cfg = {}

    logging.basicConfig(level=logging.INFO)
    logger.info("Starting dashboard at http://%s:%s", dash_cfg.get("host", "127.0.0.1"), dash_cfg.get("port", 8050))
    app.run(
        host=dash_cfg.get("host", "127.0.0.1"),
        port=dash_cfg.get("port", 8050),
        debug=dash_cfg.get("debug", True),
    )
