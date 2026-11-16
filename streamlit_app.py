"""Interactive Streamlit dashboard for Cellular Tower Frequency Optimization."""

import itertools
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ga_engine import (
    CLAUDE_PLAN,
    FREQUENCIES,
    OLD_RANDOM_PLAN,
    compute_distance_matrix,
    fitness_level,
    load_towers,
    run_ga,
)

st.set_page_config(
    page_title="GA Frequency Topology Optimizer",
    layout="wide",
)

FONT_PROFILES = {
    "Space Grotesk (Geometric Tech)": ("Space Grotesk", "https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&display=swap"),
    "Plus Jakarta Sans (Clean Modern)": ("Plus Jakarta Sans", "https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap"),
    "JetBrains Mono (Quant Terminal)": ("JetBrains Mono", "https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&display=swap"),
    "Merriweather (Editorial Serif)": ("Merriweather", "https://fonts.googleapis.com/css2?family=Merriweather:wght@400;700&display=swap"),
}

COLOR_PROFILES = {
    "Cyber Emerald (Dark)": {
        "bg": "#0b1114",
        "sidebar_bg": "#111a1f",
        "card_bg": "#142026",
        "card_border": "#1f353f",
        "text": "#e2f1f5",
        "muted": "#8ba8b2",
        "accent": "#10b981",
        "line_color": "#10b981",
        "bar_color": "#06b6d4",
        "palette": ["#10b981", "#06b6d4", "#f59e0b", "#a855f7", "#f43f5e", "#3b82f6"],
        "plotly_template": "plotly_dark",
    },
    "Midnight Quant (Dark)": {
        "bg": "#090d16",
        "sidebar_bg": "#101726",
        "card_bg": "#151e31",
        "card_border": "#263554",
        "text": "#f1f5f9",
        "muted": "#94a3b8",
        "accent": "#6366f1",
        "line_color": "#f43f5e",
        "bar_color": "#6366f1",
        "palette": ["#6366f1", "#38bdf8", "#f43f5e", "#eab308", "#22c55e", "#e879f9"],
        "plotly_template": "plotly_dark",
    },
    "Solar Warmth (Dark)": {
        "bg": "#14110f",
        "sidebar_bg": "#1d1916",
        "card_bg": "#26201c",
        "card_border": "#3d332c",
        "text": "#faf5f0",
        "muted": "#b8aaa0",
        "accent": "#f97316",
        "line_color": "#f97316",
        "bar_color": "#eab308",
        "palette": ["#f97316", "#eab308", "#14b8a6", "#ec4899", "#8b5cf6", "#84cc16"],
        "plotly_template": "plotly_dark",
    },
    "Crisp Slate (Light)": {
        "bg": "#f8fafc",
        "sidebar_bg": "#f1f5f9",
        "card_bg": "#ffffff",
        "card_border": "#e2e8f0",
        "text": "#0f172a",
        "muted": "#475569",
        "accent": "#2563eb",
        "line_color": "#dc2626",
        "bar_color": "#2563eb",
        "palette": ["#2563eb", "#0d9488", "#d97706", "#9333ea", "#e11d48", "#16a34a"],
        "plotly_template": "plotly_white",
    },
}

towers_df = load_towers()
eastings = towers_df["Easting"].to_numpy()
northings = towers_df["Northing"].to_numpy()
cell_ids = towers_df["Cell ID"].tolist()
distance_matrix = compute_distance_matrix(eastings, northings)

with st.sidebar:
    st.header("Appearance")
    selected_font_label = st.selectbox("Font Style", list(FONT_PROFILES.keys()), index=0)
    selected_theme_label = st.selectbox("Color Profile", list(COLOR_PROFILES.keys()), index=0)

    st.divider()
    st.header("GA Hyperparameters")
    pop_cap = st.slider("Population Size", min_value=50, max_value=1500, value=500, step=50)
    max_gen = st.slider("Generations", min_value=20, max_value=600, value=150, step=10)
    mutation_rate = st.slider("Mutation Rate", min_value=0.01, max_value=0.50, value=0.10, step=0.01)
    elite_size = st.slider("Elite Size", min_value=1, max_value=50, value=10, step=1)
    tournament_k = st.slider("Tournament Size (k)", min_value=2, max_value=15, value=5, step=1)
    restarts = st.slider("Independent GA Restarts", min_value=1, max_value=10, value=1, step=1)
    seed = st.number_input("Random Seed (0 = random)", min_value=0, max_value=99999, value=42, step=1)

    st.divider()
    show_links = st.checkbox("Show same-frequency links", value=True)
    run_clicked = st.button("Run Genetic Algorithm", type="primary", use_container_width=True)

font_family, font_url = FONT_PROFILES[selected_font_label]
theme = COLOR_PROFILES[selected_theme_label]

st.markdown(
    f"""
    <style>
    @import url('{font_url}');

    html, body, [class*="css"], .stApp, .stMarkdown, .stText, .stMetric {{
        font-family: '{font_family}', sans-serif !important;
        color: {theme["text"]};
    }}
    .stApp {{
        background-color: {theme["bg"]};
    }}
    section[data-testid="stSidebar"] {{
        background-color: {theme["sidebar_bg"]};
        border-right: 1px solid {theme["card_border"]};
    }}
    h1, h2, h3, [data-testid="stMetricValue"] {{
        font-family: '{font_family}', sans-serif !important;
        letter-spacing: -0.02em;
    }}
    div[data-testid="stMetric"] {{
        background-color: {theme["card_bg"]};
        border: 1px solid {theme["card_border"]};
        border-left: 4px solid {theme["accent"]};
        border-radius: 10px;
        padding: 14px 16px;
    }}
    div[data-testid="stMetricLabel"] p {{
        color: {theme["muted"]} !important;
        font-weight: 500;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Cellular Tower Frequency Distribution Optimizer")
st.caption("Genetic Algorithm minimizing inverse-square co-channel interference across 19 towers.")

if (
    "ga_result" not in st.session_state
    or run_clicked
    or not set(st.session_state["ga_result"]["best_setup"]).issubset(set(FREQUENCIES))
):
    best_result = None
    with st.spinner("Evolving tower frequency allocations..."):
        for i in range(restarts):
            run_seed = None if seed == 0 else int(seed) + i
            res = run_ga(
                distance_matrix=distance_matrix,
                freqs=FREQUENCIES,
                pop_cap=pop_cap,
                max_gen=max_gen,
                mutation_rate=mutation_rate,
                elite_size=elite_size,
                tournament_k=tournament_k,
                seed=run_seed,
            )
            if best_result is None or res["best_fitness"] > best_result["best_fitness"]:
                best_result = res
    st.session_state["ga_result"] = best_result

result = st.session_state["ga_result"]
best_setup = result["best_setup"]
ga_fitness = result["best_fitness"]
ga_penalty = result["best_penalty"]

claude_fitness = fitness_level(CLAUDE_PLAN, distance_matrix)
random_fitness = fitness_level(OLD_RANDOM_PLAN, distance_matrix)

# KPI Row
m1, m2, m3, m4 = st.columns(4)
m1.metric("Evolved GA Fitness", f"{ga_fitness:.8f}", delta=f"{ga_fitness - claude_fitness:+.8f} vs Claude")
m2.metric("Interference Penalty", f"{ga_penalty:.4e}")
m3.metric("Claude Baseline Fitness", f"{claude_fitness:.8f}")
m4.metric("Old Random Plan Fitness", f"{random_fitness:.8f}")

st.divider()

# Charts Row
col_map, col_conv = st.columns([1.2, 1.0])

def format_freq(freq_ghz: float) -> str:
    return f"{freq_ghz:.1f} GHz"


plot_df = pd.DataFrame(
    {
        "Cell ID": cell_ids,
        "Assigned Frequency": [format_freq(float(f)) for f in best_setup],
        "Easting (m)": eastings,
        "Northing (m)": northings,
    }
)

freq_order = [format_freq(f) for f in FREQUENCIES]
color_map = {freq_str: theme["palette"][i % len(theme["palette"])] for i, freq_str in enumerate(freq_order)}

with col_map:
    st.subheader("Optimized Tower Topology")
    fig_map = px.scatter(
        plot_df,
        x="Easting (m)",
        y="Northing (m)",
        text="Cell ID",
        color="Assigned Frequency",
        hover_name="Cell ID",
        category_orders={"Assigned Frequency": freq_order},
        color_discrete_map=color_map,
        template=theme["plotly_template"],
    )
    fig_map.update_traces(
        marker=dict(size=11, opacity=0.92),
        textposition="top center",
        selector=dict(mode="markers+text"),
    )

    if show_links:
        for freq in plot_df["Assigned Frequency"].unique():
            sub_df = plot_df[plot_df["Assigned Frequency"] == freq]
            if len(sub_df) >= 2:
                x_vals = sub_df["Easting (m)"].values
                y_vals = sub_df["Northing (m)"].values
                line_x, line_y = [], []
                for i, j in itertools.combinations(range(len(sub_df)), 2):
                    line_x.extend([x_vals[i], x_vals[j], None])
                    line_y.extend([y_vals[i], y_vals[j], None])
                fig_map.add_trace(
                    go.Scatter(
                        x=line_x,
                        y=line_y,
                        mode="lines",
                        line=dict(dash="dot", width=1.4, color=color_map.get(freq, theme["accent"])),
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )

    fig_map.update_yaxes(scaleanchor="x", scaleratio=1)
    fig_map.update_layout(
        height=520,
        margin=dict(l=10, r=10, t=20, b=10),
        paper_bgcolor=theme["card_bg"],
        plot_bgcolor=theme["card_bg"],
        font=dict(family=font_family, color=theme["text"]),
    )
    st.plotly_chart(fig_map, use_container_width=True)

with col_conv:
    st.subheader("GA Convergence (Interference Penalty)")
    conv_df = pd.DataFrame(
        {
            "Generation": np.arange(1, len(result["penalty_history"]) + 1),
            "Interference Penalty": result["penalty_history"],
        }
    )
    fig_conv = px.line(
        conv_df,
        x="Generation",
        y="Interference Penalty",
        log_y=True,
        template=theme["plotly_template"],
    )
    fig_conv.update_traces(line=dict(color=theme["line_color"], width=2.5))
    fig_conv.update_layout(
        height=270,
        margin=dict(l=10, r=10, t=20, b=10),
        paper_bgcolor=theme["card_bg"],
        plot_bgcolor=theme["card_bg"],
        font=dict(family=font_family, color=theme["text"]),
    )
    st.plotly_chart(fig_conv, use_container_width=True)

    st.subheader("Frequency Counts")
    freq_counts = (
        plot_df["Assigned Frequency"]
        .value_counts()
        .reindex(freq_order, fill_value=0)
        .reset_index()
    )
    freq_counts.columns = ["Frequency", "Towers"]
    fig_bar = px.bar(
        freq_counts,
        x="Frequency",
        y="Towers",
        text="Towers",
        color="Frequency",
        color_discrete_map=color_map,
        template=theme["plotly_template"],
    )
    fig_bar.update_layout(
        height=210,
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False,
        paper_bgcolor=theme["card_bg"],
        plot_bgcolor=theme["card_bg"],
        font=dict(family=font_family, color=theme["text"]),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

with st.expander("View Tower Allocation Table"):
    st.dataframe(plot_df, hide_index=True, use_container_width=True)
