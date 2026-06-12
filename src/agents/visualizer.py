"""
Visualization Agent
───────────────────
Responsibilities:
  • Auto-select chart types based on column data types and cardinality.
  • Generate publication-quality charts using Matplotlib/Seaborn.
  • Encode each chart as a Base64 PNG string for embedding in the PDF.
  • Store all charts in state["visualizations"] keyed by a descriptive name.

Charts generated (adaptively):
  1. Missing Values Heatmap
  2. Numeric Distributions (histograms with KDE)
  3. Correlation Heatmap
  4. Box Plots (outlier overview)
  5. Categorical Bar Charts (top-10 value counts)
  6. Top Correlations Bar Chart
  7. Pairplot of top-5 numeric features (if ≤ 15 K rows)
"""

import io
import base64
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")           # non-interactive backend — safe for servers
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List

from src.core.state import AnalystState

warnings.filterwarnings("ignore")

# ─── Palette ──────────────────────────────────────────────────────────────────
PALETTE    = "coolwarm"
BAR_COLOR  = "#4C72B0"
GRID_COLOR = "#EEEEEE"
FIG_DPI    = 120

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.0)


# ─── Utility ──────────────────────────────────────────────────────────────────

def _fig_to_b64(fig: plt.Figure) -> str:
    """Save a matplotlib figure to a Base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=FIG_DPI)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    buf.close()
    return encoded


def _truncate_label(label: str, max_len: int = 20) -> str:
    return str(label)[:max_len] + "…" if len(str(label)) > max_len else str(label)


# ─── Chart generators ─────────────────────────────────────────────────────────

def _chart_missing_heatmap(df: pd.DataFrame) -> str:
    """Visual heatmap of missing values across the dataset."""
    null_pct = df.isnull().mean().sort_values(ascending=False)
    null_pct = null_pct[null_pct > 0]
    if null_pct.empty:
        # Generate a "No Missing Data" placeholder
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "OK  No missing values detected",
                ha="center", va="center", fontsize=14, color="green",
                transform=ax.transAxes)
        ax.axis("off")
        return _fig_to_b64(fig)

    fig, ax = plt.subplots(figsize=(10, max(3, len(null_pct) * 0.4)))
    null_pct.plot.barh(ax=ax, color="#E07B54")
    ax.set_xlabel("Missing %")
    ax.set_title("Missing Value Rate by Column", fontweight="bold")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.set_facecolor(GRID_COLOR)
    fig.tight_layout()
    return _fig_to_b64(fig)


def _chart_numeric_distributions(df: pd.DataFrame, num_cols: List[str]) -> str:
    """Grid of histograms + KDE for each numeric column (max 12)."""
    cols_to_plot = num_cols[:12]
    n = len(cols_to_plot)
    if n == 0:
        return ""
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows))
    axes = np.array(axes).flatten()
    for i, col in enumerate(cols_to_plot):
        series = df[col].dropna()
        axes[i].hist(series, bins=30, color=BAR_COLOR, alpha=0.7, edgecolor="white")
        ax2 = axes[i].twinx()
        series.plot.kde(ax=ax2, color="#E07B54", linewidth=1.5)
        ax2.set_ylabel("")
        ax2.set_yticks([])
        axes[i].set_title(col, fontweight="bold", fontsize=9)
        axes[i].set_xlabel("")
        axes[i].set_facecolor(GRID_COLOR)
    # hide unused subplots
    for j in range(n, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Numeric Column Distributions", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    return _fig_to_b64(fig)


def _chart_correlation_heatmap(df: pd.DataFrame, num_cols: List[str]) -> str:
    """Annotated Pearson correlation heatmap."""
    cols = num_cols[:15]   # cap at 15 for readability
    if len(cols) < 2:
        return ""
    corr = df[cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(max(6, len(cols) * 0.8), max(5, len(cols) * 0.7)))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f",
        cmap=PALETTE, center=0, linewidths=0.5,
        annot_kws={"size": 8}, ax=ax,
    )
    ax.set_title("Pearson Correlation Heatmap", fontweight="bold")
    fig.tight_layout()
    return _fig_to_b64(fig)


def _chart_boxplots(df: pd.DataFrame, num_cols: List[str]) -> str:
    """Box plots for all numeric columns to visualise spread and outliers."""
    cols = num_cols[:12]
    if not cols:
        return ""
    ncols = min(3, len(cols))
    nrows = (len(cols) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows))
    axes = np.array(axes).flatten()
    for i, col in enumerate(cols):
        series = df[col].dropna()
        axes[i].boxplot(series, vert=True, patch_artist=True,
                        boxprops=dict(facecolor="#4C72B0", alpha=0.6),
                        medianprops=dict(color="#E07B54", linewidth=2))
        axes[i].set_title(col, fontweight="bold", fontsize=9)
        axes[i].set_facecolor(GRID_COLOR)
    for j in range(len(cols), len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Box Plots — Spread & Outliers", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return _fig_to_b64(fig)


def _chart_categorical_bars(df: pd.DataFrame, cat_cols: List[str]) -> Dict[str, str]:
    """Bar chart for each categorical column's top-10 value counts."""
    charts: Dict[str, str] = {}
    for col in cat_cols[:6]:   # cap at 6 categorical charts
        vc = df[col].value_counts(dropna=True).head(10)
        if vc.empty:
            continue
        labels = [_truncate_label(str(k)) for k in vc.index]
        fig, ax = plt.subplots(figsize=(8, max(3, len(vc) * 0.45)))
        bars = ax.barh(labels[::-1], vc.values[::-1], color=BAR_COLOR, alpha=0.85)
        for bar, val in zip(bars, vc.values[::-1]):
            ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                    f"{val:,}", va="center", fontsize=8)
        ax.set_title(f"Top Values — '{col}'", fontweight="bold")
        ax.set_xlabel("Count")
        ax.set_facecolor(GRID_COLOR)
        fig.tight_layout()
        charts[f"bar_{col}"] = _fig_to_b64(fig)
    return charts


def _chart_top_correlations(top_corrs: List[Dict[str, Any]]) -> str:
    """Horizontal bar chart of the top correlated pairs."""
    if not top_corrs:
        return ""
    pairs = [f"{c['col_a']} ↔ {c['col_b']}" for c in top_corrs[:10]]
    values = [c["correlation"] for c in top_corrs[:10]]
    colors = ["#E07B54" if v >= 0 else "#4C72B0" for v in values]
    fig, ax = plt.subplots(figsize=(8, max(3, len(pairs) * 0.5)))
    ax.barh(pairs[::-1], values[::-1], color=colors[::-1], alpha=0.85)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlim(-1, 1)
    ax.set_xlabel("Pearson r")
    ax.set_title("Top Feature Correlations", fontweight="bold")
    ax.set_facecolor(GRID_COLOR)
    fig.tight_layout()
    return _fig_to_b64(fig)


# ─── Main node ────────────────────────────────────────────────────────────────

def visualizer_node(state: AnalystState) -> Dict[str, Any]:
    """
    Visualization Node.

    Generates a complete set of charts from the cleaned DataFrame
    and stores them as Base64 PNG strings in state['visualizations'].
    """
    print("\n========== VISUALIZER AGENT ==========")

    df = state.get("dataset")
    statistics = state.get("statistics", {})

    if df is None:
        msg = "[VISUALIZER] No dataset found."
        print(msg)
        return {
            "error_count": state.get("error_count", 0) + 1,
            "error_log": state.get("error_log", []) + [msg],
        }

    num_cols = list(df.select_dtypes(include="number").columns)
    cat_cols = list(df.select_dtypes(include=["object", "category"]).columns)
    top_corrs = statistics.get("top_correlations", [])

    charts: Dict[str, str] = {}

    # 1. Missing value heatmap
    charts["missing_heatmap"] = _chart_missing_heatmap(df)
    print("[VISUALIZER] OK Missing heatmap")

    # 2. Numeric distributions
    if num_cols:
        dist_chart = _chart_numeric_distributions(df, num_cols)
        if dist_chart:
            charts["numeric_distributions"] = dist_chart
            print("[VISUALIZER] OK Numeric distributions")

    # 3. Correlation heatmap
    if len(num_cols) >= 2:
        corr_chart = _chart_correlation_heatmap(df, num_cols)
        if corr_chart:
            charts["correlation_heatmap"] = corr_chart
            print("[VISUALIZER] OK Correlation heatmap")

    # 4. Box plots
    if num_cols:
        box_chart = _chart_boxplots(df, num_cols)
        if box_chart:
            charts["boxplots"] = box_chart
            print("[VISUALIZER] OK Box plots")

    # 5. Categorical bar charts
    if cat_cols:
        cat_charts = _chart_categorical_bars(df, cat_cols)
        charts.update(cat_charts)
        print(f"[VISUALIZER] OK {len(cat_charts)} categorical bar chart(s)")

    # 6. Top correlations bar
    if top_corrs:
        corr_bar = _chart_top_correlations(top_corrs)
        if corr_bar:
            charts["top_correlations_bar"] = corr_bar
            print("[VISUALIZER] OK Top correlations bar chart")

    print(f"[VISUALIZER] Total charts generated: {len(charts)}")
    return {"visualizations": charts, "error_count": 0}
