"""
Statistical Analyzer Agent
──────────────────────────
Responsibilities:
  • Compute descriptive statistics (mean, median, std, IQR, skewness, kurtosis).
  • Build a correlation matrix for numeric columns.
  • Detect top correlated pairs.
  • Compute value distribution stats for categorical columns.
  • Identify skewed numeric distributions.
  • Run basic normality check (Shapiro-Wilk on small samples; D'Agostino on larger ones).
  • All outputs stored in state["statistics"] as a structured dict.
"""

import json
import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Any, List

from src.core.state import AnalystState


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _safe_float(val) -> float:
    """Convert a value to float, returning 0.0 if not possible."""
    try:
        v = float(val)
        return round(v, 4) if np.isfinite(v) else 0.0
    except Exception:
        return 0.0


def _descriptive_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """Per-column descriptive statistics for numeric columns."""
    numeric = df.select_dtypes(include="number")
    result = {}
    for col in numeric.columns:
        series = numeric[col].dropna()
        if len(series) == 0:
            continue
        result[col] = {
            "count":    int(series.count()),
            "mean":     _safe_float(series.mean()),
            "median":   _safe_float(series.median()),
            "std":      _safe_float(series.std()),
            "min":      _safe_float(series.min()),
            "max":      _safe_float(series.max()),
            "q1":       _safe_float(series.quantile(0.25)),
            "q3":       _safe_float(series.quantile(0.75)),
            "iqr":      _safe_float(series.quantile(0.75) - series.quantile(0.25)),
            "skewness": _safe_float(series.skew()),
            "kurtosis": _safe_float(series.kurtosis()),
        }
    return result


def _correlation_matrix(df: pd.DataFrame) -> Dict[str, Any]:
    """Pearson correlation matrix for numeric columns."""
    numeric = df.select_dtypes(include="number")
    if numeric.shape[1] < 2:
        return {}
    corr = numeric.corr(method="pearson")
    return {col: {c: _safe_float(corr.loc[col, c]) for c in corr.columns} for col in corr.index}


def _top_correlations(df: pd.DataFrame, n: int = 10) -> List[Dict[str, Any]]:
    """Return the top N correlated pairs (excluding self-correlations)."""
    numeric = df.select_dtypes(include="number")
    if numeric.shape[1] < 2:
        return []
    corr = numeric.corr(method="pearson").abs()
    pairs = []
    cols = list(corr.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            pairs.append({
                "col_a": cols[i],
                "col_b": cols[j],
                "correlation": _safe_float(corr.iloc[i, j]),
            })
    pairs.sort(key=lambda x: abs(x["correlation"]), reverse=True)
    return pairs[:n]


def _categorical_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """For each categorical column: top values and their frequencies."""
    result = {}
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    for col in cat_cols:
        vc = df[col].value_counts(dropna=True).head(10)
        result[col] = {
            "unique_count": int(df[col].nunique()),
            "top_values": {str(k): int(v) for k, v in vc.items()},
        }
    return result


def _normality_tests(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Test normality for each numeric column.
    Shapiro-Wilk for n ≤ 5000; D'Agostino-Pearson for larger samples.
    """
    numeric = df.select_dtypes(include="number")
    results = {}
    for col in numeric.columns:
        series = numeric[col].dropna()
        n = len(series)
        if n < 8:
            results[col] = {"test": "skipped", "reason": "too few samples"}
            continue
        try:
            if n <= 5000:
                stat, p = stats.shapiro(series)
                test_name = "shapiro-wilk"
            else:
                stat, p = stats.normaltest(series)
                test_name = "d'agostino-pearson"
            results[col] = {
                "test": test_name,
                "statistic": _safe_float(stat),
                "p_value": _safe_float(p),
                "is_normal": bool(p > 0.05),
            }
        except Exception as exc:
            results[col] = {"test": "error", "reason": str(exc)}
    return results


def _skewed_columns(descriptive: Dict[str, Any], threshold: float = 1.0) -> List[str]:
    """Return columns whose |skewness| exceeds the threshold."""
    return [col for col, s in descriptive.items() if abs(s.get("skewness", 0)) > threshold]


# ─── Main node ────────────────────────────────────────────────────────────────

def analyzer_node(state: AnalystState) -> Dict[str, Any]:
    """
    Statistical Analyzer Node.

    Computes a comprehensive statistics bundle from the cleaned DataFrame
    and stores it in state['statistics'].
    """
    print("\n========== ANALYZER AGENT ==========")

    df = state.get("dataset")
    if df is None:
        msg = "[ANALYZER] No dataset found in state."
        print(msg)
        return {
            "error_count": state.get("error_count", 0) + 1,
            "error_log": state.get("error_log", []) + [msg],
        }

    descriptive   = _descriptive_stats(df)
    corr_matrix   = _correlation_matrix(df)
    top_corrs     = _top_correlations(df)
    cat_stats     = _categorical_stats(df)
    normality     = _normality_tests(df)
    skewed_cols   = _skewed_columns(descriptive)

    statistics = {
        "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "descriptive": descriptive,
        "correlation_matrix": corr_matrix,
        "top_correlations": top_corrs,
        "categorical_stats": cat_stats,
        "normality_tests": normality,
        "skewed_columns": skewed_cols,
        "missing_after_clean": {col: int(df[col].isnull().sum()) for col in df.columns},
    }

    print(f"[ANALYZER] Computed stats for {len(descriptive)} numeric + {len(cat_stats)} categorical columns.")
    print(f"[ANALYZER] Skewed columns (|skew|>1): {skewed_cols}")

    return {"statistics": statistics, "error_count": 0}
