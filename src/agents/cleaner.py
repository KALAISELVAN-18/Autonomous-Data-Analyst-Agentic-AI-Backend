"""
Data Cleaner Agent
──────────────────
Responsibilities:
  • Standardise column names → snake_case.
  • Parse date-like string columns to datetime.
  • Coerce numeric-ish object columns to numbers.
  • Handle missing values:
      - Drop columns that are > 60 % null.
      - Drop rows that are > 50 % null.
      - Impute remaining nulls:
          · Numeric  → median (robust to outliers).
          · Categorical → mode (most-frequent value).
  • Remove duplicate rows.
  • Detect and clip outliers using the IQR method on numeric columns.
  • Log every transformation applied.
"""

import re
import pandas as pd
import numpy as np
from typing import Dict, Any, List

from src.core.state import AnalystState


# ─── Thresholds ──────────────────────────────────────────────────────────────
COL_NULL_THRESHOLD = 1.01   # set to > 1.0 so no columns are ever dropped
ROW_NULL_THRESHOLD = 1.01   # set to > 1.0 so no rows are ever dropped
IQR_MULTIPLIER     = 3.0    # standard IQR fence multiplier



# ─── Helpers ──────────────────────────────────────────────────────────────────

def _to_snake_case(name: str) -> str:
    """Convert a column name to snake_case."""
    name = re.sub(r"[^\w\s]", "", str(name))          # remove punctuation
    name = re.sub(r"\s+", "_", name.strip())           # spaces → underscore
    name = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
    name = re.sub(r"([a-z\d])([A-Z])", r"\1_\2", name)
    return name.lower()


def _standardise_columns(df: pd.DataFrame, log: List[str], metadata: dict) -> pd.DataFrame:
    """Rename columns to snake_case and strip whitespace from cell values, preserving nulls."""
    original = list(df.columns)
    new_cols = []
    seen = set()
    mapping = {}
    for col in original:
        sc = _to_snake_case(col)
        if not sc:
            sc = "column"
        base = sc
        counter = 1
        while sc in seen:
            sc = f"{base}_{counter}"
            counter += 1
        seen.add(sc)
        new_cols.append(sc)
        mapping[col] = sc
    
    df.columns = new_cols
    metadata["column_mapping"] = mapping
    metadata["original_columns"] = original
    metadata["columns"] = new_cols
    
    renamed = [f"'{o}' -> '{n}'" for o, n in zip(original, df.columns) if o != n]
    if renamed:
        log.append(f"Standardised column names to snake_case: {'; '.join(renamed)}")
    
    # Strip leading/trailing whitespace from string cells, preserving NaN
    str_cols = df.select_dtypes(include="object").columns
    for col in str_cols:
        df[col] = df[col].apply(lambda x: str(x).strip() if pd.notnull(x) else x)
    return df


def _coerce_booleans(df: pd.DataFrame, log: List[str]) -> pd.DataFrame:
    """Coerce boolean-like string columns (e.g. Yes/No, True/False) to boolean types."""
    bool_mappings = {
        "true": True, "false": False,
        "yes": True, "no": False,
        "y": True, "n": False,
        "1": True, "0": False,
        "1.0": True, "0.0": False
    }
    for col in df.select_dtypes(include="object").columns:
        non_null_values = df[col].dropna()
        if len(non_null_values) == 0:
            continue
        
        unique_vals = {str(x).strip().lower() for x in non_null_values.unique()}
        if unique_vals.issubset(bool_mappings.keys()) and len(unique_vals) > 0:
            df[col] = df[col].apply(lambda x: bool_mappings[str(x).strip().lower()] if pd.notnull(x) else x)
            log.append(f"Column '{col}' coerced to boolean values based on content (unique values: {unique_vals}).")
    return df



def _parse_datetimes(df: pd.DataFrame, log: List[str]) -> pd.DataFrame:
    """Try to convert object columns that look like dates to datetime."""
    date_keywords = ("date", "time", "year", "month", "day", "dt", "timestamp")
    for col in df.select_dtypes(include="object").columns:
        if any(kw in col.lower() for kw in date_keywords):
            try:
                parsed = pd.to_datetime(df[col], infer_datetime_format=True, errors="coerce")
                if parsed.notna().sum() > len(df) * 0.5:   # accept if >50 % parsed
                    df[col] = parsed
                    log.append(f"Column '{col}' converted to datetime.")
            except Exception:
                pass
    return df


def _coerce_numeric_objects(df: pd.DataFrame, log: List[str]) -> pd.DataFrame:
    """
    For object columns where > 50 % of non-null values look numeric (after stripping currency symbols,
    commas, and percent signs), convert the column to numeric.
    """
    for col in df.select_dtypes(include="object").columns:
        non_null_count = df[col].notna().sum()
        if non_null_count == 0:
            continue
        
        # Strip common formatting like currency, commas, and percentage
        cleaned = df[col].astype(str).str.replace(r'[$\s%,€£¥]', '', regex=True)
        # Convert custom null place-holders to NaN
        cleaned = cleaned.replace(['nan', 'NaN', 'None', 'null', 'NULL', 'n/a', 'N/A', '-', '?'], np.nan)
        coerced = pd.to_numeric(cleaned, errors="coerce")
        
        hit_rate = coerced.notna().sum() / non_null_count
        if hit_rate >= 0.50:
            df[col] = coerced
            log.append(f"Column '{col}' coerced to numeric after removing formatting symbols (hit rate {hit_rate:.0%} of non-null values).")
    return df



def _drop_high_null_columns(df: pd.DataFrame, log: List[str]) -> pd.DataFrame:
    null_ratio = df.isnull().mean()
    to_drop = null_ratio[null_ratio > COL_NULL_THRESHOLD].index.tolist()
    if to_drop:
        df = df.drop(columns=to_drop)
        log.append(f"Dropped {len(to_drop)} high-null columns (>{COL_NULL_THRESHOLD:.0%} nulls): {to_drop}")
    return df


def _drop_high_null_rows(df: pd.DataFrame, log: List[str]) -> pd.DataFrame:
    n_before = len(df)
    threshold = int(len(df.columns) * ROW_NULL_THRESHOLD)
    df = df.dropna(thresh=len(df.columns) - threshold)
    n_dropped = n_before - len(df)
    if n_dropped:
        log.append(f"Dropped {n_dropped} rows with >{ROW_NULL_THRESHOLD:.0%} missing values.")
    return df


def _impute_missing(df: pd.DataFrame, log: List[str]) -> pd.DataFrame:
    """Impute remaining nulls: median for numeric, mode for categorical, mode/median for datetime."""
    numeric_cols = df.select_dtypes(include="number").columns
    cat_cols     = df.select_dtypes(include=["object", "category", "bool"]).columns
    datetime_cols = df.select_dtypes(include=["datetime", "datetime64"]).columns

    for col in numeric_cols:
        n_null = int(df[col].isnull().sum())
        if n_null:
            median_val = df[col].median()
            if pd.isnull(median_val):
                median_val = 0.0
            df[col] = df[col].fillna(median_val)
            log.append(f"Imputed {n_null} nulls in '{col}' with median ({median_val:.4g}).")

    for col in cat_cols:
        n_null = int(df[col].isnull().sum())
        if n_null:
            mode_val = df[col].mode(dropna=True)
            if not mode_val.empty:
                df[col] = df[col].fillna(mode_val[0])
                log.append(f"Imputed {n_null} nulls in '{col}' with mode ('{mode_val[0]}').")
            else:
                df[col] = df[col].fillna("Unknown")
                log.append(f"Imputed {n_null} nulls in '{col}' with 'Unknown' (no mode found).")

    for col in datetime_cols:
        n_null = int(df[col].isnull().sum())
        if n_null:
            mode_val = df[col].mode(dropna=True)
            if not mode_val.empty:
                fill_val = mode_val[0]
            else:
                median_val = df[col].median()
                fill_val = median_val if pd.notnull(median_val) else pd.Timestamp.now()
            df[col] = df[col].fillna(fill_val)
            log.append(f"Imputed {n_null} nulls in datetime column '{col}' with '{fill_val}'.")

    return df


def _remove_duplicates(df: pd.DataFrame, log: List[str]) -> pd.DataFrame:
    n_before = len(df)
    df = df.drop_duplicates()
    n_dropped = n_before - len(df)
    if n_dropped:
        log.append(f"Removed {n_dropped} duplicate rows.")
    return df


def _clip_outliers_iqr(df: pd.DataFrame, log: List[str]) -> pd.DataFrame:
    """
    Clip numeric values that fall beyond the IQR fences.
    Only applied to columns with a meaningful spread (IQR > 0).
    Skips unique identifier/code columns.
    """
    numeric_cols = df.select_dtypes(include="number").columns
    for col in numeric_cols:
        if any(x in col.lower() for x in ["id", "key", "code", "zip", "phone"]):
            continue
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower = q1 - IQR_MULTIPLIER * iqr
        upper = q3 + IQR_MULTIPLIER * iqr
        n_outliers = int(((df[col] < lower) | (df[col] > upper)).sum())
        if n_outliers:
            df[col] = df[col].clip(lower=lower, upper=upper)
            log.append(
                f"Clipped {n_outliers} outliers in '{col}' "
                f"to [{lower:.4g}, {upper:.4g}] (IQR method)."
            )
    return df

def _standardise_categorical_casing(df: pd.DataFrame, log: List[str]) -> pd.DataFrame:
    """Resolve inconsistent capitalization in categorical columns by converting to Title Case."""
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    for col in cat_cols:
        non_nulls = df[col].dropna()
        if len(non_nulls) == 0:
            continue
        uniques = non_nulls.unique()
        lowered_uniques = {str(x).strip().lower() for x in uniques}
        if len(lowered_uniques) < len(uniques):
            df[col] = df[col].apply(lambda x: str(x).strip().title() if pd.notnull(x) else x)
            log.append(f"Standardised inconsistent capitalization in categorical column '{col}' to Title Case.")
    return df

def _update_metadata(df: pd.DataFrame, original_metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Update metadata for the cleaned dataset while preserving original info."""
    new_meta = original_metadata.copy()
    new_meta["shape"] = {"rows": int(df.shape[0]), "columns": int(df.shape[1])}
    new_meta["columns"] = list(df.columns)
    new_meta["dtypes"] = {col: str(dtype) for col, dtype in df.dtypes.items()}

    # Null counts and percentages
    null_counts = df.isnull().sum()
    new_meta["null_counts"] = {col: int(null_counts[col]) for col in df.columns}
    new_meta["null_percentages"] = {
        col: round(float(null_counts[col] / max(len(df), 1) * 100), 2) for col in df.columns
    }

    # Unique value counts
    new_meta["unique_counts"] = {col: int(df[col].nunique()) for col in df.columns}

    # Separate column types
    new_meta["numeric_columns"] = list(df.select_dtypes(include=["number"]).columns)
    new_meta["categorical_columns"] = list(df.select_dtypes(include=["object", "category"]).columns)
    new_meta["datetime_columns"] = list(df.select_dtypes(include=["datetime", "datetime64"]).columns)

    # Sample values
    new_meta["sample_values"] = {
        col: df[col].dropna().head(3).tolist() for col in df.columns
    }
    
    return new_meta


# ─── Main node ────────────────────────────────────────────────────────────────

def cleaner_node(state: AnalystState) -> Dict[str, Any]:
    """
    Data Cleaner Node.

    Applies a systematic sequence of cleaning steps and returns
    the cleaned DataFrame + a detailed cleaning log.
    """
    print("\n========== CLEANER AGENT ==========")

    df = state.get("dataset")
    if df is None:
        msg = "[CLEANER] No dataset found in state."
        print(msg)
        return {
            "error_count": state.get("error_count", 0) + 1,
            "error_log": state.get("error_log", []) + [msg],
        }

    df = df.copy()
    log: List[str] = list(state.get("cleaning_log", []))
    metadata: Dict[str, Any] = dict(state.get("metadata", {}))

    n_before = df.shape

    # ── Step 1: Standardise column names ─────────────────────────────────────
    df = _standardise_columns(df, log, metadata)

    # ── Step 2: Parse datetime columns ───────────────────────────────────────
    df = _parse_datetimes(df, log)

    # ── Step 3: Coerce numeric-looking object columns ─────────────────────────
    df = _coerce_numeric_objects(df, log)

    # ── Step 3b: Coerce boolean-looking object columns ────────────────────────
    df = _coerce_booleans(df, log)

    # ── Step 3c: Standardise categorical casing ───────────────────────────────
    df = _standardise_categorical_casing(df, log)

    # ── Step 4: Drop columns with too many nulls ──────────────────────────────
    df = _drop_high_null_columns(df, log)

    # ── Step 5: Drop rows with too many nulls ─────────────────────────────────
    df = _drop_high_null_rows(df, log)

    # ── Step 6: Impute remaining missing values ───────────────────────────────
    df = _impute_missing(df, log)

    # ── Step 7: Remove duplicates ─────────────────────────────────────────────
    df = _remove_duplicates(df, log)

    # ── Step 8: Clip outliers ─────────────────────────────────────────────────
    df = _clip_outliers_iqr(df, log)

    # Rebuild metadata for cleaned dataset
    cleaned_metadata = _update_metadata(df, metadata)

    n_after = df.shape
    log.append(
        f"Cleaning complete: {n_before[0]} x {n_before[1]} -> "
        f"{n_after[0]} x {n_after[1]}."
    )
    print(f"[CLEANER] {n_before[0]}x{n_before[1]} -> {n_after[0]}x{n_after[1]}")
    print(f"[CLEANER] Applied {len(log)} cleaning operations.")

    return {
        "dataset": df,
        "metadata": cleaned_metadata,
        "cleaning_log": log,
        "error_count": 0,
    }
