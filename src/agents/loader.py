"""
Data Loader Agent
─────────────────
Responsibilities:
  • Detect file encoding using chardet.
  • Load CSV or Excel files into a Pandas DataFrame.
  • Downcast numeric types to save memory.
  • Store both the raw original and working copy in state.
  • Build the initial metadata profile.
"""

import os
import chardet
import pandas as pd
from typing import Dict, Any

from src.core.state import AnalystState


def _detect_encoding(file_path: str) -> str:
    """Use chardet to sniff the encoding of a file."""
    with open(file_path, "rb") as f:
        raw = f.read(50_000)  # read first 50 KB for detection
    result = chardet.detect(raw)
    encoding = result.get("encoding") or "utf-8"
    print(f"[LOADER] Detected encoding: {encoding} (confidence: {result.get('confidence', 0):.0%})")
    return encoding


def _downcast_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Reduce memory footprint by downcasting numeric columns."""
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    return df


def _build_initial_metadata(df: pd.DataFrame, file_path: str) -> Dict[str, Any]:
    """Build a rich metadata profile of the raw loaded dataset."""
    metadata: Dict[str, Any] = {}

    metadata["file_name"] = os.path.basename(file_path)
    metadata["shape"] = {"rows": int(df.shape[0]), "columns": int(df.shape[1])}
    metadata["columns"] = list(df.columns)
    metadata["dtypes"] = {col: str(dtype) for col, dtype in df.dtypes.items()}

    # Null counts and percentages
    null_counts = df.isnull().sum()
    metadata["null_counts"] = {col: int(null_counts[col]) for col in df.columns}
    metadata["null_percentages"] = {
        col: round(float(null_counts[col] / len(df) * 100), 2) for col in df.columns
    }

    # Unique value counts
    metadata["unique_counts"] = {col: int(df[col].nunique()) for col in df.columns}

    # Separate column types
    metadata["numeric_columns"] = list(df.select_dtypes(include=["number"]).columns)
    metadata["categorical_columns"] = list(df.select_dtypes(include=["object", "category"]).columns)
    metadata["datetime_columns"] = list(df.select_dtypes(include=["datetime"]).columns)

    # Sample values for each column (first 3 non-null)
    metadata["sample_values"] = {
        col: df[col].dropna().head(3).tolist() for col in df.columns
    }

    return metadata


def loader_node(state: AnalystState) -> Dict[str, Any]:
    """
    Data Loader Node.

    Reads the file at `state['file_path']`, loads it into a DataFrame,
    downcasts types, and builds the initial metadata profile.
    """
    print("\n========== LOADER AGENT ==========")

    file_path = state.get("file_path", "")
    if not file_path or not os.path.exists(file_path):
        msg = f"[LOADER] File not found: '{file_path}'"
        print(msg)
        return {
            "error_count": state.get("error_count", 0) + 1,
            "error_log": state.get("error_log", []) + [msg],
        }

    try:
        ext = os.path.splitext(file_path)[1].lower()

        if ext == ".csv":
            encoding = _detect_encoding(file_path)
            # Try detected encoding; fall back to latin-1 if it fails
            try:
                df = pd.read_csv(file_path, encoding=encoding)
            except (UnicodeDecodeError, LookupError):
                print("[LOADER] Detected encoding failed; retrying with latin-1.")
                df = pd.read_csv(file_path, encoding="latin-1")

        elif ext in (".xlsx", ".xls"):
            df = pd.read_excel(file_path)

        else:
            msg = f"[LOADER] Unsupported file type: '{ext}'"
            print(msg)
            return {
                "error_count": state.get("error_count", 0) + 1,
                "error_log": state.get("error_log", []) + [msg],
            }

        # Downcast for memory efficiency
        df = _downcast_dataframe(df)

        print(f"[LOADER] Loaded dataset: {df.shape[0]} rows x {df.shape[1]} columns")

        # Build metadata
        metadata = _build_initial_metadata(df, file_path)

        return {
            "dataset": df,
            "raw_dataset": df.copy(),   # keep an untouched original
            "file_name": os.path.basename(file_path),
            "metadata": metadata,
            "cleaning_log": ["Dataset loaded successfully."],
            "error_count": 0,           # reset error count on success
        }

    except Exception as exc:
        msg = f"[LOADER] Unexpected error: {exc}"
        print(msg)
        return {
            "error_count": state.get("error_count", 0) + 1,
            "error_log": state.get("error_log", []) + [msg],
        }
