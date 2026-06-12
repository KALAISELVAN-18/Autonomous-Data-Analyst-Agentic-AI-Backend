"""
Insight Generation Agent (Gemini-Powered)
──────────────────────────────────────────
Responsibilities:
  • Send a structured analytical summary to Gemini 2.5 Flash.
  • Ask the model to produce:
      - Executive Summary (2-3 sentences)
      - Key Findings (bullet points)
      - Data Quality Notes
      - Business Recommendations (actionable bullets)
  • Parse the LLM response into structured state fields.
"""

import json
from typing import Dict, Any, List

from langchain_core.messages import HumanMessage, SystemMessage

from src.core.state import AnalystState
from src.utils.llm_factory import get_llm


SYSTEM_PROMPT = """
You are a Senior Data Analyst and Business Intelligence Expert.
You will be given a structured JSON summary of a dataset analysis.
Your job is to produce a clear, professional analytical report in the following exact format:

## Executive Summary
<2-3 sentence high-level overview of the dataset and its most important characteristics>

## Key Findings
- <finding 1>
- <finding 2>
- <finding 3>
- <finding 4>
- <finding 5>

## Data Quality Notes
- <note about data quality, cleaning steps taken, or reliability concerns>
- <additional note if applicable>

## Business Recommendations
- <specific, actionable recommendation 1>
- <specific, actionable recommendation 2>
- <specific, actionable recommendation 3>

Ensure each section is complete, factual (based only on the provided data), and written for a business audience.
Do NOT include any code or raw numbers. Explain them in natural language.
""".strip()


def _build_analysis_summary(state: AnalystState) -> str:
    """Build a concise JSON summary payload to send to Gemini."""
    metadata   = state.get("metadata", {})
    statistics = state.get("statistics", {})
    clean_log  = state.get("cleaning_log", [])

    # Trim the payload to stay within context limits
    desc = statistics.get("descriptive", {})
    top_corrs = statistics.get("top_correlations", [])[:5]
    cat_stats = statistics.get("categorical_stats", {})
    skewed    = statistics.get("skewed_columns", [])

    summary = {
        "dataset_name": metadata.get("file_name", "Unknown"),
        "shape": statistics.get("shape", metadata.get("shape", {})),
        "column_types": {
            "numeric": metadata.get("numeric_columns", []),
            "categorical": metadata.get("categorical_columns", []),
        },
        "cleaning_steps_applied": len(clean_log),
        "cleaning_log_excerpt": clean_log[-5:],      # last 5 steps
        "descriptive_stats_excerpt": {               # only mean/std/median for brevity
            col: {k: v for k, v in stats.items() if k in ("mean", "median", "std", "skewness")}
            for col, stats in list(desc.items())[:8]
        },
        "top_5_correlations": top_corrs,
        "skewed_columns": skewed,
        "categorical_highlights": {
            col: {"unique_count": info["unique_count"], "top_value": list(info["top_values"].keys())[:3]}
            for col, info in list(cat_stats.items())[:4]
        },
    }

    return json.dumps(summary, indent=2, default=str)


def _parse_sections(text: str) -> Dict[str, Any]:
    """
    Parse the LLM markdown response into structured fields:
      - insights   (full markdown text)
      - recommendations (list of strings)
    """
    recommendations: List[str] = []
    in_recommendations = False

    for line in text.splitlines():
        stripped = line.strip()
        if "## Business Recommendations" in stripped:
            in_recommendations = True
            continue
        if stripped.startswith("## ") and in_recommendations:
            in_recommendations = False
        if in_recommendations and stripped.startswith("- "):
            recommendations.append(stripped[2:].strip())

    return {
        "insights": text.strip(),
        "recommendations": recommendations,
    }


def insights_node(state: AnalystState) -> Dict[str, Any]:
    """
    Insight Generation Node.

    Sends the statistical analysis summary to Gemini 2.5 Flash
    and stores the structured narrative insights in the state.
    """
    print("\n========== INSIGHTS AGENT ==========")

    summary_payload = _build_analysis_summary(state)

    user_prompt = (
        f"Here is the dataset analysis summary in JSON format:\n\n"
        f"```json\n{summary_payload}\n```\n\n"
        f"Please generate the full analytical report following the format in your instructions."
    )

    try:
        llm = get_llm(temperature=0.3)
        response = llm.invoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ])
        raw_text = response.content
        print(f"[INSIGHTS] Received {len(raw_text)} characters from Gemini.")
    except Exception as exc:
        msg = f"[INSIGHTS] LLM call failed: {exc}"
        print(msg)
        # Fallback: generate a basic insight from statistics
        raw_text = _fallback_insights(state)

    parsed = _parse_sections(raw_text)

    return {
        "insights": parsed["insights"],
        "recommendations": parsed["recommendations"],
        "error_count": 0,
    }


def _fallback_insights(state: AnalystState) -> str:
    """Generate rule-based insights if LLM call fails."""
    stats = state.get("statistics", {})
    desc = stats.get("descriptive", {})
    shape = stats.get("shape", {})
    skewed = stats.get("skewed_columns", [])
    top_corrs = stats.get("top_correlations", [])

    lines = [
        "## Executive Summary",
        f"The dataset contains {shape.get('rows', 'N/A')} rows and "
        f"{shape.get('columns', 'N/A')} columns. "
        "Analysis was completed using automated statistical methods.",
        "",
        "## Key Findings",
    ]
    for col, s in list(desc.items())[:5]:
        lines.append(
            f"- **{col}**: mean={s['mean']:.2f}, median={s['median']:.2f}, std={s['std']:.2f}"
        )
    lines += [
        "",
        "## Data Quality Notes",
        f"- {len(state.get('cleaning_log', []))} cleaning operations were applied.",
        f"- Skewed columns (|skew| > 1): {', '.join(skewed) if skewed else 'None'}.",
        "",
        "## Business Recommendations",
        "- Investigate highly skewed columns for potential log-transformation before modeling.",
        "- Review top correlated pairs to avoid multicollinearity in predictive models.",
        "- Validate data collection procedures to prevent future missing values.",
    ]
    return "\n".join(lines)
