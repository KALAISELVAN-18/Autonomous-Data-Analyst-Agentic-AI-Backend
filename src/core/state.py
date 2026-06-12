from typing import TypedDict, Annotated, List, Dict, Any, Optional
import operator
import pandas as pd
from langchain_core.messages import BaseMessage


class AnalystState(TypedDict):
    # Conversation history
    messages: Annotated[List[BaseMessage], operator.add]

    # Core data artifacts
    dataset: Optional[Any]           # pandas DataFrame (stored as Any to avoid TypedDict issues)
    raw_dataset: Optional[Any]       # Original unmodified copy
    file_path: Optional[str]
    file_name: Optional[str]

    # Profiling & metadata
    metadata: Dict[str, Any]         # Schema, dtype map, shape, missing counts
    cleaning_log: List[str]          # Human-readable cleaning steps

    # Analysis outputs
    statistics: Dict[str, Any]       # Descriptive stats, correlations, hypothesis tests

    # Visualization artifacts (chart_name -> base64 PNG string)
    visualizations: Dict[str, str]

    # LLM-generated content
    insights: str                    # Markdown narrative from Gemini
    recommendations: List[str]       # Bullet-point recommendations

    # Report
    report_path: Optional[str]       # Absolute path to generated PDF

    # Orchestration control
    current_plan: List[str]
    next_agent: str
    error_count: int
    error_log: List[str]
