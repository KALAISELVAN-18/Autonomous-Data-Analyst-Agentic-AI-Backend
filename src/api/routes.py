"""
API Routes
──────────
All endpoints for the Autonomous Data Analyst backend.

POST /upload      → Upload a CSV/XLSX dataset
POST /analyze     → Run the full multi-agent pipeline
POST /chat        → Ask a natural language question about the loaded dataset
GET  /report/{id} → Download the generated PDF report
"""

import os
import shutil
from typing import Optional

from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import FileResponse
from langchain_core.messages import HumanMessage

from src.core.config import settings
from src.core.orchestrator import analyst_graph
from src.core.state import AnalystState

router = APIRouter()

# In-memory session store (replace with Redis for production)
_sessions: dict = {}


@router.post("/upload", summary="Upload a CSV or Excel dataset")
async def upload_dataset(file: UploadFile = File(...)):
    """
    Upload a CSV or Excel file for analysis.
    Returns a session_id and the saved file path.
    """
    allowed = (".csv", ".xlsx", ".xls")
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {allowed}",
        )

    os.makedirs(settings.upload_dir, exist_ok=True)
    file_path = os.path.join(settings.upload_dir, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Create a new session
    session_id = f"{os.path.splitext(file.filename)[0]}_{os.urandom(4).hex()}"
    _sessions[session_id] = {
        "file_path": file_path,
        "file_name": file.filename,
        "state": None,
    }

    return {
        "message": "File uploaded successfully.",
        "session_id": session_id,
        "file_path": file_path,
        "file_name": file.filename,
    }


@router.post("/analyze", summary="Run the full autonomous analysis pipeline")
async def trigger_analysis(
    session_id: str = Query(..., description="Session ID returned by /upload"),
    prompt: Optional[str] = Query(
        "Perform a complete autonomous analysis on this dataset.",
        description="Optional user prompt",
    ),
):
    """
    Triggers the full LangGraph multi-agent pipeline for the uploaded dataset.

    Pipeline: Loader → Cleaner → Analyzer → Visualizer → Insights → Reporter

    Returns statistics, cleaning log, chart names, insights,
    recommendations, and the path to the generated PDF report.
    """
    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")

    file_path = session["file_path"]
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Uploaded file not found on disk.")

    # Build initial state
    initial_state: AnalystState = {
        "messages": [HumanMessage(content=prompt)],
        "dataset": None,
        "raw_dataset": None,
        "file_path": file_path,
        "file_name": session["file_name"],
        "metadata": {},
        "cleaning_log": [],
        "statistics": {},
        "visualizations": {},
        "insights": "",
        "recommendations": [],
        "report_path": None,
        "current_plan": [],
        "next_agent": "planner",
        "error_count": 0,
        "error_log": [],
    }

    # Execute the graph (synchronous stream)
    agents_run = []
    final_state = initial_state

    for chunk in analyst_graph.stream(initial_state):
        node_name = list(chunk.keys())[0]
        agents_run.append(node_name)
        # Merge state updates
        for k, v in chunk[node_name].items():
            final_state[k] = v

    # Persist final state to session
    _sessions[session_id]["state"] = final_state

    # Build response (exclude raw DataFrame for JSON serialisation)
    return {
        "message":        "Analysis complete.",
        "session_id":     session_id,
        "pipeline_steps": agents_run,
        "cleaning_log":   final_state.get("cleaning_log", []),
        "statistics": {
            "shape":             final_state.get("statistics", {}).get("shape", {}),
            "top_correlations":  final_state.get("statistics", {}).get("top_correlations", [])[:5],
            "skewed_columns":    final_state.get("statistics", {}).get("skewed_columns", []),
        },
        "chart_names":      list(final_state.get("visualizations", {}).keys()),
        "insights":         final_state.get("insights", ""),
        "recommendations":  final_state.get("recommendations", []),
        "report_path":      final_state.get("report_path"),
        "error_log":        final_state.get("error_log", []),
    }


@router.post("/chat", summary="Ask a natural language question about the dataset")
async def chat_with_data(
    session_id: str = Query(..., description="Session ID from /upload"),
    query: str = Query(..., description="Your question about the data"),
):
    """
    Routes a user query through the Dataset Chat Agent (Gemini text-to-Pandas).
    Requires an analysis session to have been run first.
    """
    session = _sessions.get(session_id)
    if not session or not session.get("state"):
        raise HTTPException(
            status_code=400,
            detail="Run /analyze first before chatting.",
        )

    state = dict(session["state"])
    state["messages"] = state.get("messages", []) + [HumanMessage(content=query)]

    # Directly call the chat node (no full pipeline re-run needed)
    from src.agents.chat import chat_node
    updates = chat_node(state)

    # Append response to session messages
    session["state"]["messages"] = state["messages"] + updates.get("messages", [])

    reply_messages = updates.get("messages", [])
    answer = reply_messages[-1].content if reply_messages else "No response generated."

    return {"session_id": session_id, "query": query, "answer": answer}


@router.get("/report/{session_id}", summary="Download the generated PDF report")
async def download_report(session_id: str):
    """
    Returns the generated PDF file for a completed analysis session.
    """
    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")

    state = session.get("state")
    if not state:
        raise HTTPException(status_code=400, detail="Analysis not yet run.")

    report_path = state.get("report_path")
    if not report_path or not os.path.exists(report_path):
        raise HTTPException(status_code=404, detail="Report not found. Run /analyze first.")

    return FileResponse(
        path=report_path,
        filename=os.path.basename(report_path),
        media_type="application/pdf",
    )


@router.get("/download-cleaned/{session_id}", summary="Download the cleaned dataset")
async def download_cleaned(
    session_id: str,
    format: str = Query("csv", description="Format to download: 'csv' or 'excel'"),
):
    """
    Returns the cleaned dataset for a completed session in CSV or Excel format.
    """
    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")

    state = session.get("state")
    if not state:
        raise HTTPException(status_code=400, detail="Analysis not yet run.")

    df = state.get("dataset")
    if df is None:
        raise HTTPException(status_code=404, detail="Cleaned dataset not found.")

    # Restore original column names for the download using the column mapping if available
    df = df.copy()
    metadata = state.get("metadata", {})
    mapping = metadata.get("column_mapping", {})
    if mapping:
        inv_mapping = {clean_col: orig_col for orig_col, clean_col in mapping.items()}
        df = df.rename(columns=inv_mapping)

    os.makedirs(os.path.join(settings.report_dir, "..", "exports"), exist_ok=True)
    export_dir = os.path.normpath(os.path.join(settings.report_dir, "..", "exports"))

    file_name = session.get("file_name", "dataset")
    base_name = os.path.splitext(file_name)[0]
    clean_name = base_name.replace(" ", "_")

    if format.lower() == "csv":
        export_path = os.path.join(export_dir, f"{clean_name}_cleaned.csv")
        df.to_csv(export_path, index=False)
        media_type = "text/csv"
        dl_filename = f"{clean_name}_cleaned.csv"
    elif format.lower() in ("excel", "xlsx"):
        export_path = os.path.join(export_dir, f"{clean_name}_cleaned.xlsx")
        df.to_excel(export_path, index=False, engine="openpyxl")
        media_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        dl_filename = f"{clean_name}_cleaned.xlsx"
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{format}'. Use 'csv' or 'excel'."
        )

    return FileResponse(
        path=export_path,
        filename=dl_filename,
        media_type=media_type,
    )



@router.get("/charts/{session_id}/{chart_name}", summary="Get a chart image (Base64)")
async def get_chart(session_id: str, chart_name: str):
    """
    Returns the Base64-encoded PNG for a specific chart generated during analysis.
    """
    session = _sessions.get(session_id)
    if not session or not session.get("state"):
        raise HTTPException(status_code=404, detail="Session or state not found.")

    charts = session["state"].get("visualizations", {})
    chart = charts.get(chart_name)
    if not chart:
        raise HTTPException(
            status_code=404,
            detail=f"Chart '{chart_name}' not found. Available: {list(charts.keys())}",
        )

    return {"chart_name": chart_name, "image_base64": chart}


@router.get("/sessions", summary="List all active sessions")
async def list_sessions():
    return {
        "sessions": [
            {
                "session_id": sid,
                "file_name": s.get("file_name"),
                "analyzed": s.get("state") is not None,
            }
            for sid, s in _sessions.items()
        ]
    }
