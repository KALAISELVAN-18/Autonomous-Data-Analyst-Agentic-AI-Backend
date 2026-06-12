"""
Planner Agent
─────────────
Responsibilities:
  • Determine the correct execution sequence based on current state.
  • Decide if analysis is complete or if re-routing is needed.
  • Detect and handle error conditions (max retries exceeded).

The planner uses a deterministic rule-based approach so the system
behaves predictably and avoids unnecessary LLM calls for routing.
"""

from typing import Dict, Any
from src.core.state import AnalystState

# Maximum consecutive errors before we stop.
MAX_ERRORS = 3

# The canonical full-analysis pipeline executed in this exact order.
FULL_PIPELINE = ["loader", "cleaner", "analyzer", "visualizer", "insight_gen", "report_gen"]


def planner_node(state: AnalystState) -> Dict[str, Any]:
    """
    Planner Node — decides the next agent to run.

    Logic:
      1. If error threshold exceeded → terminate with an error message.
      2. If there are remaining steps in current_plan → pop and run the next one.
      3. If no plan exists yet → build the FULL_PIPELINE plan.
      4. If plan is exhausted → END.
    """
    print("\n========== PLANNER AGENT ==========")

    error_count = state.get("error_count", 0)
    error_log = state.get("error_log", [])

    # ── Guard: abort if too many errors ──────────────────────────────────────
    if error_count >= MAX_ERRORS:
        print(f"[PLANNER] Error limit reached ({MAX_ERRORS}). Aborting pipeline.")
        return {
            "next_agent": "END",
            "current_plan": [],
            "error_log": error_log + [f"Pipeline aborted after {error_count} errors."],
        }

    plan = list(state.get("current_plan", []))  # make a mutable copy

    # ── First call: initialise the pipeline ─────────────────────────────────
    if not plan:
        plan = list(FULL_PIPELINE)
        print(f"[PLANNER] Initialising pipeline: {plan}")

    # ── Pop the next agent from the front of the plan ───────────────────────
    if plan:
        next_agent = plan.pop(0)
        print(f"[PLANNER] Routing to -> '{next_agent}'  |  Remaining: {plan}")
        return {"current_plan": plan, "next_agent": next_agent}

    # ── Plan exhausted ───────────────────────────────────────────────────────
    print("[PLANNER] Pipeline complete -> END")
    return {"current_plan": [], "next_agent": "END"}
