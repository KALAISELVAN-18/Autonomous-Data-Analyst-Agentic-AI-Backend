"""
LangGraph Orchestrator
───────────────────────
Compiles the multi-agent state graph that routes through all analyst nodes.
"""

from langgraph.graph import StateGraph, END
from typing import Any

from src.core.state import AnalystState
from src.agents.planner    import planner_node
from src.agents.loader     import loader_node
from src.agents.cleaner    import cleaner_node
from src.agents.analyzer   import analyzer_node
from src.agents.visualizer import visualizer_node
from src.agents.insights   import insights_node
from src.agents.chat       import chat_node
from src.agents.reporter   import reporter_node


AGENT_NODES = {
    "loader":      loader_node,
    "cleaner":     cleaner_node,
    "analyzer":    analyzer_node,
    "visualizer":  visualizer_node,
    "insight_gen": insights_node,
    "chat":        chat_node,
    "report_gen":  reporter_node,
}


def route_next_agent(state: AnalystState) -> str:
    """
    Routing function called after every planner execution.
    Returns the name of the next agent or END.
    """
    next_agent = state.get("next_agent", "END")
    if next_agent in AGENT_NODES:
        return next_agent
    return END


def build_graph() -> Any:
    """Compile and return the LangGraph StatefulGraph."""
    workflow = StateGraph(AnalystState)

    # Register all nodes
    workflow.add_node("planner", planner_node)
    for name, fn in AGENT_NODES.items():
        workflow.add_node(name, fn)

    # Entry point is always the planner
    workflow.set_entry_point("planner")

    # Planner uses conditional routing
    workflow.add_conditional_edges(
        "planner",
        route_next_agent,
        {**{name: name for name in AGENT_NODES}, END: END},
    )

    # After every worker node → return to planner for next decision
    for name in AGENT_NODES:
        if name == "report_gen":
            workflow.add_edge("report_gen", END)   # report_gen is always the last step
        else:
            workflow.add_edge(name, "planner")

    return workflow.compile()


# Singleton graph instance
analyst_graph = build_graph()
