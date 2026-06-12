import asyncio
import os
import shutil
from src.core.orchestrator import analyst_graph
from langchain_core.messages import HumanMessage
from src.core.config import settings

def debug():
    # Setup dummy session
    os.makedirs(settings.upload_dir, exist_ok=True)
    file_path = os.path.join(settings.upload_dir, "unclean_sales_data.csv")
    with open(file_path, "w") as f:
        f.write("A,B\n1,2\n3,4")

    prompt = "Perform a complete autonomous analysis on this dataset."
    initial_state = {
        "messages": [HumanMessage(content=prompt)],
        "dataset": None,
        "raw_dataset": None,
        "file_path": file_path,
        "file_name": "unclean_sales_data.csv",
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

    print("Starting stream...")
    try:
        for chunk in analyst_graph.stream(initial_state):
            node_name = list(chunk.keys())[0]
            print(f"Executed node: {node_name}")
            print(f"Result: {chunk[node_name].keys()}")
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug()
