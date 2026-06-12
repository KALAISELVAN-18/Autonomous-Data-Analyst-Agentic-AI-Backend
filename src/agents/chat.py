"""
Dataset Chat Agent
──────────────────
Responsibilities:
  • Accept a natural language question about the dataset.
  • Use Gemini to translate the question into a valid Pandas expression.
  • Safely execute that expression against the loaded DataFrame.
  • Return a clean, human-readable answer (text + optional mini-table).

Uses a ReAct-style loop:
  Thought → Generate code → Execute → Observe → Respond.
"""

import re
import traceback
import pandas as pd
import numpy as np
from typing import Dict, Any

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from src.core.state import AnalystState
from src.utils.llm_factory import get_llm


CHAT_SYSTEM_PROMPT = """
You are a data analyst assistant. The user will ask questions about a pandas DataFrame stored in the variable `df`.

Your job:
1. Write a valid Python code block (using pandas and numpy) that answers the question.
2. Store the final answer in a local variable named `result`. The `result` variable can contain a number, string, list, dictionary, Series, or DataFrame.
3. Wrap your code in exactly one ```python ... ``` block.
4. After the code block, explain in one sentence what the code does.

Rules:
- Do NOT import any system modules (like os, sys, subprocess). You can use pandas (`pd`) and numpy (`np`).
- Do NOT read/write any files.
- Do NOT call `print()`.
- Store the final answer in `result`.
- Keep the code concise and correct.
- If the question cannot be answered with the dataset, state that clearly and do not write a code block.
- Note that the column names in `df` have been cleaned and standardized to snake_case. You MUST map any column names referenced by the user (which might be the original names, like "Sales Amount" or "Order ID") to their standardized snake_case names (like "sales_amount" or "order_id") present in `df` using the provided column mapping.

Example:
Question: What is the average sales amount for Electronics?
```python
result = df[df['product_category'] == 'Electronics']['sales_amount'].mean().round(2)
```
This filters the data for 'Electronics' category and calculates the average sales amount.
""".strip()


def _extract_code(response_text: str) -> str | None:
    """Extract Python code from a markdown code block."""
    pattern = r"```python\s*(.*?)\s*```"
    match = re.search(pattern, response_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def _safe_exec(code: str, df: pd.DataFrame) -> tuple[Any, str | None]:
    """
    Safely execute a Python code snippet on the DataFrame.
    Returns (result, error_message).
    """
    # Basic safety guards — block dangerous operations
    BLOCKED = ["import ", "open(", "exec(", "eval(", "os.", "sys.", "__", "subprocess", "builtins"]
    for bad in BLOCKED:
        if bad in code:
            return None, f"Blocked unsafe operation: '{bad}'"

    local_ns = {"df": df, "pd": pd, "np": np}
    try:
        # We run the block of code, allowing variable assignments
        exec(code, {"__builtins__": {}}, local_ns)
        if "result" not in local_ns:
            return None, "The code did not assign a value to the 'result' variable. Please assign the final answer to the variable `result`."
        return local_ns["result"], None
    except Exception:
        return None, traceback.format_exc(limit=2)


def _format_result(result: Any) -> str:
    """Format the eval result into a human-readable string."""
    if isinstance(result, pd.DataFrame):
        return result.head(10).to_markdown(index=True)
    if isinstance(result, pd.Series):
        return result.head(10).to_string()
    if isinstance(result, (list, dict)):
        return str(result)[:2000]
    return str(result)


def chat_node(state: AnalystState) -> Dict[str, Any]:
    """
    Dataset Chat Node with self-correction/reflection loop.

    Processes the last HumanMessage in the conversation as a query,
    generates and executes a Pandas script using Gemini,
    with self-correction on execution failures, and appends the result to state.
    """
    print("\n========== CHAT AGENT ==========")

    df = state.get("dataset")
    messages = state.get("messages", [])

    if df is None:
        msg = "[CHAT] No dataset loaded. Please upload a dataset first."
        print(msg)
        return {
            "messages": [AIMessage(content=msg)],
        }

    # Get the most recent human query
    human_query = ""
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            human_query = m.content
            break

    if not human_query:
        return {"messages": [AIMessage(content="No question received.")]}

    # Build context for the LLM
    metadata = state.get("metadata", {})
    column_mapping = metadata.get("column_mapping", {})
    
    mapping_str = ""
    if column_mapping:
        mapping_str = "Column mapping (original name -> snake_case name):\n" + "\n".join([f" - '{orig}' -> '{clean}'" for orig, clean in column_mapping.items()])

    df_info = (
        f"DataFrame shape: {df.shape[0]} rows x {df.shape[1]} columns\n"
        f"Columns: {list(df.columns)}\n"
        f"dtypes:\n{df.dtypes.to_string()}\n"
        f"Sample (first 3 rows):\n{df.head(3).to_string()}"
    )
    if mapping_str:
        df_info += f"\n\n{mapping_str}"

    user_content = (
        f"Dataset info:\n```\n{df_info}\n```\n\n"
        f"Question: {human_query}"
    )

    llm = get_llm(temperature=0.1)
    
    attempts = 3
    failed_attempts_log = []
    
    for attempt in range(attempts):
        print(f"[CHAT] Generation attempt {attempt + 1}/{attempts}...")
        
        if attempt == 0:
            messages_to_send = [
                SystemMessage(content=CHAT_SYSTEM_PROMPT),
                HumanMessage(content=user_content),
            ]
        else:
            # Add self-correction context
            last_failed = failed_attempts_log[-1]
            correction_prompt = (
                f"Your previous Python code failed with the following error:\n\n"
                f"```\n{last_failed['error']}\n```\n\n"
                f"Here is the code you wrote:\n```python\n{last_failed['code']}\n```\n\n"
                f"Please review the column names: {list(df.columns)} and data types carefully.\n"
                f"Write a corrected block of Python code. Remember to store your final answer in the `result` variable."
            )
            messages_to_send = [
                SystemMessage(content=CHAT_SYSTEM_PROMPT),
                HumanMessage(content=user_content),
                AIMessage(content=last_failed['raw_response']),
                HumanMessage(content=correction_prompt),
            ]
            
        try:
            response = llm.invoke(messages_to_send)
            raw = response.content
        except Exception as exc:
            print(f"[CHAT] LLM call failed: {exc}")
            if attempt == attempts - 1:
                answer = f"Sorry, I encountered an error while communicating with the model: {exc}"
                return {"messages": [AIMessage(content=answer)]}
            continue

        # Extract and execute code
        code = _extract_code(raw)
        if not code:
            # LLM answered in prose — return it directly
            print("[CHAT] No code block generated. Returning prose response.")
            return {"messages": [AIMessage(content=raw)]}

        result, error = _safe_exec(code, df)

        if error:
            print(f"[CHAT] Execution failed with error:\n{error}")
            failed_attempts_log.append({
                "code": code,
                "error": error,
                "raw_response": raw
            })
        else:
            # Success! Format the result and return it
            formatted = _format_result(result)
            explanation = raw.split("```")[-1].strip()
            answer = (
                f"**Answer:**\n\n{formatted}\n\n"
                f"_{explanation}_"
            )
            print(f"[CHAT] Query answered successfully on attempt {attempt + 1}. Code: `{code.strip()}`")
            return {"messages": [AIMessage(content=answer)]}

    # If we run out of attempts
    last_failed = failed_attempts_log[-1]
    answer = (
        f"I tried to answer your question but the code execution failed after {attempts} attempts:\n\n"
        f"```\n{last_failed['error']}\n```\n\n"
        f"Last generated code:\n```python\n{last_failed['code']}\n```"
    )
    return {"messages": [AIMessage(content=answer)]}
