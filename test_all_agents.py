"""
End-to-End Test Script for the Autonomous Data Analyst Agent
────────────────────────────────────────────────────────────
Run this script to test every feature of the backend:
  ✅ Test 1 : Health check (server is online)
  ✅ Test 2 : File upload
  ✅ Test 3 : Full analysis pipeline (all 6 agents)
  ✅ Test 4 : PDF report download
  ✅ Test 5 : Dataset chat (ask a question)

Usage:
    python test_all_agents.py

Requirements:
  - Server must be running: python main.py (or uvicorn)
  - API Key must be set in .env
"""

import os
import sys
import json
import time
import requests

BASE_URL    = "http://localhost:8000/api/v1"
SAMPLE_CSV  = os.path.join("data", "sample_sales.csv")
REPORT_DIR  = os.path.join("data", "reports")

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

passed = 0
failed = 0


def section(title: str):
    print(f"\n{BOLD}{CYAN}{'='*55}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{BOLD}{CYAN}{'='*55}{RESET}")


def ok(msg: str):
    global passed
    passed += 1
    print(f"  {GREEN}PASS{RESET}  {msg}")


def fail(msg: str, detail: str = ""):
    global failed
    failed += 1
    print(f"  {RED}FAIL{RESET}  {msg}")
    if detail:
        print(f"         {RED}{detail}{RESET}")


def info(msg: str):
    print(f"  {YELLOW}INFO  {msg}{RESET}")


# ─────────────────────────────────────────────────────────────────────────────

def test_health():
    section("TEST 1: Server Health Check")
    try:
        r = requests.get("http://localhost:8000", timeout=5)
        if r.status_code == 200 and r.json().get("status") == "online":
            ok("Server is online and healthy.")
        else:
            fail("Unexpected response from server.", str(r.json()))
    except requests.ConnectionError:
        fail("Could not connect to server.", "Make sure the server is running: python main.py")
        sys.exit(1)


def test_upload() -> str | None:
    section("TEST 2: File Upload")
    if not os.path.exists(SAMPLE_CSV):
        fail("Sample CSV not found.", f"Expected at: {SAMPLE_CSV}")
        return None

    try:
        with open(SAMPLE_CSV, "rb") as f:
            r = requests.post(
                f"{BASE_URL}/upload",
                files={"file": ("sample_sales.csv", f, "text/csv")},
                timeout=15,
            )

        if r.status_code == 200:
            data = r.json()
            session_id = data.get("session_id")
            ok(f"File uploaded successfully.")
            info(f"Session ID -> {session_id}")
            info(f"File saved -> {data.get('file_path')}")
            return session_id
        else:
            fail(f"Upload failed with status {r.status_code}", r.text[:300])
            return None
    except Exception as exc:
        fail("Upload request failed.", str(exc))
        return None


def test_analyze(session_id: str) -> dict | None:
    section("TEST 3: Full Analysis Pipeline (All 6 Agents)")
    info("Running: Loader -> Cleaner -> Analyzer -> Visualizer -> InsightGen -> ReportGen")
    info("This may take 30-90 seconds (includes Gemini API call)...")

    try:
        start = time.time()
        r = requests.post(
            f"{BASE_URL}/analyze",
            params={
                "session_id": session_id,
                "prompt": "Perform a complete autonomous analysis on this sales dataset.",
            },
            timeout=180,
        )
        elapsed = time.time() - start

        if r.status_code == 200:
            data = r.json()
            ok(f"Analysis completed in {elapsed:.1f}s")

            # Check pipeline steps
            steps = data.get("pipeline_steps", [])
            info(f"Pipeline steps executed: {steps}")

            # Check cleaning log
            cleaning = data.get("cleaning_log", [])
            ok(f"Cleaning agent: {len(cleaning)} cleaning operations logged.")
            for step in cleaning[:4]:
                info(f"  -> {step}")

            # Check statistics
            stats = data.get("statistics", {})
            shape = stats.get("shape", {})
            ok(f"Analyzer: Dataset shape = {shape.get('rows')} rows × {shape.get('columns')} columns")
            top_corrs = stats.get("top_correlations", [])
            if top_corrs:
                ok(f"Analyzer: Top correlation = {top_corrs[0]['col_a']} <-> {top_corrs[0]['col_b']} (r={top_corrs[0]['correlation']:.3f})")

            # Check charts
            charts = data.get("chart_names", [])
            ok(f"Visualizer: {len(charts)} charts generated -> {charts}")

            # Check insights
            insights = data.get("insights", "")
            if insights and len(insights) > 50:
                ok(f"Insight Agent: {len(insights)} characters of insights generated.")
                info(f"Preview: {insights[:200].strip()}...")
            else:
                fail("Insight Agent: No insights returned.")

            # Check recommendations
            recs = data.get("recommendations", [])
            if recs:
                ok(f"Insight Agent: {len(recs)} recommendations generated.")
                for r_item in recs[:2]:
                    info(f"  -> {r_item}")
            else:
                fail("Insight Agent: No recommendations found.")

            # Check report
            report_path = data.get("report_path")
            if report_path and os.path.exists(report_path):
                size_kb = os.path.getsize(report_path) / 1024
                ok(f"Reporter: PDF generated at {report_path} ({size_kb:.1f} KB)")
            elif report_path:
                fail("Reporter: report_path returned but file not found on disk.", report_path)
            else:
                fail("Reporter: No report_path in response.")

            # Print error log if any
            errors = data.get("error_log", [])
            if errors:
                print(f"\n  {YELLOW}WARNING  Error log:{RESET}")
                for e in errors:
                    info(f"  {e}")

            return data
        else:
            fail(f"Analysis failed with status {r.status_code}", r.text[:300])
            return None

    except requests.Timeout:
        fail("Analysis timed out after 180s.")
        return None
    except Exception as exc:
        fail("Analysis request failed.", str(exc))
        return None


def test_download_pdf(session_id: str):
    section("TEST 4: PDF Report Download")
    try:
        r = requests.get(
            f"{BASE_URL}/report/{session_id}",
            timeout=15,
        )
        if r.status_code == 200:
            content_type = r.headers.get("content-type", "")
            if "pdf" in content_type:
                save_path = os.path.join(REPORT_DIR, f"downloaded_test_report.pdf")
                with open(save_path, "wb") as f:
                    f.write(r.content)
                size_kb = len(r.content) / 1024
                ok(f"PDF downloaded successfully ({size_kb:.1f} KB)")
                info(f"Saved to -> {save_path}")
            else:
                fail("Response is not a PDF.", f"Content-Type: {content_type}")
        else:
            fail(f"PDF download failed with status {r.status_code}", r.text[:200])
    except Exception as exc:
        fail("PDF download request failed.", str(exc))


def test_chat(session_id: str):
    section("TEST 5: Dataset Chat Agent")
    queries = [
        "What is the average sales amount?",
        "Which region has the highest total profit?",
        "How many orders are in the Electronics category?",
    ]

    for query in queries:
        try:
            r = requests.post(
                f"{BASE_URL}/chat",
                params={"session_id": session_id, "query": query},
                timeout=60,
            )
            if r.status_code == 200:
                answer = r.json().get("answer", "")
                if answer and len(answer) > 5:
                    ok(f"Q: '{query}'")
                    info(f"   A: {answer[:150].strip()}")
                else:
                    fail(f"Empty answer for: '{query}'")
            else:
                fail(f"Chat failed for: '{query}'", r.text[:200])
        except Exception as exc:
            fail(f"Chat request failed for: '{query}'", str(exc))


# ─────────────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{BOLD}{'='*55}")
    print("  Autonomous Data Analyst — Full System Test")
    print(f"{'='*55}{RESET}")
    print(f"  Backend URL : {BASE_URL}")
    print(f"  Sample Data : {SAMPLE_CSV}")

    test_health()

    session_id = test_upload()
    if not session_id:
        print(f"\n{RED}Stopping: Upload failed.{RESET}")
        sys.exit(1)

    analysis_data = test_analyze(session_id)

    if analysis_data:
        test_download_pdf(session_id)
        test_chat(session_id)

    # Final summary
    section("TEST SUMMARY")
    total = passed + failed
    print(f"  {GREEN}Passed: {passed}/{total}{RESET}")
    if failed:
        print(f"  {RED}Failed: {failed}/{total}{RESET}")
    else:
        print(f"  {BOLD}{GREEN}All tests passed! System is fully operational.{RESET}")
    print()


if __name__ == "__main__":
    main()
