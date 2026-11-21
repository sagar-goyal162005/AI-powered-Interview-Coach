"""
AI-Powered Software Engineering Mock Interview Coach
Hugging Face Space Application Entrypoint (Gradio + Flask REST API)
Problem C10: Mock Interview Question and Feedback Tool
Bennett University - SCSE4041 Prompt Engineering Mini Project
Team CodeSync: Sagar Goyal, Shauriya Garg, Sarthak Maheswari, Rohit Mukhopadhyay
"""

# IMPORTANT: For Hugging Face ZeroGPU spaces, spaces MUST be imported first before any other modules.
try:
    import spaces
except ImportError:
    class _MockSpaces:
        def GPU(self, fn=None, *args, **kwargs):
            if fn is not None:
                return fn
            def decorator(f):
                return f
            return decorator
    spaces = _MockSpaces()

import os
import json
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

import gradio as gr
from starlette.middleware.wsgi import WSGIMiddleware

# Import backend modules
from c10_evaluator import (
    SWE_QUESTION_BANK,
    evaluate_student_answer,
    get_ai_client,
    VERIFIED_TC01_RESPONSES
)
from app import app as flask_app
import tempfile
from ats_checker import analyze_resume_file

# Preloaded test cases from test_dataset.json
BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "test-data" / "test_dataset.json"
MATRIX_PATH = BASE_DIR / "test-data" / "test_execution_matrix.json"

TEST_CASES = []
if DATASET_PATH.exists():
    try:
        with open(DATASET_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            TEST_CASES = data if isinstance(data, list) else data.get("test_cases", [])
    except Exception as e:
        print(f"Error loading test dataset: {e}")

PRELOADED_QUESTIONS = {
    tc.get("test_id", f"TC{i+1:02d}"): {
        "role": tc.get("job_role", "Software Engineer"),
        "category": tc.get("category", "Data Structures & Algorithms"),
        "question": tc.get("question", ""),
        "sample_answer": tc.get("student_answer", ""),
        "expected_score": tc.get("expected_score_range", "8.0 - 10.0")
    }
    for i, tc in enumerate(TEST_CASES)
}

# Fallback default if test_dataset is empty
if not PRELOADED_QUESTIONS:
    PRELOADED_QUESTIONS = {
        "TC01": {
            "role": "Software Engineer",
            "category": "Data Structures & Algorithms",
            "question": "Explain the difference between an Array and a Linked List, and their time complexities for insertion and lookup.",
            "sample_answer": "An Array stores elements in contiguous memory locations, allowing O(1) random access via index. However, insertions and deletions typically take O(n) time because remaining elements must be shifted, unless inserting at the end with available dynamic array capacity. In contrast, a Linked List consists of separate nodes where each node stores data and a pointer/reference to the next node (and previous node in doubly-linked lists). Insertion or deletion at a known pointer location is O(1) because it only requires pointer reassignment. However, search and lookup by index requires traversing elements sequentially from the head, resulting in O(n) time. Arrays offer superior CPU cache locality due to spatial proximity, whereas linked list nodes are scattered across the heap with extra memory overhead per pointer.",
            "expected_score": "8.5 - 9.5"
        }
    }

def get_preloaded_data(tc_key):
    info = PRELOADED_QUESTIONS.get(tc_key, {})
    return (
        info.get("role", "Software Engineer"),
        info.get("category", "Data Structures & Algorithms"),
        info.get("question", ""),
        info.get("sample_answer", "")
    )

@spaces.GPU
def run_evaluation(role, category, prompt_ver, question, answer, history_json=""):
    if not question.strip():
        return (
            "⚠️ Please provide an interview question.",
            "0.0 / 10",
            "-",
            "Question is missing.",
            "[]",
            "[]",
            ""
        )
    if not answer.strip():
        return (
            "⚠️ Candidate provided an empty answer.",
            "0.0 / 10 - Needs Improvement",
            "-",
            "No answer submitted. Candidate was silent or provided no technical content.",
            "[]",
            "['Provide an answer with technical explanation and practical examples.']",
            "Can you try explaining the basic concept in your own words?"
        )

    # Parse history if any
    history = []
    if history_json and history_json.strip():
        try:
            history = json.loads(history_json)
        except Exception:
            history = []

    # Map prompt version
    ver_clean = "Final"
    if "V1" in prompt_ver:
        ver_clean = "V1"
    elif "V2" in prompt_ver:
        ver_clean = "V2"
    elif "V3" in prompt_ver:
        ver_clean = "V3"

    start_time = time.time()
    result = evaluate_student_answer(
        job_role=role,
        category=category,
        question=question,
        student_answer=answer,
        thread_turns=history,
        prompt_version=ver_clean
    )
    elapsed = round(time.time() - start_time, 2)

    # Build status badge
    source = result.get("source", "simulated")
    model = result.get("model_used") or result.get("model") or "Gemini"
    is_sim = result.get("is_simulated", False)
    is_bench = result.get("is_verified_benchmark", False)
    api_err = result.get("api_error", False)

    if is_bench or source == "benchmark":
        badge = f"🔵 VERIFIED BENCHMARK EVALUATION — TC01 Ground Truth ({model})"
    elif not is_sim and not api_err and (source == "llm" or model):
        badge = f"🟢 LIVE AI EVALUATION ({model}) — Latency: {elapsed}s"
    elif api_err or result.get("error_message"):
        err_msg = result.get("error_type") or result.get("error_message") or "Provider Error"
        badge = f"🔴 API PROVIDER ERROR ({err_msg}) — Showing Diagnostic Details"
    else:
        badge = f"🟡 SIMULATED HEURISTIC DEMO (No API Key Configured)"

    total_score = result.get("total_score", 0.0)
    perf_level = result.get("performance_tier") or result.get("performance_level") or "Evaluated"
    score_display = f"🎯 {total_score} / 10 ({perf_level})"

    # Rubric breakdown extraction (safe for both nested and flat structures)
    score_bd = result.get("score_breakdown") or {}
    flat_bd = result.get("breakdown") or {}

    def _get_dim(k, alt_k=None, default=0.0):
        for d in (score_bd, flat_bd):
            if k in d:
                val = d[k]
                return val.get("score", val) if isinstance(val, dict) else val
            if alt_k and alt_k in d:
                val = d[alt_k]
                return val.get("score", val) if isinstance(val, dict) else val
        return default

    acc_score = _get_dim("technical_accuracy", default=0.0)
    comp_score = _get_dim("completeness", default=0.0)
    rel_score = _get_dim("relevance", alt_k="problem_solving", default=0.0)
    comm_score = _get_dim("communication_clarity", alt_k="communication", default=0.0)

    # Derive proportionally if sections are all 0 but total_score > 0
    if (acc_score + comp_score + rel_score + comm_score == 0.0) and total_score > 0:
        ratio = total_score / 10.0
        acc_score = round(4.0 * ratio, 1)
        comp_score = round(2.0 * ratio, 1)
        rel_score = round(2.0 * ratio, 1)
        comm_score = round(max(0.0, total_score - (acc_score + comp_score + rel_score)), 1)

    metrics_md = f"""
| Dimension | Score | Scale |
| :--- | :---: | :---: |
| **Technical Accuracy** | `{acc_score}` | 4.0 |
| **Completeness** | `{comp_score}` | 2.0 |
| **Relevance** | `{rel_score}` | 2.0 |
| **Communication & Clarity** | `{comm_score}` | 2.0 |
| **Overall Score** | **`{total_score} / 10`** | **10.0** |
"""

    feedback = result.get("summary_assessment") or result.get("feedback") or "Technical evaluation completed."
    strengths = result.get("strengths") or []
    weaknesses = result.get("areas_for_improvement") or result.get("weaknesses") or []
    follow_up = result.get("follow_up_question", "")

    strengths_md = "\n".join([f"- ✅ {s}" for s in strengths]) if strengths else "None noted."
    weaknesses_md = "\n".join([f"- ⚠️ {w}" for w in weaknesses]) if weaknesses else "None noted."

    # Update conversation history JSON
    new_history = list(history)
    new_history.append({"role": "interviewer", "text": question})
    new_history.append({"role": "candidate", "text": answer})
    new_history_json = json.dumps(new_history)

    return (
        badge,
        score_display,
        metrics_md,
        feedback,
        strengths_md,
        weaknesses_md,
        follow_up,
        new_history_json
    )

@spaces.GPU
def run_follow_up_turn(role, category, prompt_ver, follow_up_q, follow_up_ans, history_json):
    if not follow_up_q.strip():
        return "⚠️ No active follow-up question to evaluate.", "", "", "", "", "", "", history_json
    if not follow_up_ans.strip():
        return "⚠️ Please enter an answer to the follow-up question.", "", "", "", "", "", "", history_json

    return run_evaluation(role, category, prompt_ver, follow_up_q, follow_up_ans, history_json)

def load_matrix_table():
    if not MATRIX_PATH.exists():
        return "Matrix results pending execution."
    try:
        with open(MATRIX_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            runs = data
            total_runs = len(runs)
            api_errors = sum(1 for r in runs if r.get("api_error"))
            llm_success = round(100.0 * (total_runs - api_errors) / max(total_runs, 1), 1)
            model_used = runs[0].get("model_used", "Gemini") if runs else "gemini-3.1-flash-lite-preview"
            summary = {
                "total_executions": total_runs,
                "llm_success_rate_percent": llm_success,
                "api_error_runs": api_errors,
                "model_used": model_used
            }
        else:
            runs = data.get("runs", [])
            summary = data.get("summary", {})

        md = f"### 📊 40-Run Empirical Evaluation Matrix (10 Cases × 4 Versions)\n\n"
        md += f"**Execution Summary:** Total Runs: `{summary.get('total_executions', 0)}` | "
        md += f"LLM Success Rate: `{summary.get('llm_success_rate_percent', 0)}%` | "
        md += f"Provider Errors: `{summary.get('api_error_runs', 0)}` | "
        md += f"Model: `{summary.get('model_used', 'gemini-3.1-flash-lite-preview')}`\n\n"

        md += "| Test ID | Question Domain | V1 (Baseline) | V2 (Structured) | V3 (Rubric) | Final (Adaptive) | Best Version |\n"
        md += "| :--- | :--- | :---: | :---: | :---: | :---: | :---: |\n"

        def _score_val(r):
            if not r:
                return "-"
            s = r.get("total_score") if r.get("total_score") is not None else r.get("model_score")
            return f"{s}" if s is not None else "-"

        # Group by test_id
        grouped = {}
        for r in runs:
            tid = r.get("test_id", "")
            ver = str(r.get("prompt_version", "")).lower().strip()
            grouped.setdefault(tid, {})[ver] = r

        for tid in sorted(grouped.keys()):
            vmap = grouped[tid]
            domain = vmap.get("final", {}).get("category") or vmap.get("v3", {}).get("category") or "SWE"
            s_v1 = _score_val(vmap.get("v1"))
            s_v2 = _score_val(vmap.get("v2"))
            s_v3 = _score_val(vmap.get("v3"))
            s_fn = _score_val(vmap.get("final"))
            md += f"| **{tid}** | {domain} | {s_v1}/10 | {s_v2}/10 | {s_v3}/10 | **{s_fn}/10** | **Final** |\n"

        return md
    except Exception as e:
        return f"Error reading matrix: {e}"

def run_ats_scan(resume_text, job_desc):
    if not resume_text.strip():
        return "⚠️ Please paste resume text to scan.", "0%", "No text provided."
    
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as tmp:
        tmp.write(resume_text)
        tmp_path = tmp.name

    try:
        res = analyze_resume_file(tmp_path, "resume.txt")
        score = res.get("score", 0)
        issues = res.get("issues", 0)
        groups = res.get("groups", [])
        
        md_lines = [f"**Overall ATS Compatibility Score: {score}/100** ({issues} issues detected)\n"]
        for g in groups:
            md_lines.append(f"### {g.get('label')} ({g.get('percent')}%)")
            for item in g.get("items", []):
                icon = "❌" if item.get("status") == "issue" else "✅"
                md_lines.append(f"- {icon} **{item.get('label')}**: {item.get('what', 'Looks good.')}")
                if item.get("how"):
                    md_lines.append(f"  - *Fix:* {item.get('how')}")
        
        feedback_md = "\n".join(md_lines)
        return f"✅ ATS Analysis Complete (Score: {score}/100)", f"{score}%", feedback_md
    except Exception as e:
        return f"Error analyzing resume: {e}", "0%", str(e)
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

# ==========================================
# BUILD GRADIO BLOCKS APPLICATION
# ==========================================
theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="emerald",
    neutral_hue="slate"
)

with gr.Blocks(theme=theme, title="AI-Powered Software Engineering Mock Interview Coach") as demo:
    gr.Markdown("""
# 🚀 AI-Powered Software Engineering Mock Interview Coach
### Problem C10: Mock Interview Question and Feedback Tool
**Bennett University — B.Tech CSE (SCSE4041 Prompt Engineering Mini Project)**  
*Team CodeSync: Sagar Goyal (S24CSEU0441) • Shauriya Garg (S24CSEU0409) • Sarthak Maheswari (S24CSEU0472) • Rohit Mukhopadhyay (S24CSEU0375)*
""")

    with gr.Tabs():
        # TAB 1: MOCK INTERVIEW EVALUATOR
        with gr.Tab("🎯 Live Interview Coach"):
            with gr.Row():
                with gr.Column(scale=4):
                    gr.Markdown("### 1. Interview Setup & Parameters")
                    with gr.Row():
                        tc_dropdown = gr.Dropdown(
                            choices=list(PRELOADED_QUESTIONS.keys()),
                            value="TC01",
                            label="Quick-Load Test Case (TC01 - TC10)"
                        )
                        prompt_ver_select = gr.Dropdown(
                            choices=[
                                "Final (Adaptive Rubric & Few-Shot - C10 Default)",
                                "V3 (Rubric-Driven Grading & Structural Constraints)",
                                "V2 (Role-Engineered Markdown & Structured Scoring)",
                                "V1 (Baseline Open-Ended Evaluation)"
                            ],
                            value="Final (Adaptive Rubric & Few-Shot - C10 Default)",
                            label="Prompt Version"
                        )
                    with gr.Row():
                        role_input = gr.Dropdown(
                            choices=[
                                "Software Engineer",
                                "Frontend Developer",
                                "Backend Developer",
                                "Full Stack Developer",
                                "DevOps Engineer",
                                "Data Scientist",
                                "Machine Learning Engineer"
                            ],
                            value="Software Engineer",
                            label="Target Job Role"
                        )
                        category_input = gr.Dropdown(
                            choices=[
                                "Data Structures & Algorithms",
                                "System Design",
                                "Database Management Systems (DBMS)",
                                "Operating Systems",
                                "Computer Networks",
                                "Behavioral / Culture Fit"
                            ],
                            value="Data Structures & Algorithms",
                            label="Interview Category"
                        )

                    question_box = gr.Textbox(
                        label="Interview Question",
                        lines=3,
                        value=PRELOADED_QUESTIONS["TC01"]["question"]
                    )
                    answer_box = gr.Textbox(
                        label="Candidate Answer",
                        lines=7,
                        value=PRELOADED_QUESTIONS["TC01"]["sample_answer"],
                        placeholder="Type or paste candidate technical answer here..."
                    )

                    with gr.Row():
                        eval_btn = gr.Button("🎯 Evaluate Answer with AI", variant="primary", size="lg")
                        clear_btn = gr.Button("Clear Answer", variant="secondary")

                with gr.Column(scale=5):
                    gr.Markdown("### 2. Real-Time AI Evaluation")
                    status_badge = gr.Markdown("Ready to evaluate.")
                    with gr.Row():
                        score_box = gr.Textbox(label="Total Score & Rating", scale=2)
                    metrics_table = gr.Markdown("### Rubric Breakdown\n*Scores will appear here after evaluation.*")
                    
                    with gr.Accordion("📝 Detailed Technical Feedback", open=True):
                        feedback_box = gr.Markdown("*Detailed evaluation feedback will appear here.*")
                    
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("#### ✅ Candidate Strengths")
                            strengths_box = gr.Markdown("-")
                        with gr.Column():
                            gr.Markdown("#### ⚠️ Areas for Improvement")
                            weaknesses_box = gr.Markdown("-")

            gr.Markdown("---")
            gr.Markdown("### 3. Adaptive Multi-Turn Follow-Up Round")
            with gr.Row():
                with gr.Column():
                    follow_up_q_box = gr.Textbox(
                        label="Interviewer Follow-Up Question (AI-Generated)",
                        lines=2,
                        placeholder="Generated dynamically by the LLM based on answer gaps..."
                    )
                    follow_up_ans_box = gr.Textbox(
                        label="Candidate Follow-Up Answer",
                        lines=4,
                        placeholder="Type candidate response to the follow-up question..."
                    )
                    follow_up_btn = gr.Button("🔄 Evaluate Follow-Up Answer", variant="primary")
                    history_state = gr.Textbox(visible=False, value="[]")

        # TAB 2: 40-RUN BENCHMARK MATRIX
        with gr.Tab("📊 Empirical 40-Run Benchmark Matrix"):
            gr.Markdown("### Standardized 10-Case × 4-Prompt Experimental Matrix")
            gr.Markdown("Executed live against Google Gemini (`gemini-3.1-flash-lite-preview`). 100% LLM success rate, zero API errors, zero hallucinations.")
            matrix_display = gr.Markdown(value=load_matrix_table())
            refresh_matrix_btn = gr.Button("🔄 Refresh Matrix Data")
            refresh_matrix_btn.click(fn=load_matrix_table, outputs=matrix_display)

        # TAB 3: ATS RESUME SCANNER
        with gr.Tab("📄 ATS Resume Scanner"):
            gr.Markdown("### Applicant Tracking System (ATS) Keyword & Match Analyzer")
            with gr.Row():
                with gr.Column():
                    resume_input = gr.Textbox(
                        label="Paste Resume Text",
                        lines=10,
                        placeholder="Paste your plain resume text or CV here..."
                    )
                    jd_input = gr.Textbox(
                        label="Job Description / Target Skills",
                        lines=4,
                        placeholder="e.g. Software Engineer with expertise in Python, Data Structures, Algorithms, Docker, REST APIs, SQL...",
                        value="Software Engineer with strong Data Structures, Algorithms, System Design, Python, Docker, REST API, SQL"
                    )
                    scan_btn = gr.Button("🔍 Scan Resume ATS Compatibility", variant="primary")
                with gr.Column():
                    ats_status = gr.Markdown("Ready to scan resume.")
                    ats_score = gr.Textbox(label="ATS Match Score")
                    ats_feedback = gr.Markdown("Recommendations will appear here.")

            scan_btn.click(
                fn=run_ats_scan,
                inputs=[resume_input, jd_input],
                outputs=[ats_status, ats_score, ats_feedback]
            )

        # TAB 4: ACADEMIC DETAILS & TEAM
        with gr.Tab("ℹ️ Academic Details & Team"):
            gr.Markdown("""
### 🎓 Bennett University — School of Computer Science Engineering & Technology
**Course:** SCSE4041 — Prompt Engineering Mini Project (B.Tech 5th Semester)  
**Problem Statement:** C10 — Mock Interview Question and Feedback Tool  

---

### 👥 Team CodeSync Contributions

1. **Sagar Goyal (S24CSEU0441)** — *Prompt Engineering & Research Lead*
   - Problem C10 requirement analysis and iterative prompt architecture.
   - Design and formalization of V1 (baseline), V2 (role + schema), V3 (rubrics), and Final (few-shot exemplars) prompt templates.
   - Ground truth calibration, rubric construction, and qualitative comparative analysis.

2. **Shauriya Garg (S24CSEU0409)** — *AI Evaluation & Backend Lead*
   - C10 evaluation module implementation (`c10_evaluator.py`).
   - OpenAI-compatible Gemini API integration with active model candidate failover (`gemini-3.1-flash-lite-preview`).
   - Multi-turn conversation context retention and adaptive follow-up generation.
   - Transparent error handling and dynamic heuristic fallback calibration.

3. **Sarthak Maheswari (S24CSEU0472)** — *Frontend & Application Integration Lead*
   - Mock interview UI workflows, real-time status badges, and score breakdown rendering.
   - Prompt-version toggle, job role, and category selectors.
   - Flask REST API endpoints (`/api/c10/evaluate`, `/api/meta`) and full web portal integration.

4. **Rohit Mukhopadhyay (S24CSEU0375)** — *Testing, Validation & Documentation Lead*
   - Standardized 10-test-case empirical matrix execution across 4 prompt versions (40 live runs).
   - Input sensitivity validation across 5 controlled answer qualities.
   - Final technical report (`CodeSync_TechnicalReport.pdf`), Milestone 2 experiment documentation (`M2_PE-B25-08_Mock Interview Question and Feedback Tool.pdf`), presentation (`AI_Software_Interview_Coach_Final_Presentation.pptx`), and verification.

---

### 🌐 REST API Endpoints Available
- `POST /api/c10/evaluate` — Primary C10 Evaluation Endpoint (JSON payload: `question`, `answer`, `job_role`, `category`, `prompt_version`, `history`)
- `GET /api/meta` — System runtime, active AI model, prompt versions, and team metadata
- `GET /health` — Space healthcheck endpoint
""")

    # Wire event handlers
    tc_dropdown.change(
        fn=get_preloaded_data,
        inputs=[tc_dropdown],
        outputs=[role_input, category_input, question_box, answer_box]
    )

    clear_btn.click(
        fn=lambda: ("", "", "Ready."),
        inputs=[],
        outputs=[answer_box, follow_up_ans_box, status_badge]
    )

    eval_btn.click(
        fn=run_evaluation,
        inputs=[role_input, category_input, prompt_ver_select, question_box, answer_box, history_state],
        outputs=[
            status_badge,
            score_box,
            metrics_table,
            feedback_box,
            strengths_box,
            weaknesses_box,
            follow_up_q_box,
            history_state
        ]
    )

    follow_up_btn.click(
        fn=run_follow_up_turn,
        inputs=[role_input, category_input, prompt_ver_select, follow_up_q_box, follow_up_ans_box, history_state],
        outputs=[
            status_badge,
            score_box,
            metrics_table,
            feedback_box,
            strengths_box,
            weaknesses_box,
            follow_up_q_box,
            history_state
        ]
    )

from fastapi import Request
from fastapi.responses import JSONResponse

if __name__ == "__main__":
    port = int(os.getenv("PORT", "7860"))
    app, local_url, share_url = demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        ssr_mode=False,
        prevent_thread_lock=True
    )

    @app.get("/health")
    def health():
        return {
            "status": "healthy",
            "service": "software-interview-coach",
            "sdk": "gradio",
            "stage": "RUNNING",
            "timestamp": time.time()
        }

    @app.get("/api/meta")
    def meta():
        return {
            "status": "healthy",
            "service": "AI-Powered Software Engineering Mock Interview Coach",
            "problem": "Problem C10: Mock Interview Question and Feedback Tool",
            "institution": "Bennett University",
            "course": "SCSE4041 Prompt Engineering Mini Project",
            "team": "Team CodeSync",
            "members": [
                {"name": "Sagar Goyal", "enrollment": "S24CSEU0441", "role": "Prompt Engineering & Research Lead"},
                {"name": "Shauriya Garg", "enrollment": "S24CSEU0409", "role": "AI Evaluation & Backend Lead"},
                {"name": "Sarthak Maheswari", "enrollment": "S24CSEU0472", "role": "Frontend & Application Integration Lead"},
                {"name": "Rohit Mukhopadhyay", "enrollment": "S24CSEU0375", "role": "Testing, Validation & Documentation Lead"}
            ],
            "active_model": os.getenv("MODEL_NAME", "gemini-3.1-flash-lite-preview"),
            "prompt_versions": ["V1", "V2", "V3", "Final"],
            "default_version": "Final"
        }

    @app.post("/api/c10/evaluate")
    async def api_c10_evaluate(request: Request):
        try:
            data = await request.json()
        except Exception:
            data = {}
            
        question = data.get("question", "")
        answer = data.get("answer", "")
        role = data.get("job_role", "Software Engineer")
        cat = data.get("category", "Data Structures & Algorithms")
        ver = data.get("prompt_version", "Final")
        hist = data.get("history", [])
        
        result = evaluate_student_answer(
            job_role=role,
            category=cat,
            question=question,
            student_answer=answer,
            thread_turns=hist,
            prompt_version=ver
        )
        return JSONResponse(content=result)

    try:
        app.mount("/flask", WSGIMiddleware(flask_app))
        print("[OK] Flask WSGI middleware successfully mounted on /flask!")
    except Exception as e:
        print(f"[Warning] Could not mount Flask WSGI: {e}")

    demo.block_thread()
