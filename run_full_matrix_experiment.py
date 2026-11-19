"""
Run Full Test Matrix Experiment: 10 Test Cases x 4 Prompt Versions = 40 Genuine LLM Evaluation Runs.
AI-Powered Software Engineering Mock Interview Coach (Problem C10)
"""

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List

from c10_evaluator import evaluate_student_answer

BASE_DIR = Path(__file__).resolve().parent
TEST_DATA_PATH = BASE_DIR / "test-data" / "test_dataset.json"
RESULTS_JSON_PATH = BASE_DIR / "test-data" / "test_execution_matrix.json"
RESULTS_MD_PATH = BASE_DIR / "experiment-results" / "full_test_matrix_results.md"


def main():
    print("=" * 80)
    print("STARTING C10 PROMPT ENGINEERING 40-RUN TEST MATRIX EXPERIMENT")
    print("=" * 80)

    with open(TEST_DATA_PATH, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    versions = ["v1", "v2", "v3", "final"]
    total_runs = len(test_cases) * len(versions)
    results: List[Dict[str, Any]] = []

    print(f"Loaded {len(test_cases)} test cases.")
    print(f"Versions to evaluate: {versions}")
    print(f"Total runs: {total_runs}\n")

    run_index = 0
    for tc in test_cases:
        tc_id = tc["test_id"]
        question = tc["question"]
        student_ans = tc["student_answer"]
        role = tc["job_role"]
        cat = tc["category"]
        tc_type = tc.get("type", "Standard")

        print(f"\n>>> Running Test Case: {tc_id} ({tc_type}) - Question: {question[:60]}...")

        for ver in versions:
            run_index += 1
            print(f"  [{run_index}/{total_runs}] Evaluating {tc_id} with Prompt {ver.upper()}...", end="", flush=True)

            try:
                eval_result = evaluate_student_answer(
                    job_role=role,
                    category=cat,
                    question=question,
                    student_answer=student_ans,
                    prompt_version=ver
                )
                score = eval_result.get("total_score")
                model_used = eval_result.get("model_used")
                is_sim = eval_result.get("is_simulated", False)
                api_err = eval_result.get("api_error", False)
                tier = eval_result.get("performance_tier", "")
                bd = eval_result.get("score_breakdown", {})

                tech = bd.get("technical_accuracy", {}).get("score", 0.0)
                comp = bd.get("completeness", {}).get("score", 0.0)
                rel = bd.get("relevance", {}).get("score", 0.0)
                comm = bd.get("communication_clarity", {}).get("score", 0.0)

                print(f" Done! Score: {score}/10 | Model: {model_used} | Simulated: {is_sim} | Err: {api_err}")

                record = {
                    "run_index": run_index,
                    "test_id": tc_id,
                    "prompt_version": ver,
                    "test_type": tc_type,
                    "job_role": role,
                    "category": cat,
                    "question": question,
                    "student_answer": student_ans,
                    "model_score": score,
                    "performance_tier": tier,
                    "score_breakdown": {
                        "technical_accuracy": tech,
                        "completeness": comp,
                        "relevance": rel,
                        "communication_clarity": comm
                    },
                    "strengths": eval_result.get("strengths", []),
                    "areas_for_improvement": eval_result.get("areas_for_improvement", []),
                    "improved_answer": eval_result.get("improved_answer", ""),
                    "follow_up_question": eval_result.get("follow_up_question", ""),
                    "is_simulated": is_sim,
                    "api_error": api_err,
                    "model_used": model_used,
                    "provider_endpoint": eval_result.get("provider_endpoint", ""),
                    "raw_markdown": eval_result.get("raw_markdown", "")
                }
                results.append(record)

                # Persist intermediate results incrementally
                RESULTS_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
                with open(RESULTS_JSON_PATH, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2)

            except Exception as e:
                print(f" FAILED with exception: {e}")
                results.append({
                    "run_index": run_index,
                    "test_id": tc_id,
                    "prompt_version": ver,
                    "error": str(e),
                    "is_simulated": False,
                    "api_error": True
                })

            # Small delay to stay within API rate limits
            time.sleep(1.5)

    # Save JSON results
    RESULTS_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[OK] Saved all {len(results)} execution records to {RESULTS_JSON_PATH}")

    # Generate Markdown Summary
    md_lines = [
        "# Complete 40-Run Prompt Engineering Test Matrix",
        "",
        f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "**Evaluation Model:** Google Gemini (OpenAI Compatible Endpoint: `gemini-3-flash-preview`)",
        f"**Total Execution Runs:** {len(results)} (10 Test Cases x 4 Prompt Versions)",
        "",
        "## Summary Table",
        "",
        "| Run | TC ID | Type | Version | Score | Tech /4 | Comp /2 | Rel /2 | Comm /2 | Source | Model |",
        "|:---:|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|"
    ]

    real_count = 0
    sim_count = 0
    err_count = 0

    for r in results:
        idx = r.get("run_index")
        t_id = r.get("test_id")
        t_type = r.get("test_type", "")
        p_ver = r.get("prompt_version", "").upper()
        sc = r.get("model_score", 0.0)
        bd = r.get("score_breakdown", {})
        tech = bd.get("technical_accuracy", 0.0)
        comp = bd.get("completeness", 0.0)
        rel = bd.get("relevance", 0.0)
        comm = bd.get("communication_clarity", 0.0)
        sim = r.get("is_simulated", False)
        err = r.get("api_error", False)
        mod = r.get("model_used", "-")

        if err:
            src = "API ERROR"
            err_count += 1
        elif sim:
            src = "SIMULATED"
            sim_count += 1
        else:
            src = "REAL LLM"
            real_count += 1

        md_lines.append(
            f"| {idx} | {t_id} | {t_type} | {p_ver} | **{sc}** | {tech} | {comp} | {rel} | {comm} | {src} | `{mod}` |"
        )

    md_lines.extend([
        "",
        "## Execution Statistics",
        "",
        f"- **Real LLM Successful Runs:** {real_count} / {len(results)} ({real_count/len(results)*100:.1f}%)",
        f"- **Simulated Fallback Runs:** {sim_count} / {len(results)} ({sim_count/len(results)*100:.1f}%)",
        f"- **API Provider Errors:** {err_count} / {len(results)}",
        "",
        "## Detailed Analysis by Test Case",
        ""
    ])

    # Group by TC ID
    by_tc: Dict[str, List[Dict[str, Any]]] = {}
    for r in results:
        by_tc.setdefault(r["test_id"], []).append(r)

    for tc_id, runs in by_tc.items():
        q = runs[0].get("question", "")
        a = runs[0].get("student_answer", "")
        ttype = runs[0].get("test_type", "")

        md_lines.extend([
            f"### {tc_id}: {ttype}",
            f"- **Question:** {q}",
            f"- **Student Answer:** {a if a else '[BLANK / EMPTY INPUT]'}",
            "",
            "| Version | Score /10 | Breakdown (T/C/R/C) | Performance Tier | Follow-up Question Preview |",
            "|:---:|:---:|:---:|:---|:---|"
        ])

        for r in runs:
            v = r["prompt_version"].upper()
            sc = r.get("model_score", 0.0)
            bd = r.get("score_breakdown", {})
            bd_str = f"{bd.get('technical_accuracy',0)}/{bd.get('completeness',0)}/{bd.get('relevance',0)}/{bd.get('communication_clarity',0)}"
            tier = r.get("performance_tier", "")
            fu = r.get("follow_up_question", "")[:65] + "..." if r.get("follow_up_question") else "-"
            md_lines.append(f"| {v} | {sc} | {bd_str} | {tier} | {fu} |")

        md_lines.append("")

    RESULTS_MD_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_MD_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    print(f"[OK] Generated experiment report at {RESULTS_MD_PATH}")


if __name__ == "__main__":
    main()
