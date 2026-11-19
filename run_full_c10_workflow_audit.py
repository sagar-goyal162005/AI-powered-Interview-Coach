"""
Full C10 Workflow and System Verification Audit Script
Exercises all endpoints and workflows, validating schemas, scoring, and continuity.
"""

import json
import sys
from datetime import datetime
from app import app

def run_audit():
    print("=" * 60)
    print("AI-POWERED SOFTWARE ENGINEERING MOCK INTERVIEW COACH")
    print("C10 COMPREHENSIVE WORKFLOW & VERIFICATION AUDIT")
    print("=" * 60)
    
    client = app.test_client()
    
    # ---------------- 1. API META & MODEL CONFIGURATION ----------------
    print("\n[TEST 1] Verifying System Metadata & Model Configuration...")
    res = client.get("/api/meta")
    assert res.status_code == 200
    meta = res.get_json()
    print(f"  ✓ Status Code: 200 OK")
    print(f"  ✓ Active Model Engine: {meta.get('model_name')}")
    print(f"  ✓ Provider Endpoint: {meta.get('api_base_url')}")
    print(f"  ✓ API Key Present: {meta.get('has_api_key')}")
    print(f"  ✓ SWE Question Bank Loaded: {len(meta.get('swe_questions', {}))} categories")
    assert "Data Structures & Algorithms" in meta.get("swe_questions", {})
    assert "Core CS - DBMS" in meta.get("swe_questions", {})
    assert "Core CS - Operating Systems" in meta.get("swe_questions", {})
    assert "Core CS - OOP" in meta.get("swe_questions", {})
    assert "Core CS - Computer Networks" in meta.get("swe_questions", {})
    assert "Software Development Projects" in meta.get("swe_questions", {})

    # ---------------- 2. QUESTION BANK ENDPOINT ----------------
    print("\n[TEST 2] Verifying /api/c10/questions Categorization...")
    res = client.get("/api/c10/questions")
    assert res.status_code == 200
    q_data = res.get_json()
    cats = q_data.get("categories", {})
    total_q = sum(len(q_list) for q_list in cats.values())
    print(f"  ✓ Total Categories: {len(cats)}")
    print(f"  ✓ Total Questions: {total_q}")
    for cat_name, q_list in cats.items():
        print(f"    - {cat_name}: {len(q_list)} questions")

    # ---------------- 3. PROMPT ITERATION VERIFICATION (TC01) ----------------
    print("\n[TEST 3] Verifying Prompt Iterations on Benchmark TC01 (Array vs Linked List)...")
    tc01_q = "What is the difference between an Array and a Linked List?"
    tc01_ans = (
        "An array stores elements in continuous memory locations, while a linked list stores "
        "elements in nodes connected using pointers. Arrays allow faster access using indexes, "
        "whereas linked lists make insertion and deletion easier in certain situations."
    )
    
    versions = ["v1", "v2", "v3", "final"]
    expected_scores = {"v1": 8.0, "v2": 8.5, "v3": 9.0, "final": 9.0}
    
    for ver in versions:
        payload = {
            "job_role": "Software Developer",
            "category": "Data Structures & Algorithms",
            "question": tc01_q,
            "answer": tc01_ans,
            "prompt_version": ver
        }
        res = client.post("/api/c10/evaluate", json=payload)
        assert res.status_code == 200, f"Evaluation failed for {ver}: {res.status_code}"
        eval_out = res.get_json()
        score = eval_out.get("total_score")
        exp_score = expected_scores[ver]
        print(f"  ✓ Prompt {ver.upper()}: Score = {score}/10 (Expected: {exp_score}) | Model: {eval_out.get('model_used')}")
        assert score == exp_score, f"Mismatch for {ver}: expected {exp_score}, got {score}"
        assert "score_breakdown" in eval_out
        assert "strengths" in eval_out
        assert "areas_for_improvement" in eval_out
        assert "improved_answer" in eval_out
        assert "follow_up_question" in eval_out

    # ---------------- 4. EDGE CASE AUDIT ----------------
    print("\n[TEST 4] Verifying Strict Edge-Case Guardrails...")
    
    # 4A: Empty Answer
    res = client.post("/api/c10/evaluate", json={
        "job_role": "Software Developer",
        "category": "Core CS - OOP",
        "question": "What is method overloading?",
        "answer": "",
        "prompt_version": "final"
    })
    eval_empty = res.get_json()
    print(f"  ✓ Edge Case 1 (Empty Answer): Score = {eval_empty.get('total_score')}/10 | Tier = {eval_empty.get('performance_tier')}")
    assert eval_empty.get("total_score") == 0.0
    assert eval_empty.get("is_simulated") is True
    assert "No technical answer was provided" in eval_empty.get("summary_assessment")
    
    # 4B: Evasive 'idk' Answer
    res = client.post("/api/c10/evaluate", json={
        "job_role": "Software Developer",
        "category": "Core CS - Operating Systems",
        "question": "What is a deadlock?",
        "answer": "idk",
        "prompt_version": "final"
    })
    eval_idk = res.get_json()
    print(f"  ✓ Edge Case 2 ('idk' Evasive): Score = {eval_idk.get('total_score')}/10 | Tier = {eval_idk.get('performance_tier')}")
    assert eval_idk.get("total_score") == 0.0

    # ---------------- 5. MULTI-TURN CONTINUITY WORKFLOW ----------------
    print("\n[TEST 5] Testing Multi-Turn Interview Continuity Workflow...")
    # Login a test candidate
    client.post("/api/login", json={"username": "test_swe_student", "password": "Password123!"})
    
    # Turn 1
    t1_payload = {
        "job_role": "Software Developer",
        "category": "Data Structures & Algorithms",
        "question": "What is the difference between an Array and a Linked List?",
        "answer": tc01_ans,
        "prompt_version": "final",
        "thread_turns": []
    }
    t1_res = client.post("/api/practice/text", json=t1_payload)
    assert t1_res.status_code == 200
    t1_data = t1_res.get_json()
    followup_1 = t1_data.get("follow_up_question")
    print(f"  ✓ Turn 1 Evaluation Complete: Score = {t1_data['c10_evaluation']['total_score']}/10")
    print(f"    Follow-up Generated: {followup_1}")
    
    # Turn 2: Answering the follow-up
    t2_ans = (
        "In a music playlist where songs are frequently inserted and deleted from the beginning, "
        "a Linked List is preferred because head insertion is O(1) compared to array O(n) shifting. "
        "However, accessing a random track by song number takes O(n) traversal in a linked list."
    )
    t2_payload = {
        "job_role": "Software Developer",
        "category": "Data Structures & Algorithms",
        "root_question": "What is the difference between an Array and a Linked List?",
        "question": followup_1,
        "answer": t2_ans,
        "prompt_version": "final",
        "thread_turns": [
            {"question": t1_payload["question"], "answer": t1_payload["answer"]}
        ]
    }
    t2_res = client.post("/api/practice/text", json=t2_payload)
    assert t2_res.status_code == 200
    t2_data = t2_res.get_json()
    print(f"  ✓ Turn 2 Evaluation Complete: Score = {t2_data['c10_evaluation']['total_score']}/10")
    print(f"    Follow-up 2 Generated: {t2_data.get('follow_up_question')}")
    print(f"    Simulated Flag: {t2_data['c10_evaluation'].get('is_simulated')}")

    # ---------------- 6. REPORT & ANALYTICS PERSISTENCE ----------------
    print("\n[TEST 6] Verifying Session Report Persistence...")
    rep_res = client.get("/api/reports")
    assert rep_res.status_code == 200
    reports = rep_res.get_json().get("reports", [])
    print(f"  ✓ Persisted Reports Count: {len(reports)}")
    prog_res = client.get("/api/progress")
    assert prog_res.status_code == 200
    prog = prog_res.get_json()
    print(f"  ✓ Progress Analytics: {prog.get('sessions', 0)} sessions tracked")

    print("\n" + "=" * 60)
    print("AUDIT RESULT: 100% OF C10 WORKFLOWS & CHECKS PASSED!")
    print("=" * 60)

if __name__ == "__main__":
    run_audit()
