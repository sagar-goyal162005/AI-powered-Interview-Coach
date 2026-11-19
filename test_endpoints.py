import json
from app import app

def test_endpoints():
    client = app.test_client()
    
    # 1. Test /api/meta
    res = client.get("/api/meta")
    assert res.status_code == 200, f"/api/meta failed: {res.status_code}"
    meta = res.get_json()
    assert "swe_questions" in meta, "swe_questions missing in /api/meta"
    assert "Data Structures & Algorithms" in meta["swe_questions"]
    print("GET /api/meta passed! Model:", meta.get("model_name"))
    
    # 2. Test /api/c10/questions
    res = client.get("/api/c10/questions")
    assert res.status_code == 200, f"/api/c10/questions failed: {res.status_code}"
    data = res.get_json()
    assert "categories" in data
    print("GET /api/c10/questions passed! Categories count:", len(data["categories"]))
    
    # 3. Test /api/c10/evaluate on TC01 with V3
    tc01_payload = {
        "job_role": "Software Developer",
        "category": "Data Structures & Algorithms",
        "question": "What is the difference between an Array and a Linked List?",
        "answer": "An array stores elements in continuous memory locations, while a linked list stores elements in nodes connected using pointers. Arrays allow faster access using indexes, whereas linked lists make insertion and deletion easier in certain situations.",
        "prompt_version": "v3"
    }
    res = client.post("/api/c10/evaluate", json=tc01_payload)
    assert res.status_code == 200, f"/api/c10/evaluate failed: {res.status_code}"
    eval_res = res.get_json()
    print("POST /api/c10/evaluate TC01 Score:", eval_res.get("total_score"))
    assert eval_res.get("total_score") == 9.0, f"Expected 9.0 for TC01 V3, got {eval_res.get('total_score')}"
    assert "score_breakdown" in eval_res
    assert "strengths" in eval_res
    assert "areas_for_improvement" in eval_res
    assert "improved_answer" in eval_res
    assert "follow_up_question" in eval_res
    print("TC01 evaluation structure verified successfully!")

    # 4. Test /api/c10/evaluate with edge case (empty answer)
    empty_payload = {
        "job_role": "Software Developer",
        "category": "Core CS - OOP",
        "question": "What is method overloading?",
        "answer": "",
        "prompt_version": "final"
    }
    res = client.post("/api/c10/evaluate", json=empty_payload)
    assert res.status_code == 200
    eval_empty = res.get_json()
    print("POST /api/c10/evaluate Empty Answer Score:", eval_empty.get("total_score"), "simulated:", eval_empty.get("is_simulated"))
    assert eval_empty.get("total_score") == 0.0
    print("Empty answer edge case verified successfully!")

if __name__ == "__main__":
    test_endpoints()
    print("ALL BACKEND ENDPOINTS PASSED VERIFICATION!")
