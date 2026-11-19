import json
from app import app

def test_full_application():
    client = app.test_client()
    
    # 1. Test Static & HTML
    res = client.get("/")
    assert res.status_code == 200
    html = res.get_data(as_text=True)
    assert "c10-setup-grid" in html, "c10-setup-grid missing from index.html"
    assert "c10FeedbackContainer" in html, "c10FeedbackContainer missing from index.html"
    assert "categorySelect" in html
    assert "promptVersionSelect" in html
    print("[PASS] index.html served with C10 components")
    
    # 2. Test User Signup & Login Flow
    signup_data = {
        "username": "test_swe_student",
        "password": "Password123!",
        "confirm_password": "Password123!",
        "full_name": "Test Student",
        "email": "teststudent@example.com",
        "phone": "9876543210",
        "university": "Bennett University",
        "college_year": "3",
        "degree": "B.Tech CSE",
        "major": "Computer Science",
        "linkedin": "https://linkedin.com/in/teststudent",
        "about": "Aspiring Software Development Engineer"
    }
    
    login_res = client.post("/api/login", json={"username": "test_swe_student", "password": "Password123!"})
    if login_res.status_code != 200:
        res = client.post("/api/signup", json=signup_data)
        assert res.status_code in (200, 201), f"Signup failed: {res.status_code} - {res.get_data(as_text=True)}"
        print("[PASS] User signup successful")
        login_res = client.post("/api/login", json={"username": "test_swe_student", "password": "Password123!"})
        assert login_res.status_code == 200
        print("[PASS] User login successful")
    else:
        print("[PASS] User login successful directly")
    
    # 3. Test Practice Text endpoint with Problem C10 evaluation (Turn 1)
    turn1_payload = {
        "job_role": "Software Developer",
        "category": "Data Structures & Algorithms",
        "question": "What is the difference between an Array and a Linked List?",
        "answer": "An array stores elements in continuous memory locations, while a linked list stores elements in nodes connected using pointers. Arrays allow faster access using indexes, whereas linked lists make insertion and deletion easier in certain situations.",
        "prompt_version": "v3",
        "thread_turns": []
    }
    res = client.post("/api/practice/text", json=turn1_payload)
    assert res.status_code == 200, f"/api/practice/text failed: {res.status_code} - {res.get_data(as_text=True)}"
    turn1_data = res.get_json()
    assert "c10_evaluation" in turn1_data, "c10_evaluation missing in /api/practice/text"
    c10_eval = turn1_data["c10_evaluation"]
    assert c10_eval["total_score"] == 9.0, f"Expected 9.0 for TC01 V3, got {c10_eval['total_score']}"
    assert "score_breakdown" in c10_eval
    assert "follow_up_question" in turn1_data
    follow_up_q = turn1_data["follow_up_question"]
    print(f"[PASS] Turn 1 C10 Evaluation Passed! Score: {c10_eval['total_score']}/10.")
    print(f"       Follow-up Question generated: {follow_up_q[:65]}...")
    
    # 4. Test Multi-Turn Continuity (Turn 2)
    turn2_payload = {
        "job_role": "Software Developer",
        "category": "Data Structures & Algorithms",
        "root_question": "What is the difference between an Array and a Linked List?",
        "question": follow_up_q,
        "answer": "For frequent insertions at the beginning, I would choose a Linked List because inserting at the head takes O(1) time by updating the head pointer, whereas an array requires shifting all n elements taking O(n) time.",
        "prompt_version": "final",
        "thread_turns": [
            {
                "question": "What is the difference between an Array and a Linked List?",
                "answer": turn1_payload["answer"]
            }
        ]
    }
    res = client.post("/api/practice/text", json=turn2_payload)
    assert res.status_code == 200
    turn2_data = res.get_json()
    assert "c10_evaluation" in turn2_data
    print(f"[PASS] Turn 2 Multi-Turn Evaluation Passed! Score: {turn2_data['c10_evaluation']['total_score']}/10")
    
    # 5. Test Reports & Progress
    rep_res = client.get("/api/reports")
    assert rep_res.status_code == 200
    prog_res = client.get("/api/progress")
    assert prog_res.status_code == 200
    print("[PASS] User reports and progress analytics functional")
    
    print("[PASS] Core existing systems verified intact")

if __name__ == "__main__":
    test_full_application()
    print("\n==========================================")
    print("ALL APPLICATION INTEGRATION TESTS PASSED!!")
    print("==========================================")
