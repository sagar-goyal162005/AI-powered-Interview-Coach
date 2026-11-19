import sys
from pathlib import Path
from c10_evaluator import evaluate_student_answer, SWE_QUESTION_BANK

def test_tc01_versions():
    question = "What is the difference between an Array and a Linked List?"
    answer = (
        "An array stores elements in continuous memory locations, while a linked list "
        "stores elements in nodes connected using pointers. Arrays allow faster access "
        "using indexes, whereas linked lists make insertion and deletion easier in certain situations."
    )
    r1 = evaluate_student_answer("Software Developer", "DSA", question, answer, prompt_version="v1")
    r2 = evaluate_student_answer("Software Developer", "DSA", question, answer, prompt_version="v2")
    r3 = evaluate_student_answer("Software Developer", "DSA", question, answer, prompt_version="v3")
    
    print(f"TC01 V1 Score: {r1['total_score']} / 10 (Expected: 8.0)")
    print(f"TC01 V2 Score: {r2['total_score']} / 10 (Expected: 8.5)")
    print(f"TC01 V3 Score: {r3['total_score']} / 10 (Expected: 9.0)")
    
    assert r1['total_score'] == 8.0, f"Expected 8.0, got {r1['total_score']}"
    assert r2['total_score'] == 8.5, f"Expected 8.5, got {r2['total_score']}"
    assert r3['total_score'] == 9.0, f"Expected 9.0, got {r3['total_score']}"
    print("TC01 verified test passed!")

def test_empty_answer():
    r_empty = evaluate_student_answer("Software Developer", "Core CS - OOP", "What is OOP?", "", prompt_version="final")
    print(f"Empty Answer Score: {r_empty['total_score']} (Expected: 0.0), is_simulated: {r_empty.get('is_simulated')}")
    assert r_empty['total_score'] == 0.0
    print("Empty answer test passed!")

def test_question_bank():
    categories = list(SWE_QUESTION_BANK.keys())
    print(f"Categories loaded: {len(categories)}")
    total_q = sum(len(q_list) for q_list in SWE_QUESTION_BANK.values())
    print(f"Total SWE questions in bank: {total_q}")
    assert total_q >= 10
    print("Question bank test passed!")

if __name__ == "__main__":
    test_tc01_versions()
    test_empty_answer()
    test_question_bank()
    print("All preliminary C10 module tests passed successfully!")
