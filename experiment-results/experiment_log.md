# Prompt Engineering Experiment Log

**Project Title:** AI-Powered Software Engineering Mock Interview Coach  
**Problem Statement:** C10 — Mock Interview Question and Feedback Tool  
**Academic Course:** SCSE4041 Prompt Engineering Mini Project (Bennett University)  
**Team:** CodeSync  
**Members:**  
- Sagar Goyal (S24CSEU0441)
- Shauriya Garg (S24CSEU0409)
- Sarthak Maheswari (S24CSEU0472)
- Rohit Mukhopadhyay (S24CSEU0375)

---

## 1. Prompt Iteration Log (V1 -> V2 -> V3 -> Final)

| Version | Core Changes Made | Why It Was Changed | Observed Experimental Result |
| :--- | :--- | :--- | :--- |
| **V1** | Role prompting as Interview Coach, open instructions, general feedback request, follow-up question generation. | Initial baseline experiment. | Feedback was helpful but scores were subjective and vague. No fixed rubric. Score on TC01: **8/10**. |
| **V2** | Introduced strict 10-mark scoring rubric (Tech Accuracy: 4, Completeness: 2, Relevance: 2, Clarity: 2) and Markdown table format. | Need for objective, reproducible scoring breakdown and clearer visual structure. | Score on TC01: **8.5/10**. Strengths and rubric breakdown were clear. However, complex Markdown tables proved fragile in rendering, and edge cases weren't handled. |
| **V3** | Explicit anchor scoring rules (4/3/2/1/0, 2/1/0), prohibited fragile markdown tables, added evidence-based rules and complexity verification. | To eliminate score ambiguity, ensure strict arithmetic consistency, and prevent formatting breakage. | Score on TC01: **9/10**. Highly consistent, clear evidence for every mark awarded, clean bulleted layout. |
| **Final** | Added multi-turn conversation context (`{interview_context}`), explicit edge case guardrails (empty answers, "idk", Hinglish, off-topic), and JSON-ready section anchors. | To support dynamic multi-turn interview flow and prevent hallucinations on blank/adversarial inputs in the live application. | Validated in application integration; handles blank and invalid inputs gracefully. |

---

## 2. Benchmark Experiment: TC01 (Verified Run)

- **Job Role:** Software Developer
- **Category:** Data Structures and Algorithms
- **Question:** What is the difference between an Array and a Linked List?
- **Student Answer:** "An array stores elements in continuous memory locations, while a linked list stores elements in nodes connected using pointers. Arrays allow faster access using indexes, whereas linked lists make insertion and deletion easier in certain situations."

### Experimental Comparison Table

| Metric | Version 1 (V1) | Version 2 (V2) | Version 3 (V3) | Final Prompt |
| :--- | :--- | :--- | :--- | :--- |
| **Prompt Type** | Role-based baseline | Rubric-based (10 marks) | Rubric + Anchor rules | Multi-turn + Edge guardrails |
| **Overall Score** | **8/10** | **8.5/10** | **9/10** | **9/10** |
| **Technical Accuracy** | 9/10 (unweighted) | 4/4 | 4/4 | 4/4 |
| **Completeness** | 7/10 (unweighted) | 1.5/2 | 1/2 | 1/2 |
| **Relevance** | Not separately scored | 2/2 | 2/2 | 2/2 |
| **Clarity & Communication** | 8/10 (unweighted) | 1/2 | 2/2 | 2/2 |
| **Table Formatting** | Basic table | Complex Markdown table | Structured bullet points | Structured bullet points |
| **Follow-up Question** | Basic choice question | Time complexity follow-up | Practical playlist scenario | Adaptive contextual follow-up |
| **Evaluation Evidence** | General statements | Partial rubric explanation | Rigorous evidence citations | Strict evidence + context |

---

## 3. Complete Test Matrix (10 Test Cases)

| Test ID | Category | Type | Input Question (Short) | Student Answer Summary | V1 Score | V2 Score | V3 Score | Final Score | Status |
| :---: | :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **TC01** | DSA | Normal | Array vs Linked List | Contiguous vs nodes, pointer access | 8.0/10 | 8.5/10 | 9.0/10 | 9.0/10 | **VERIFIED** |
| **TC02** | DBMS | Normal | Clustered vs Non-clustered | Physical sorting vs separate index | PENDING | PENDING | PENDING | PENDING | **PENDING** |
| **TC03** | OS | Partially Correct | Deadlock & 4 conditions | Recalls 3 conditions, misses 4th | PENDING | PENDING | PENDING | PENDING | **PENDING** |
| **TC04** | CN | Incorrect | TCP vs UDP for FTP | Inverted facts (claims UDP encrypts) | PENDING | PENDING | PENDING | PENDING | **PENDING** |
| **TC05** | OOP | Edge (Empty) | Overloading vs Overriding | [Blank / Empty Input] | PENDING | PENDING | PENDING | PENDING | **PENDING** |
| **TC06** | DSA | Edge (Hinglish) | Cycle detection in LL | Floyd's two-pointer in Hinglish | PENDING | PENDING | PENDING | PENDING | **PENDING** |
| **TC07** | Projects | Normal | JWT Authentication | Access token + HTTP-only refresh | PENDING | PENDING | PENDING | PENDING | **PENDING** |
| **TC08** | DSA | Partially Correct | QuickSort complexity | Correct time, flawed O(1) space | PENDING | PENDING | PENDING | PENDING | **PENDING** |
| **TC09** | DBMS | Edge (Off-topic) | ACID properties | CSS Grid & Flexbox response | PENDING | PENDING | PENDING | PENDING | **PENDING** |
| **TC10** | Projects | Normal | SQL query optimization | EXPLAIN ANALYZE, indexing | PENDING | PENDING | PENDING | PENDING | **PENDING** |

*Note on Experimental Integrity: In compliance with project guidelines, only TC01 has been executed and experimentally verified on ChatGPT. Test cases TC02 through TC10 are marked as PENDING and are not fabricated.*
