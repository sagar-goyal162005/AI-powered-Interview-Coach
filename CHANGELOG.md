# Changelog — AI-Powered Software Engineering Mock Interview Coach

All notable changes to this project will be documented in this file.
This project follows academic project milestones for Bennett University (SCSE4041 Prompt Engineering Mini Project, Problem C10).

### Team CodeSync
- **Sagar Goyal** (S24CSEU0441)
- **Shauriya Garg** (S24CSEU0409)
- **Sarthak Maheswari** (S24CSEU0472)
- **Rohit Mukhopadhyay** (S24CSEU0375)

---

## [Unreleased / Milestone 2] - 2026-11-17

### Fixed
- **Root Cause Fix for Repeated 7.5/10 Simulated Output:**
  - Diagnosed that Google retired `gemini-1.5-flash` on the OpenAI-compatible endpoint, resulting in 404 NotFound.
  - Eliminated silent exception swallowing in `c10_evaluator.py` that previously defaulted to an offline demo template.
  - Configured active Google Gemini model (`gemini-3.1-flash-lite-preview` / `gemini-3-flash-preview`) with automatic multi-model candidate failover.
  - Transparent error reporting: API provider errors (401, 429 quota, connection timeout) are now surfaced explicitly with `api_error: True` instead of being masked as fake scores.
  - Dynamic simulation heuristics: Offline demo mode now uses input-sensitive heuristic scoring (empty=0, short=2-5, comprehensive=7-8.5) and is explicitly labeled `[SIMULATED DEMO - NO API KEY]`.

### Added
- **40-Run Empirical Test Matrix (`test-data/test_execution_matrix.json` & `experiment-results/full_test_matrix_results.md`):**
  - Executed 10 comprehensive test cases across all 4 prompt versions (V1, V2, V3, Final).
  - 100% genuine LLM runs (40/40 successful real model calls, 0 simulated runs, 0 API errors).
  - Demonstrated clear prompt evolution: V1 uncalibrated -> V2 rubric-based -> V3 anchored scoring -> Final multi-turn context + edge-case guardrails.
  - Proved input sensitivity: Excellent (9-10/10), Partial (7/10 for TC09 QuickSort space error), and Empty (0.0/10 for TC10).
- **Updated Project Documentation & Academic Deliverables:**
  - Regenerated `docs/Prompt_Engineering_Experiment_Documentation.docx` with Section 7 full empirical results table.
  - Regenerated `docs/AI_Software_Interview_Coach_Final_Report.docx` with Chapter 6 & 7 empirical testing discussion.
  - Regenerated `presentation/AI_Software_Interview_Coach_Final_Presentation.pptx` with Slide 12 40-run verification matrix.

### Added (Initial C10 Release)
- **Problem C10 Prompt Architecture (`prompts/`):**
  - `prompts/v1/evaluator_v1.txt`: Baseline role-prompted evaluator with open-ended feedback generation.
  - `prompts/v1/response_v1.md`: Verified ChatGPT benchmark output for TC01 (Score: 8.0/10).
  - `prompts/v2/evaluator_v2.txt`: Structured evaluator with fixed 10-mark scoring rubric and table formatting.
  - `prompts/v2/response_v2.md`: Verified ChatGPT benchmark output for TC01 (Score: 8.5/10).
  - `prompts/v3/evaluator_v3.txt`: Refined evaluator with explicit scoring anchors (4/3/2/1/0, 2/1/0) and evidence verification rules.
  - `prompts/v3/response_v3.md`: Verified ChatGPT benchmark output for TC01 (Score: 9.0/10).
  - `prompts/final/evaluator_final.txt`: Production-grade final prompt featuring multi-turn interview context (`{interview_context}`) and edge-case guardrails (empty answers, Hinglish phrasing, irrelevant/off-topic answers).
  - `prompts/prompt_loader.py`: Reusable Python loader module to dynamically inject variables and format prompts.

- **Standardized Test Dataset (`test-data/`):**
  - `test-data/test_dataset.json`: 10 standardized technical test cases covering:
    - Data Structures & Algorithms (Array vs Linked List, Floyd's cycle detection, QuickSort complexities).
    - Core CS: DBMS (Clustered vs Non-clustered indexing, ACID properties).
    - Core CS: Operating Systems (Deadlock & Coffman conditions).
    - Core CS: Object-Oriented Programming (Overloading vs Overriding).
    - Core CS: Computer Networks (TCP vs UDP for FTP).
    - Software Development Projects (JWT authentication security, SQL query optimization).
    - 3 distinct edge cases: Empty/blank answer (TC05), Hinglish phrasing (TC06), completely off-topic answer (TC09).
  - Preserved verified TC01 benchmark outputs; clearly designated TC02–TC10 as **PENDING** to preserve experimental integrity.

- **Core C10 Evaluation Module (`c10_evaluator.py`):**
  - `SWE_QUESTION_BANK`: Categorized repository of 26 technical software engineering placement interview questions.
  - `get_ai_client()`: Configurable AI provider factory supporting Google Gemini (OpenAI-compatible endpoint), OpenAI, and offline simulation fallback.
  - `parse_evaluation_markdown()`: Robust parser extracting total score, 4-tier rubric metrics, strengths, areas for improvement, model answer, and follow-up question.
  - `evaluate_student_answer()`: Unified evaluation engine supporting live LLM execution, verified benchmark replay for TC01, and graceful offline demo mode.

- **Application Integration (`app.py`):**
  - New endpoints: `GET /api/c10/questions` and `POST /api/c10/evaluate`.
  - Integrated C10 evaluation into existing `/api/practice/text` and `/api/practice/audio` workflows.
  - Multi-turn context preservation passing conversation history across turns.

- **Student-Friendly Frontend UI (`templates/index.html`, `static/app.js`, `static/styles.css`):**
  - Added Job Role selector (Software Developer, Backend, Frontend, Full Stack, DevOps, Data Engineer).
  - Added Interview Category selector with dynamic question population.
  - Added Prompt Version selector (Final, V3, V2, V1) for live academic comparison.
  - Added rich C10 Feedback Panel displaying score badge, rubric breakdown cards, strengths, areas for improvement, model answer with 1-click copy, and adaptive follow-up question button.

- **Automated Test Suites:**
  - `test_c10.py`: Unit tests for prompt loading, TC01 score verification, and empty answer guardrails.
  - `test_integration.py`: End-to-end integration test validating frontend rendering, user auth, Turn 1 evaluation, Turn 2 multi-turn continuation, and existing subsystem stability.

### Changed
- Preserved existing Flask architecture, ATS resume scanner, video posture detection via OpenCV, and reinforcement learning modules.
- Updated `.env_example` and `.env` to default to Google Gemini OpenAI-compatible API while keeping provider and model configurable.

---

## [Milestone 1] - 2026-10-11
- Initial repository setup and baseline interview coach prototype.
- Rule-based text sentiment and keyword matching.
- Basic reinforcement learning environment (`InterviewCoachEnv`).
- Audio recording and OpenCV posture monitoring.
- ATS resume scanner and PDF report generation.
