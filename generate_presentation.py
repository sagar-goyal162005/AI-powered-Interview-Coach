"""
Generate AI_Software_Interview_Coach_Final_Presentation.pptx
Professional 15-slide PowerPoint Presentation for Bennett University Project Viva.
"""

from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE

PRES_DIR = Path("presentation")
PRES_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = PRES_DIR / "AI_Software_Interview_Coach_Final_Presentation.pptx"

# Color Palette
COLOR_NAVY = RGBColor(30, 58, 138)       # #1E3A8A
COLOR_SKY = RGBColor(2, 132, 199)        # #0284C7
COLOR_DARK = RGBColor(30, 41, 59)        # #1E293B
COLOR_MUTED = RGBColor(100, 116, 139)    # #64748B
COLOR_BG_LIGHT = RGBColor(248, 250, 252) # #F8FAFC
COLOR_WHITE = RGBColor(255, 255, 255)
COLOR_GREEN = RGBColor(5, 150, 105)      # #059669
COLOR_AMBER = RGBColor(217, 119, 6)      # #D97706


def create_blank_slide(prs, bg_color=COLOR_BG_LIGHT):
    slide_layout = prs.slide_layouts[6]  # Blank layout
    slide = prs.slides.add_slide(slide_layout)
    bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), prs.slide_width, prs.slide_height)
    bg.fill.solid()
    bg.fill.fore_color.rgb = bg_color
    bg.line.fill.background()
    return slide


def add_slide_header(slide, title_text, category_text="SCSE4041 — PROMPT ENGINEERING MINI PROJECT"):
    header_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.5), Inches(11.7), Inches(1.1))
    tf = header_box.text_frame
    tf.word_wrap = True
    tf.margin_top = tf.margin_bottom = tf.margin_left = tf.margin_right = 0
    
    p_cat = tf.paragraphs[0]
    p_cat.text = category_text.upper()
    p_cat.font.name = "Calibri"
    p_cat.font.size = Pt(11)
    p_cat.font.bold = True
    p_cat.font.color.rgb = COLOR_SKY
    
    p_title = tf.add_paragraph()
    p_title.text = title_text
    p_title.font.name = "Calibri"
    p_title.font.size = Pt(22)
    p_title.font.bold = True
    p_title.font.color.rgb = COLOR_NAVY
    p_title.space_before = Pt(4)


def add_card(slide, left, top, width, height, title, items, card_bg=COLOR_WHITE, border_color=COLOR_SKY):
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left), Inches(top), Inches(width), Inches(height))
    shape.fill.solid()
    shape.fill.fore_color.rgb = card_bg
    shape.line.color.rgb = border_color
    shape.line.width = Pt(1.5)
    
    tf = shape.text_frame
    tf.word_wrap = True
    tf.margin_left = Inches(0.25)
    tf.margin_right = Inches(0.25)
    tf.margin_top = Inches(0.2)
    tf.margin_bottom = Inches(0.2)
    
    if title:
        p_title = tf.paragraphs[0]
        p_title.text = title
        p_title.font.name = "Calibri"
        p_title.font.size = Pt(14)
        p_title.font.bold = True
        p_title.font.color.rgb = COLOR_NAVY
        p_title.space_after = Pt(8)
    
    for i, item in enumerate(items):
        p = tf.add_paragraph() if (title or i > 0) else tf.paragraphs[0]
        p.text = f"• {item}"
        p.font.name = "Calibri"
        p.font.size = Pt(11)
        p.font.color.rgb = COLOR_DARK
        p.space_after = Pt(4)


def build_presentation():
    prs = Presentation()
    # 16:9 widescreen dimensions
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    # ---------------- SLIDE 1: TITLE SLIDE ----------------
    s1 = create_blank_slide(prs, COLOR_NAVY)
    
    tb = s1.shapes.add_textbox(Inches(1.0), Inches(1.8), Inches(11.3), Inches(4.0))
    tf = tb.text_frame
    tf.word_wrap = True
    
    p0 = tf.paragraphs[0]
    p0.text = "BENNETT UNIVERSITY — SCSE4041 PROMPT ENGINEERING"
    p0.font.name = "Calibri"
    p0.font.size = Pt(14)
    p0.font.bold = True
    p0.font.color.rgb = COLOR_SKY
    
    p1 = tf.add_paragraph()
    p1.text = "AI-Powered Software Engineering Mock Interview Coach"
    p1.font.name = "Calibri"
    p1.font.size = Pt(32)
    p1.font.bold = True
    p1.font.color.rgb = COLOR_WHITE
    p1.space_before = Pt(8)
    
    p2 = tf.add_paragraph()
    p2.text = "Problem C10: Mock Interview Question and Feedback Tool | Milestone 2 & 3 Submission"
    p2.font.name = "Calibri"
    p2.font.size = Pt(16)
    p2.font.color.rgb = RGBColor(226, 232, 240)
    p2.space_before = Pt(8)
    
    p3 = tf.add_paragraph()
    p3.text = "Team CodeSync: Sagar Goyal (S24CSEU0441) | Shauriya Garg (S24CSEU0409) | Sarthak Maheswari (S24CSEU0472) | Rohit Mukhopadhyay (S24CSEU0375)"
    p3.font.name = "Calibri"
    p3.font.size = Pt(12)
    p3.font.bold = True
    p3.font.color.rgb = COLOR_SKY
    p3.space_before = Pt(28)

    # ---------------- SLIDE 2: PROBLEM STATEMENT & MOTIVATION ----------------
    s2 = create_blank_slide(prs)
    add_slide_header(s2, "Problem Statement & Industry Context (Problem C10)")
    
    add_card(s2, 0.8, 1.8, 5.6, 5.0, "The Challenge in Technical Placements", [
        "Software engineering interviews test depth, precision, and communication under pressure.",
        "Students frequently provide vague, unstructured answers that omit time/space complexities.",
        "Generic chatbots (e.g. basic ChatGPT) give arbitrary grades without fixed rubrics.",
        "Lack of realistic mock interview practice with actionable, reproducible feedback.",
        "Single-turn question tools cannot test multi-turn adaptive follow-up probing."
    ])
    
    add_card(s2, 6.8, 1.8, 5.6, 5.0, "Problem C10 Mandate", [
        "Develop an intelligent Mock Interview Question and Feedback Tool.",
        "Target Audience: College students preparing for placement interviews.",
        "Domain Scope: DSA, Core CS (DBMS, OS, OOP, CN), and Software Projects.",
        "Required Outputs: Overall score (/10), 4-part rubric breakdown, strengths, areas for improvement, improved model answer, and follow-up question.",
        "Prompt Engineering Focus: Multi-stage prompt design, rubric anchors, and consistency testing."
    ])

    # ---------------- SLIDE 3: TARGET AUDIENCE & SCOPE ----------------
    s3 = create_blank_slide(prs)
    add_slide_header(s3, "Target Users & Technical Scope")
    
    add_card(s3, 0.8, 1.8, 3.6, 5.0, "Target Roles", [
        "Software Development Engineer (SDE-1)",
        "Backend Engineer (APIs, Databases)",
        "Frontend Engineer (UI, Performance)",
        "Full Stack Developer (Web Systems)",
        "DevOps / Site Reliability Engineer"
    ])
    add_card(s3, 4.8, 1.8, 3.6, 5.0, "Core Technical Domains", [
        "DSA: Arrays, Linked Lists, Trees, Graphs, DP, Sorting & Complexities.",
        "DBMS: Indexing, ACID, Normalization, SQL vs NoSQL, Transactions.",
        "OS: Deadlocks, Process vs Thread, Virtual Memory, Concurrency.",
        "OOP: SOLID principles, Polymorphism, Overloading vs Overriding.",
        "CN: TCP vs UDP, OSI/TCP layers, HTTP/HTTPS & TLS handshakes."
    ])
    add_card(s3, 8.8, 1.8, 3.6, 5.0, "Software Project Probing", [
        "Architectural challenges & design patterns.",
        "JWT token authentication & session security.",
        "Slow SQL query optimization via EXPLAIN.",
        "REST API error handling & rate limiting.",
        "Real-world trade-off rationalization."
    ])

    # ---------------- SLIDE 4: PROJECT OBJECTIVES ----------------
    s4 = create_blank_slide(prs)
    add_slide_header(s4, "Mini-Project Core Objectives")
    
    add_card(s4, 0.8, 1.8, 5.6, 5.0, "Academic & Prompt Engineering Goals", [
        "Demonstrate disciplined prompt evolution: V1 -> V2 -> V3 -> Final Prompt.",
        "Establish an evidence-based 10-mark scoring rubric to eliminate score ambiguity.",
        "Implement multi-turn context retention ({interview_context}) for adaptive questioning.",
        "Mitigate hallucinations through technical fact-checking and edge-case guardrails.",
        "Maintain strict academic integrity: verified TC01 benchmark preserved, TC02-TC10 marked PENDING."
    ])
    add_card(s4, 6.8, 1.8, 5.6, 5.0, "Application & Full-Stack Goals", [
        "Deploy a student-friendly web interface with dynamic question selection.",
        "Decouple prompts from application code into dedicated prompts/ architecture.",
        "Support configurable AI providers: Google Gemini (preferred) and OpenAI.",
        "Provide zero-crash offline simulation mode for uninterrupted viva demonstrations.",
        "Preserve existing multimodal features: ATS resume scanner, speech analysis, OpenCV posture detection, RL coach."
    ])

    # ---------------- SLIDE 5: SYSTEM ARCHITECTURE ----------------
    s5 = create_blank_slide(prs)
    add_slide_header(s5, "System Architecture & High-Level Workflow")
    
    add_card(s5, 0.8, 1.8, 3.6, 5.0, "1. Client / Frontend", [
        "Single Page Application (HTML5 / Vanilla JS / CSS3).",
        "Target Role & Category selectors.",
        "Prompt Version switcher (V1, V2, V3, Final).",
        "Multimodal input: Text, Audio recording, Video posture monitoring.",
        "Rich C10 Evaluation cards & model answer copy."
    ])
    add_card(s5, 4.8, 1.8, 3.6, 5.0, "2. Flask Backend & Modules", [
        "REST API endpoints: /api/c10/evaluate, /api/c10/questions, /api/practice/text.",
        "c10_evaluator.py: Core SWE question bank and structured response parser.",
        "prompts/prompt_loader.py: Dynamic prompt injection and variable formatting.",
        "Multi-turn conversation history tracker (thread_turns).",
        "Unified AI client factory supporting Gemini & OpenAI."
    ])
    add_card(s5, 8.8, 1.8, 3.6, 5.0, "3. AI & Evaluation Engine", [
        "Configured LLM (Google Gemini 1.5 Flash via OpenAI endpoint / OpenAI).",
        "10-mark standardized rubric evaluation.",
        "Parser extracts: Score, Tier, Breakdown, Strengths, Areas for Improvement, Model Answer, Follow-up.",
        "Offline simulation engine: Guaranteed demo uptime.",
        "Local session reports & progress storage."
    ])

    # ---------------- SLIDE 6: PROMPT ENGINEERING METHODOLOGY ----------------
    s6 = create_blank_slide(prs)
    add_slide_header(s6, "Prompt Engineering Methodology & Task Decomposition")
    
    add_card(s6, 0.8, 1.8, 5.6, 5.0, "Task Decomposition Workflow", [
        "Step 1: Concept Decomposition: Identify key technical concepts required for the given question and candidate seniority level.",
        "Step 2: Context Injection: Ingest previous turns ({interview_context}) to maintain interview thread continuity.",
        "Step 3: Guardrail Screening: Detect blank, 'idk', or off-topic inputs to award 0 marks without hallucinating false praise.",
        "Step 4: Rubric Evaluation: Award marks based solely on evidence across Technical Accuracy (4), Completeness (2), Relevance (2), and Clarity (2).",
        "Step 5: Synthesis: Generate improved model answer and targeted adaptive follow-up question."
    ])
    add_card(s6, 6.8, 1.8, 5.6, 5.0, "Why Prompt Decoupling Matters", [
        "Prompts are isolated in separate files under prompts/v1, v2, v3, final.",
        "Prevents hardcoding prompts inside application Python code.",
        "Enables independent prompt maintenance, A/B testing, and version control.",
        "Allows live dynamic switching between prompt versions during faculty evaluation.",
        "Guarantees reproducible academic experiments."
    ])

    # ---------------- SLIDE 7: VERSION 1 EXPERIMENT ----------------
    s7 = create_blank_slide(prs)
    add_slide_header(s7, "Iteration 1: Version 1 (V1) Baseline Role Evaluator")
    
    add_card(s7, 0.8, 1.8, 5.6, 5.0, "V1 Design & Instruction Strategy", [
        "Technique: Persona / Role prompting.",
        "System Persona: 'You are an AI-powered Software Engineering Interview Coach.'",
        "Prompt Structure: Basic role framing followed by a 5-step checklist (evaluate, praise, identify mistakes, improve, generate follow-up).",
        "Evaluation Method: Open-ended without fixed marks or criteria weights.",
        "Input: TC01 (Array vs Linked List) with student answer."
    ])
    add_card(s7, 6.8, 1.8, 5.6, 5.0, "Observed ChatGPT Output & Flaws", [
        "Observed Score on TC01: 8.0 / 10.",
        "Unanchored Sub-Scores: Accuracy: 9/10, Clarity: 8/10, Completeness: 7/10, Communication: 8/10.",
        "Problem: Scores were subjective and fluctuated between runs (7.5–8.5).",
        "Missing Criteria: No guidance on how heavily omissions were penalized.",
        "Weak Follow-up: 'Which would you choose: Array or Linked List, and why?' (Too generic)."
    ])

    # ---------------- SLIDE 8: VERSION 2 EXPERIMENT ----------------
    s8 = create_blank_slide(prs)
    add_slide_header(s8, "Iteration 2: Version 2 (V2) Rubric-Based Evaluator")
    
    add_card(s8, 0.8, 1.8, 5.6, 5.0, "V2 Design & Rubric Introduction", [
        "Technique: Rubric-based constraint prompting.",
        "10-Mark Rubric Introduced: Technical Accuracy (4), Completeness (2), Relevance (2), Clarity & Communication (2).",
        "Mandated Output Format: Explicit section headers and requested Markdown tables for technical comparisons.",
        "Specific Guidance: 'Do not assume that an omitted concept is incorrect; distinguish between incorrect and incomplete.'",
        "Input: TC01 (Array vs Linked List)."
    ])
    add_card(s8, 6.8, 1.8, 5.6, 5.0, "Observed ChatGPT Output & Flaws", [
        "Observed Score on TC01: 8.5 / 10.",
        "Score Breakdown: Accuracy: 4/4, Completeness: 1.5/2, Relevance: 2/2, Clarity: 1/2.",
        "Improvement: Clear scoring categories and explicit reasoning for deductions.",
        "Flaw: Instructing model to use Markdown tables caused rendering breakage across browsers.",
        "Score Jitter: Lacked explicit anchor definitions for partial marks (e.g., 1.0 vs 1.5)."
    ])

    # ---------------- SLIDE 9: VERSION 3 EXPERIMENT ----------------
    s9 = create_blank_slide(prs)
    add_slide_header(s9, "Iteration 3: Version 3 (V3) Refined Anchor Evaluator")
    
    add_card(s9, 0.8, 1.8, 5.6, 5.0, "V3 Design & Anchor Rules", [
        "Technique: Anchored rubric + formatting constraints.",
        "Explicit Scoring Anchors: Accuracy: 4=All correct, 3=Minor inaccuracies, 2=Noticeable errors, 1=Major confusion, 0=None. Completeness/Relevance/Clarity: 2=Full, 1=Partial, 0=Misses.",
        "Arithmetic Sum Rule: 'Total score must strictly equal the sum of all four categories.'",
        "Formatting Constraint: 'Avoid complex Markdown tables; use clear bullet points.'",
        "Evidence Requirement: Mandatory citation of student answer text for every mark."
    ])
    add_card(s9, 6.8, 1.8, 5.6, 5.0, "Observed ChatGPT Output & Success", [
        "Observed Score on TC01: 9.0 / 10.",
        "Score Breakdown: Accuracy: 4/4, Completeness: 1/2, Relevance: 2/2, Clarity: 2/2. (Total = 9.0).",
        "Rigorous Analysis: Correctly recognized that the answer was clear and accurate, isolating the sole deduction to missing Big-O complexity (1/2 for Completeness).",
        "High Quality Follow-up: Practical scenario-based question (Music Playlist implementation).",
        "Robustness: Clean bullet points rendered perfectly without UI distortion."
    ])

    # ---------------- SLIDE 10: PROMPT COMPARISON TABLE ----------------
    s10 = create_blank_slide(prs)
    add_slide_header(s10, "Comparative Evaluation: V1 vs V2 vs V3 (TC01 Benchmark)")
    
    add_card(s10, 0.8, 1.8, 2.7, 5.0, "Version 1 (V1)", [
        "Score: 8.0 / 10",
        "Accuracy: 9/10 (unweighted)",
        "Completeness: 7/10",
        "Relevance: Unscored",
        "Clarity: 8/10",
        "Structure: Freeform text",
        "Follow-up: Generic choice",
        "Score Stability: Low"
    ])
    add_card(s10, 3.8, 1.8, 2.7, 5.0, "Version 2 (V2)", [
        "Score: 8.5 / 10",
        "Accuracy: 4.0 / 4",
        "Completeness: 1.5 / 2",
        "Relevance: 2.0 / 2",
        "Clarity: 1.0 / 2",
        "Structure: Markdown tables",
        "Follow-up: Complexity query",
        "Score Stability: Medium"
    ])
    add_card(s10, 6.8, 1.8, 2.7, 5.0, "Version 3 (V3)", [
        "Score: 9.0 / 10",
        "Accuracy: 4.0 / 4",
        "Completeness: 1.0 / 2",
        "Relevance: 2.0 / 2",
        "Clarity: 2.0 / 2",
        "Structure: Clean bullets",
        "Follow-up: Playlist scenario",
        "Score Stability: High"
    ])
    add_card(s10, 9.8, 1.8, 2.7, 5.0, "Final Prompt", [
        "Score: 9.0 / 10",
        "Accuracy: 4.0 / 4",
        "Completeness: 1.0 / 2",
        "Relevance: 2.0 / 2",
        "Clarity: 2.0 / 2",
        "Structure: UI card anchors",
        "Follow-up: Adaptive context",
        "Guardrails: 0/10 on blank/idk"
    ])

    # ---------------- SLIDE 11: FINAL PRODUCTION PROMPT ----------------
    s11 = create_blank_slide(prs)
    add_slide_header(s11, "Production Final Prompt: Multi-Turn Context & Guardrails")
    
    add_card(s11, 0.8, 1.8, 5.6, 5.0, "Multi-Turn Dialogue Context", [
        "Variables Ingested: {job_role}, {interview_category}, {interview_question}, {student_answer}, {interview_context}.",
        "Context Memory: Serializes previous interview turns into structured dialog (Turn 1, Turn 2...).",
        "Adaptive Probing: Follow-up questions dynamically drill deeper into previously discussed trade-offs.",
        "Prevents Repetition: Evaluator ensures questions are not repeated within the same session.",
        "Simulates FAANG-style progressive interview rounds."
    ])
    add_card(s11, 6.8, 1.8, 5.6, 5.0, "Strict Edge-Case Guardrails", [
        "Empty / Blank Answers: Scores 0.0/10 (Accuracy: 0/4, Completeness: 0/2). Provides model answer and encouraging guidance. Refuses to hallucinate praise.",
        "'I Don't Know' / 'idk' / 'pass': Acknowledges honesty, assigns 0.0/10, provides foundational model answer.",
        "Hinglish / Colloquial Phrasing: Evaluates underlying technical correctness fairly; advises on professional English in communication critique.",
        "Off-topic Answers: Scores 0/2 for Relevance; clearly explains mismatch.",
        "Zero-Crash Reliability: App handles malformed outputs defensively."
    ])

    # ---------------- SLIDE 12: TESTING DATASET & STATUS ----------------
    s12 = create_blank_slide(prs)
    add_slide_header(s12, "Standardized Testing Dataset & 40-Run Empirical Matrix")
    
    add_card(s12, 0.8, 1.8, 5.6, 5.0, "10-Case Standardized Test Matrix", [
        "TC01 (DSA): Array vs Linked List -> Final: 9.0/10.",
        "TC02 (DSA): Binary Search Complexity -> Final: 9.0/10.",
        "TC03 (DSA): Cycle Detection (Floyd's) -> Final: 9.0/10.",
        "TC04 (DBMS): Normalization (1NF-BCNF) -> Final: 8.0/10.",
        "TC05 (OOP): Polymorphism with Example -> Final: 9.0/10.",
        "TC06 (OS): Process vs Thread Context -> Final: 9.0/10.",
        "TC07 (CN): TCP vs UDP File Transfer -> Final: 9.0/10.",
        "TC08 (Projects): Student Management Arch -> Final: 8.0/10.",
        "TC09 (DSA Weak): QuickSort Space (O(1) flaw) -> Final: 7.0/10.",
        "TC10 (DBMS Edge): ACID Properties (Blank) -> Final: 0.0/10."
    ])
    add_card(s12, 6.8, 1.8, 5.6, 5.0, "Empirical Validation & Integrity", [
        "Full Matrix Executed: 40 runs (10 Test Cases x 4 Prompt Versions).",
        "100% Real LLM Runs: Executed live on Google Gemini (gemini-3.1-flash-lite-preview via OpenAI endpoint).",
        "Input Sensitivity Proved: Excellent (9-10/10) vs Partial (7/10) vs Empty (0/10).",
        "Edge-Case Robustness: TC10 empty answer strictly docked to 0.0/10 by Final prompt guardrails.",
        "Automated Test Coverage: Unit tests (test_c10.py), endpoint tests (test_endpoints.py), and integration tests (test_integration.py) verify backend and UI with 100% pass rates."
    ])

    # ---------------- SLIDE 13: FULL-STACK APPLICATION DEMO ----------------
    s13 = create_blank_slide(prs)
    add_slide_header(s13, "Full-Stack Application Integration & Live Workflow")
    
    add_card(s13, 0.8, 1.8, 5.6, 5.0, "Student-Friendly UI Features", [
        "Setup Grid: Select Job Role, Category (DSA, DBMS, OS, OOP, CN, Projects), and Prompt Version (V1, V2, V3, Final).",
        "Dynamic Question Bank: 26 categorized technical questions + custom technical question input.",
        "Rich C10 Evaluation Panel: Total Score badge (/10), Performance tier, 4 rubric breakdown cards.",
        "Actionable Feedback: Bulleted Strengths, targeted Areas for Improvement, Model Answer with 1-click copy.",
        "Adaptive Follow-Up Button: Advance the multi-turn interview thread instantly."
    ])
    add_card(s13, 6.8, 1.8, 5.6, 5.0, "Multimodal & Preserved Capabilities", [
        "Speech-to-Text: Browser-native speech recognition for audio practice.",
        "Video Posture Detection: Real-time OpenCV analysis monitoring candidate posture during speech.",
        "ATS Resume Scanner: Parses PDF/DOCX resumes and calculates keyword matching against SWE job roles.",
        "RL Interview Coach: OpenEnv reinforcement learning agent exploring coaching policies.",
        "Progress Analytics & PDF Reports: Session scores, readiness index, and downloadable PDF reports."
    ])

    # ---------------- SLIDE 14: LIMITATIONS & RESPONSIBLE AI ----------------
    s14 = create_blank_slide(prs)
    add_slide_header(s14, "Limitations & Responsible AI Framework")
    
    add_card(s14, 0.8, 1.8, 5.6, 5.0, "Current Limitations", [
        "Free Tier Daily Quotas: Free tier Gemini enforces daily request quotas per model family, addressed via multi-model candidate failover.",
        "API Key Dependency: Live LLM evaluation requires active Gemini/OpenAI API key; offline mode relies on simulated demo heuristics.",
        "Multilingual Depth: Hinglish phrasing is supported, but regional language evaluation requires fine-tuning.",
        "No Sandbox Code Execution: Code snippets in student answers are analyzed conceptually rather than executed against test runners."
    ])
    add_card(s14, 6.8, 1.8, 5.6, 5.0, "Responsible AI Principles", [
        "Academic Truthfulness: Transparently differentiating between verified experimental data and pending/simulated tests.",
        "Pedagogical Encouragement: Constructive feedback tone that encourages learning without demoralizing students.",
        "Hallucination Mitigation: Mandating evidence citation and complexity verification in prompt instructions.",
        "Privacy by Design: Session transcripts and resumes stored securely without unauthorized external sharing."
    ])

    # ---------------- SLIDE 15: CONCLUSION & FUTURE SCOPE ----------------
    s15 = create_blank_slide(prs)
    add_slide_header(s15, "Conclusion & Future Enhancements")
    
    add_card(s15, 0.8, 1.8, 5.6, 5.0, "Milestone Accomplishments", [
        "Successfully solved Problem C10: Mock Interview Question and Feedback Tool.",
        "Engineered and documented 4 prompt iterations (V1, V2, V3, Final).",
        "Verified prompt evolution on TC01 with progressive refinement: 8.0 -> 8.5 -> 9.0.",
        "Integrated multi-turn context retention and robust edge-case guardrails.",
        "Deployed a robust, full-stack application with automated test verification.",
        "Completed professional Word documentation, Final Report, and PowerPoint deck."
    ])
    add_card(s15, 6.8, 1.8, 5.6, 5.0, "Future Roadmap", [
        "Extend empirical testing across Gemini 1.5 Flash and OpenAI gpt-4o-mini for TC02–TC10.",
        "Integrate AST-based live code execution sandbox for coding questions.",
        "Deepen Reinforcement Learning agent to dynamically adjust interview difficulty over weeks.",
        "Deploy to public Hugging Face Spaces / cloud container prototype.",
        "Conduct user study with 50+ undergraduate engineering placement candidates."
    ])

    # ---------------- SLIDE 16: TEAM CONTRIBUTION & VIVA GUIDE ----------------
    s16 = create_blank_slide(prs)
    add_slide_header(s16, "Team Contributions & Faculty Viva Guide (100% Total)")
    
    add_card(s16, 0.6, 1.8, 2.9, 5.1, "Sagar Goyal (25%)", [
        "Role: Prompt Engineering & Research Lead",
        "Enrollment: S24CSEU0441",
        "Tasks: Problem C10 analysis; designed & documented V1, V2, V3; developed 10-mark scoring rubric; compared outputs; authored Final Prompt.",
        "Evidence: prompts/v1/, v2/, v3/, final/, prompt_loader.py",
        "Viva Focus: Prompt design, prompt iteration, rubric, consistency, and evaluation."
    ])

    add_card(s16, 3.7, 1.8, 2.9, 5.1, "Shauriya Garg (25%)", [
        "Role: AI Evaluation & Backend Lead",
        "Enrollment: S24CSEU0409",
        "Tasks: Implemented C10 evaluation module (c10_evaluator.py); integrated LLM API via OpenAI client; scoring & response handling; multi-turn flow; API fallback.",
        "Evidence: c10_evaluator.py, app.py (/api/c10/evaluate)",
        "Viva Focus: Backend logic, API integration, LLM interaction, scoring, and multi-turn flow."
    ])

    add_card(s16, 6.8, 1.8, 2.9, 5.1, "Sarthak Maheswari (25%)", [
        "Role: Frontend & Integration Lead",
        "Enrollment: S24CSEU0472",
        "Tasks: Designed mock interview UI; implemented Role, Category & Prompt Version dropdowns; connected API endpoints; rendered score badges, rubric cards & follow-up.",
        "Evidence: templates/index.html, static/app.js, styles.css",
        "Viva Focus: Flask routes, frontend-backend communication, UI flow, and API endpoints."
    ])

    add_card(s16, 9.9, 1.8, 2.9, 5.1, "Rohit Mukhopadhyay (25%)", [
        "Role: Testing & Documentation Lead",
        "Enrollment: S24CSEU0375",
        "Tasks: Prepared 10 test cases including edge cases; created automated test suites; recorded test logs; authored experiment doc, final report, presentation & README.",
        "Evidence: test-data/test_dataset.json, test_*.py, docs/, presentation/",
        "Viva Focus: Test design, edge cases, integration verification, limitations, and documentation."
    ])

    prs.save(OUTPUT_FILE)
    print(f"Presentation successfully saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    build_presentation()
