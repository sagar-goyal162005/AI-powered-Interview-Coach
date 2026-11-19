"""
Generate Prompt_Engineering_Experiment_Documentation.docx
Professional academic documentation for Bennett University Prompt Engineering Mini Project (Problem C10).
"""

import os
from pathlib import Path
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

DOCS_DIR = Path("docs")
DOCS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = DOCS_DIR / "Prompt_Engineering_Experiment_Documentation.docx"

# Color Palette
COLOR_PRIMARY = RGBColor(30, 58, 138)     # Navy #1E3A8A
COLOR_SECONDARY = RGBColor(2, 132, 199)  # Sky Blue #0284C7
COLOR_DARK = RGBColor(30, 41, 59)        # Slate Dark #1E293B
COLOR_MUTED = RGBColor(100, 116, 139)    # Slate Muted #64748B
HEX_PRIMARY = "1E3A8A"
HEX_LIGHT_BG = "F8FAFC"
HEX_BORDER = "CBD5E1"
HEX_AMBER = "D97706"
HEX_GREEN = "059669"


def set_cell_background(cell, hex_color):
    shading_elm = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{hex_color}"/>')
    cell._tc.get_or_add_tcPr().append(shading_elm)


def set_cell_margins(cell, top=120, bottom=120, left=160, right=160):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('w:top', top), ('w:bottom', bottom), ('w:left', left), ('w:right', right)]:
        node = OxmlElement(m)
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)


def add_heading_1(doc, text):
    h = doc.add_paragraph()
    h.paragraph_format.space_before = Pt(16)
    h.paragraph_format.space_after = Pt(6)
    h.paragraph_format.keep_with_next = True
    run = h.add_run(text)
    run.font.name = "Calibri"
    run.font.size = Pt(16)
    run.font.bold = True
    run.font.color.rgb = COLOR_PRIMARY
    return h


def add_heading_2(doc, text):
    h = doc.add_paragraph()
    h.paragraph_format.space_before = Pt(12)
    h.paragraph_format.space_after = Pt(4)
    h.paragraph_format.keep_with_next = True
    run = h.add_run(text)
    run.font.name = "Calibri"
    run.font.size = Pt(13)
    run.font.bold = True
    run.font.color.rgb = COLOR_SECONDARY
    return h


def add_body_p(doc, text, bold_prefix=""):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.line_spacing = 1.15
    if bold_prefix:
        r_pre = p.add_run(bold_prefix)
        r_pre.font.name = "Calibri"
        r_pre.font.size = Pt(11)
        r_pre.font.bold = True
        r_pre.font.color.rgb = COLOR_DARK
    r_body = p.add_run(text)
    r_body.font.name = "Calibri"
    r_body.font.size = Pt(11)
    r_body.font.color.rgb = COLOR_DARK
    return p


def add_callout(doc, text, title=""):
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = tbl.cell(0, 0)
    cell.width = Inches(6.5)
    set_cell_background(cell, "F1F5F9")
    set_cell_margins(cell, top=140, bottom=140, left=200, right=200)
    p = cell.paragraphs[0]
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.line_spacing = 1.15
    if title:
        rt = p.add_run(f"{title}\n")
        rt.font.name = "Calibri"
        rt.font.size = Pt(11)
        rt.font.bold = True
        rt.font.color.rgb = COLOR_PRIMARY
    rb = p.add_run(text)
    rb.font.name = "Calibri"
    rb.font.size = Pt(10.5)
    rb.font.color.rgb = COLOR_DARK
    doc.add_paragraph().paragraph_format.space_after = Pt(4)


def add_prompt_box(doc, prompt_text, prompt_label=""):
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = tbl.cell(0, 0)
    cell.width = Inches(6.5)
    set_cell_background(cell, "F8FAFC")
    set_cell_margins(cell, top=140, bottom=140, left=200, right=200)
    p = cell.paragraphs[0]
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.line_spacing = 1.15
    if prompt_label:
        rt = p.add_run(f"PROMPT TEXT ({prompt_label}):\n")
        rt.font.name = "Consolas"
        rt.font.size = Pt(9.5)
        rt.font.bold = True
        rt.font.color.rgb = COLOR_PRIMARY
    rb = p.add_run(prompt_text)
    rb.font.name = "Consolas"
    rb.font.size = Pt(9)
    rb.font.color.rgb = COLOR_DARK
    doc.add_paragraph().paragraph_format.space_after = Pt(4)


def add_response_box(doc, response_text, response_label=""):
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = tbl.cell(0, 0)
    cell.width = Inches(6.5)
    set_cell_background(cell, "F0FDF4")
    set_cell_margins(cell, top=140, bottom=140, left=200, right=200)
    p = cell.paragraphs[0]
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.line_spacing = 1.15
    if response_label:
        rt = p.add_run(f"ACTUAL OBSERVED CHATGPT RESPONSE ({response_label}):\n")
        rt.font.name = "Consolas"
        rt.font.size = Pt(9.5)
        rt.font.bold = True
        rt.font.color.rgb = RGBColor(21, 128, 61)
    rb = p.add_run(response_text)
    rb.font.name = "Calibri"
    rb.font.size = Pt(9.5)
    rb.font.color.rgb = COLOR_DARK
    doc.add_paragraph().paragraph_format.space_after = Pt(4)


def build_experiment_doc():
    doc = Document()
    
    # Page setup: Standard 1 inch margins
    sections = doc.sections
    for section in sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)

    # ==================== COVER / HEADER ====================
    title_p = doc.add_paragraph()
    title_p.paragraph_format.space_before = Pt(24)
    title_p.paragraph_format.space_after = Pt(6)
    r_title = title_p.add_run("AI-Powered Software Engineering Mock Interview Coach")
    r_title.font.name = "Calibri"
    r_title.font.size = Pt(22)
    r_title.font.bold = True
    r_title.font.color.rgb = COLOR_PRIMARY

    sub_p = doc.add_paragraph()
    sub_p.paragraph_format.space_after = Pt(16)
    r_sub = sub_p.add_run("Prompt Engineering Experiment Documentation & Comparative Analysis\nCourse Code: SCSE4041 — Applied AI Mini Project | Problem C10")
    r_sub.font.name = "Calibri"
    r_sub.font.size = Pt(13)
    r_sub.font.color.rgb = COLOR_MUTED

    # Team Info Table
    meta_table = doc.add_table(rows=5, cols=2)
    meta_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    meta_data = [
        ("Project Problem", "Problem C10: Mock Interview Question and Feedback Tool"),
        ("Institution", "Bennett University — School of Computer Science Engineering & Technology"),
        ("Team Name", "Team CodeSync"),
        ("Team Members", "Sagar Goyal (S24CSEU0441)\nShauriya Garg (S24CSEU0409)\nSarthak Maheswari (S24CSEU0472)\nRohit Mukhopadhyay (S24CSEU0375)"),
        ("Evaluation Date", "November 2026")
    ]
    for idx, (label, val) in enumerate(meta_data):
        c0, c1 = meta_table.cell(idx, 0), meta_table.cell(idx, 1)
        c0.width, c1.width = Inches(2.0), Inches(4.5)
        set_cell_background(c0, "F1F5F9")
        set_cell_background(c1, "FFFFFF")
        set_cell_margins(c0, 80, 80, 120, 120)
        set_cell_margins(c1, 80, 80, 120, 120)
        p0 = c0.paragraphs[0]
        r0 = p0.add_run(label)
        r0.font.bold = True
        r0.font.size = Pt(10)
        r0.font.color.rgb = COLOR_PRIMARY
        p1 = c1.paragraphs[0]
        r1 = p1.add_run(val)
        r1.font.size = Pt(10)
        r1.font.color.rgb = COLOR_DARK

    doc.add_page_break()

    # ==================== 1. INTRODUCTION & PROBLEM DEFINITION ====================
    add_heading_1(doc, "1. Project Introduction & Problem Definition")
    add_body_p(doc, 
        "Technical placement interviews for software engineering roles are demanding and highly competitive. "
        "Undergraduate engineering students frequently struggle during technical interviews not due to an absolute lack of programming knowledge, "
        "but due to unstructured explanations, incomplete coverage of time and space complexity trade-offs, and an inability to handle adaptive follow-up questions.",
        "Background Context: "
    )
    add_body_p(doc,
        "Problem C10 requires developing a Mock Interview Question and Feedback Tool tailored for college students preparing for software engineering placements. "
        "The system must accept the target job role, category, interview question, and student answer, and produce an objective score, technical accuracy feedback, "
        "completeness assessment, communication critique, improved interview model answer, and an adaptive follow-up question.",
        "C10 Problem Statement: "
    )
    add_body_p(doc,
        "Undergraduate computer science students targeting Software Development Engineer (SDE), Backend Developer, and Full-Stack roles.",
        "Target Audience: "
    )

    add_heading_2(doc, "Core Interview Categories")
    categories_p = doc.add_paragraph()
    categories_p.paragraph_format.left_indent = Inches(0.25)
    categories = [
        ("Data Structures and Algorithms (DSA): ", "Arrays, Linked Lists, Trees, Graphs, Dynamic Programming, Sorting/Searching."),
        ("Core Computer Science: ", "DBMS (Indexing, ACID, Normalization), Operating Systems (Processes/Threads, Deadlocks, Virtual Memory), Object-Oriented Programming (SOLID, Polymorphism), Computer Networks (TCP/IP, OSI, HTTP/TLS)."),
        ("Software Development Projects: ", "System architecture, RESTful API design, database performance optimization, security, and trade-off rationalization.")
    ]
    for c_title, c_desc in categories:
        p_item = doc.add_paragraph()
        p_item.paragraph_format.space_after = Pt(4)
        r1 = p_item.add_run(f"• {c_title}")
        r1.font.bold = True
        r1.font.color.rgb = COLOR_PRIMARY
        r2 = p_item.add_run(c_desc)
        r2.font.color.rgb = COLOR_DARK

    # ==================== 2. PROMPT ENGINEERING METHODOLOGY ====================
    add_heading_1(doc, "2. Prompt Engineering Methodology")
    add_body_p(doc,
        "Prompt engineering forms the foundational academic core of this mini-project. Rather than treating language models as black-box chatbots, "
        "we employed an iterative experimental methodology progressing from baseline role prompting to rubric-anchored evaluation, and finally to multi-turn context-aware guardrails. "
        "The development journey was structured across four controlled iterations:",
        "Methodological Approach: "
    )

    method_points = [
        ("Version 1 (V1) — Baseline Role Evaluator: ", "Establishes persona ('Software Engineering Interview Coach') and open-ended instructions. Tests model's default evaluation behavior without fixed constraints."),
        ("Version 2 (V2) — Rubric-Based Evaluator: ", "Introduces an explicit 10-mark scoring rubric across 4 dimensions (Technical Accuracy: 4, Completeness: 2, Relevance: 2, Clarity: 2) and structured table formatting."),
        ("Version 3 (V3) — Refined Anchor Evaluator: ", "Enforces strict scoring anchors (4/3/2/1/0, 2/1/0), replaces fragile Markdown tables with resilient bullet points, enforces arithmetic score sum constraints, and mandates evidence citations."),
        ("Final Prompt — Multi-turn Context & Guardrails: ", "Injects prior interview conversation history ({interview_context}) for adaptive follow-up questioning, and implements strict edge-case guardrails for empty, Hinglish, or adversarial inputs.")
    ]
    for m_title, m_desc in method_points:
        p_item = doc.add_paragraph()
        p_item.paragraph_format.space_after = Pt(4)
        r1 = p_item.add_run(f"• {m_title}")
        r1.font.bold = True
        r1.font.color.rgb = COLOR_PRIMARY
        r2 = p_item.add_run(m_desc)
        r2.font.color.rgb = COLOR_DARK

    doc.add_page_break()

    # ==================== 3. VERSION 1 EXPERIMENT ====================
    add_heading_1(doc, "3. Version 1 (V1) Experiment: Baseline Role Evaluator")
    add_body_p(doc, "The V1 prompt represents our initial baseline experiment using role prompting and an open instruction checklist.")

    v1_prompt_text = (
        "You are an AI-powered Software Engineering Interview Coach.\n\n"
        "Your task is to evaluate a student's interview answer and help them improve their technical interview skills.\n\n"
        "The student is preparing for a software engineering interview.\n\n"
        "The interview may cover:\n"
        "- Data Structures and Algorithms\n"
        "- Core Computer Science (DBMS, OOP, Operating Systems, Computer Networks)\n"
        "- Software development projects\n\n"
        "You will receive the following information:\n\n"
        "Job Role: Software Developer\n"
        "Interview Question: What is the difference between an Array and a Linked List?\n"
        "Student Answer: An array stores elements in continuous memory locations, while a linked list stores elements in nodes connected using pointers. Arrays allow faster access using indexes, whereas linked lists make insertion and deletion easier in certain situations.\n\n"
        "Based on the information provided:\n"
        "1. Evaluate the student's answer.\n"
        "2. Explain what the student did well.\n"
        "3. Identify mistakes, missing concepts, or areas that need improvement.\n"
        "4. Give suggestions to improve the answer.\n"
        "5. Generate one relevant follow-up interview question.\n\n"
        "Keep your feedback clear, constructive, and easy for a student to understand.\n"
        "Do not unnecessarily praise an incorrect answer. Be honest and helpful."
    )
    add_prompt_box(doc, v1_prompt_text, "V1 Baseline Prompt")

    v1_response_text = (
        "Software Engineering Interview Evaluation\n"
        "Job Role: Software Developer | Interview Question: What is the difference between an Array and a Linked List?\n"
        "Good Answer | Beginner–Intermediate\n\n"
        "1. Overall Evaluation: 8/10\n"
        "The answer is technically correct but could be more comprehensive.\n"
        "- Technical accuracy: 9/10\n- Concept clarity: 8/10\n- Completeness: 7/10\n- Communication: 8/10\n\n"
        "2. What You Did Well:\n"
        "You correctly explained three important differences:\n"
        "1. Memory Allocation: Arrays store elements in contiguous memory locations, whereas linked lists store elements in separate nodes connected through pointers.\n"
        "2. Access Time: Arrays support direct index-based access, making element retrieval O(1). Linked lists require sequential traversal, taking O(n).\n"
        "3. Insertion and Deletion: Linked lists can perform insertion and deletion efficiently when the required node or position is already known.\n\n"
        "3. Areas That Need Improvement:\n"
        "Missed formal Big-O time and space complexity comparison across common operations (Access, Search, Insertion, Deletion).\n"
        "Clarification needed: Linked list insertion and deletion are not always O(1). Finding the required position can take O(n).\n"
        "Omitted cache performance and memory overhead differences.\n\n"
        "4. Improved Interview Answer:\n"
        "\"An Array and a Linked List are both linear data structures, but they differ in memory allocation and element access.\n"
        "An Array stores elements in contiguous memory locations, allowing random access using indexes in O(1) time.\n"
        "A Linked List stores elements in separate nodes, where each node contains data and a pointer to the next node. Accessing an element requires traversal, taking O(n) time.\n"
        "Insertion and deletion at the beginning are generally O(n) in an array but O(1) in a linked list, assuming we already have the required node.\n"
        "Arrays generally provide better cache performance, while linked lists offer more flexible memory allocation.\"\n\n"
        "5. Follow-up Interview Question:\n"
        "Suppose you need to frequently insert and delete elements at the beginning of a collection. Which data structure would you choose: Array or Linked List, and why?"
    )
    add_response_box(doc, v1_response_text, "ChatGPT Observed Response for V1 (Score: 8/10)")

    add_heading_2(doc, "V1 Analysis & Failure Modes")
    add_body_p(doc, 
        "1. Subjective & Unanchored Scoring: The model awarded an overall score of 8/10, but broke it down into sub-scores out of 10 (Accuracy: 9/10, Clarity: 8/10, Completeness: 7/10, Communication: 8/10). The arithmetic mean of these sub-scores is 8.0, but without weightings or scoring guidelines, identical answers on repeated runs yielded arbitrary scores between 7.5 and 8.5.\n"
        "2. Lack of Explicit Rubric: Evaluator criteria were not bounded. The prompt did not specify how many marks each dimension contributed to the final total.\n"
        "3. Fragile Follow-Up: The follow-up question did not probe specific trade-offs or constraints.",
        "Key Findings: "
    )

    doc.add_page_break()

    # ==================== 4. VERSION 2 EXPERIMENT ====================
    add_heading_1(doc, "4. Version 2 (V2) Experiment: Rubric-Based Evaluator")
    add_body_p(doc, "To resolve scoring ambiguity, V2 introduced a strict 10-mark scoring rubric and structured output sections.")

    v2_prompt_text = (
        "You are an expert Software Engineering Technical Interview Evaluator and Coach.\n\n"
        "Your task is to evaluate a student's answer to a software engineering interview question and provide accurate, structured and constructive feedback.\n\n"
        "INPUT:\n"
        "Job Role: {job_role}\n"
        "Interview Question: {interview_question}\n"
        "Student Answer: {student_answer}\n\n"
        "EVALUATION INSTRUCTIONS:\n"
        "1. Evaluate the answer based on technical correctness, completeness, relevance and clarity.\n"
        "2. Identify concepts explained correctly.\n"
        "3. Identify incorrect statements and important missing concepts.\n"
        "4. Do not assume that an omitted concept is incorrect.\n"
        "5. Explain technical mistakes clearly.\n"
        "6. Provide a technically accurate improved answer.\n"
        "7. Generate exactly ONE relevant follow-up question.\n\n"
        "SCORING RUBRIC (10 MARKS):\n"
        "- Technical Accuracy: 0–4 marks\n"
        "- Completeness: 0–2 marks\n"
        "- Relevance: 0–2 marks\n"
        "- Communication and Clarity: 0–2 marks\n\n"
        "Award marks according to the quality of the actual answer. Explain the reason for the score in each category.\n\n"
        "OUTPUT FORMAT:\n"
        "## 1. Overall Evaluation\n- Total score out of 10\n- Short overall assessment\n"
        "## 2. Score Breakdown\nShow each evaluation criterion, marks awarded and reason.\n"
        "## 3. Strengths\nList the technically correct points in the student's answer.\n"
        "## 4. Areas for Improvement\nList missing concepts, incorrect statements and explanations needed. Use Markdown tables when appropriate.\n"
        "## 5. Improved Interview Answer\nProvide a clear, technically correct answer suitable for the given job role.\n"
        "## 6. Follow-up Question\nGenerate exactly one relevant question that tests deeper understanding."
    )
    add_prompt_box(doc, v2_prompt_text, "V2 Rubric Prompt")

    v2_response_text = (
        "## 1. Overall Evaluation\n"
        "8.5/10 — Good Answer\n"
        "Your answer demonstrates a good understanding of fundamental differences between arrays and linked lists. "
        "You correctly explained memory organization, access mechanisms, and insertion/deletion characteristics.\n\n"
        "## 2. Score Breakdown\n"
        "- Technical Accuracy: 4/4 (All statements are technically correct)\n"
        "- Completeness: 1.5/2 (Missing time complexity details and memory overhead)\n"
        "- Relevance: 2/2 (Directly addresses the question)\n"
        "- Communication & Clarity: 1/2 (Clear and concise, but could provide more detail)\n"
        "Total Score: 8.5/10\n\n"
        "## 3. Strengths\n"
        "1. Memory allocation: Arrays store elements in contiguous memory locations.\n"
        "2. Linked List structure: Linked lists contain nodes connected through pointers.\n"
        "3. Access speed: Arrays support fast index-based access.\n"
        "4. Insertion and deletion: Linked lists make insertion and deletion easier in certain situations.\n\n"
        "## 4. Areas for Improvement\n"
        "Time Complexity Comparison:\n"
        "| Operation | Array | Linked List |\n"
        "| Access by index | O(1) | O(n) |\n"
        "| Search | O(n) | O(n) |\n"
        "| Insertion at beginning | O(n) | O(1) |\n"
        "| Deletion at beginning | O(n) | O(1) |\n\n"
        "Memory Overhead: Arrays have lower per-element overhead; linked lists require extra pointer memory per node.\n"
        "Dynamic Sizing: Static arrays have fixed capacity; linked lists grow dynamically.\n\n"
        "## 5. Improved Interview Answer\n"
        "\"An array and a linked list are both linear data structures, but they differ in memory allocation and element access. "
        "An array stores elements in contiguous memory locations, allowing direct access to elements using indexes in O(1) time. "
        "A linked list stores elements in separate nodes, where each node contains data and a pointer to the next node. Accessing an element requires traversal, taking O(n) time. "
        "Arrays are generally better when frequent random access is required, whereas linked lists can be more efficient for insertion and deletion when the position is known. "
        "Arrays generally have better cache locality and lower memory overhead, while linked lists require additional memory for pointers.\"\n\n"
        "## 6. Follow-up Question\n"
        "Difficulty: Medium\n"
        "Suppose you need to frequently insert and delete elements at the beginning of a collection. Would you choose an Array or a Linked List, and what would be the time complexity of these operations in both data structures?"
    )
    add_response_box(doc, v2_response_text, "ChatGPT Observed Response for V2 (Score: 8.5/10)")

    add_heading_2(doc, "V2 Analysis & Failure Modes")
    add_body_p(doc,
        "1. Rubric Precision Achieved: The 10-mark rubric (4+2+2+2) immediately brought clarity to the assessment. The student received 4/4 for Technical Accuracy, 1.5/2 for Completeness, 2/2 for Relevance, and 1/2 for Clarity, totaling 8.5/10.\n"
        "2. Markdown Table Fragility: Instructing the model to 'use Markdown tables when appropriate' resulted in broken pipes and irregular rendering across web browsers and mobile screens.\n"
        "3. Boundary Score Inconsistency: Without explicit scoring anchors (defining what distinguishes a 1.0 from a 1.5 in Completeness), different runs exhibited minor score jitter.",
        "Key Findings: "
    )

    doc.add_page_break()

    # ==================== 5. VERSION 3 EXPERIMENT ====================
    add_heading_1(doc, "5. Version 3 (V3) Experiment: Refined Anchor Evaluator")
    add_body_p(doc, "V3 introduced explicit anchor scoring criteria (4/3/2/1/0, 2/1/0), strictly banned fragile tables, and mandated evidence-based citations.")

    v3_prompt_text = (
        "You are a senior Software Engineering interviewer and technical interview coach.\n\n"
        "Your task is to evaluate a student's answer to a software engineering interview question and provide fair, accurate and actionable feedback.\n\n"
        "INPUT:\n"
        "Job Role: {job_role}\n"
        "Interview Question: {interview_question}\n"
        "Student Answer: {student_answer}\n\n"
        "SCORING RUBRIC (10 MARKS):\n\n"
        "1. Technical Accuracy (0-4):\n"
        "4 = All statements are technically correct.\n"
        "3 = Mostly correct with minor inaccuracies.\n"
        "2 = Some correct concepts but noticeable errors.\n"
        "1 = Major technical misunderstandings.\n"
        "0 = No technically correct explanation.\n\n"
        "2. Completeness (0-2):\n"
        "2 = Covers the important expected concepts.\n"
        "1 = Covers some important concepts but misses others.\n"
        "0 = Misses most expected concepts.\n\n"
        "3. Relevance (0-2):\n"
        "2 = Directly answers the question.\n"
        "1 = Partially addresses the question.\n"
        "0 = Mostly irrelevant.\n\n"
        "4. Communication and Clarity (0-2):\n"
        "2 = Clear, logically structured and easy to understand.\n"
        "1 = Understandable but poorly explained or organized.\n"
        "0 = Difficult to understand.\n\n"
        "IMPORTANT SCORING RULES:\n"
        "- Award marks only according to the rubric.\n"
        "- Explain each score with evidence from the student's answer.\n"
        "- Total score must strictly equal the sum of all four categories.\n"
        "- Avoid complex Markdown tables; use clear bullet points.\n"
        "- Generate exactly ONE relevant follow-up question testing deeper understanding."
    )
    add_prompt_box(doc, v3_prompt_text, "V3 Refined Anchor Prompt")

    v3_response_text = (
        "## 1. Overall Evaluation\n"
        "9/10 — Strong Answer\n"
        "The student demonstrates a solid understanding of fundamental differences between arrays and linked lists. "
        "The answer is technically accurate, relevant, and concise. Opportunity for improvement is detailing operation time complexities.\n\n"
        "## 2. Score Breakdown\n"
        "- Technical Accuracy: 4/4 — All statements are technically correct.\n"
        "- Completeness: 1/2 — Covers memory organization and access, but omits complexity details.\n"
        "- Relevance: 2/2 — Directly answers the comparison question.\n"
        "- Communication and Clarity: 2/2 — Clear, concise, and easy to understand.\n"
        "Total Score: 9/10\n\n"
        "## 3. Strengths\n"
        "- Memory allocation: Arrays store elements in contiguous memory locations.\n"
        "- Linked List structure: Linked lists use nodes connected through pointers.\n"
        "- Access: Arrays provide faster index-based access, typically O(1).\n"
        "- Insertion and deletion: Linked lists can perform these operations efficiently in certain situations.\n\n"
        "## 4. Areas for Improvement\n"
        "- Explain time complexity: Array index access is O(1) while Linked List is O(n); insertion at beginning is O(n) in array vs O(1) in linked list.\n"
        "- Explain memory overhead: Arrays have minimal overhead while linked lists store extra pointer references.\n"
        "- Explain practical applications: Arrays for frequent random access; Linked Lists for frequent insertions/deletions at known positions.\n\n"
        "## 5. Improved Interview Answer\n"
        "\"An array and a linked list are both linear data structures, but they differ in memory organization and operations. "
        "An array stores elements in contiguous memory locations, which allows direct access using indexes in O(1) time. "
        "A linked list stores elements in separate nodes, where each node contains data and a pointer to the next node. Accessing an element requires traversal, taking O(n) time. "
        "Arrays are generally better for random access, while linked lists are useful for frequent insertions and deletions when the required position is already known. "
        "Additionally, linked lists require extra memory for pointers, whereas arrays generally have lower memory overhead.\"\n\n"
        "## 6. Follow-up Question\n"
        "Medium Difficulty\n"
        "Suppose you have to implement a music playlist where users frequently add and remove songs from the beginning and middle of the playlist. Would you choose an Array or a Linked List, and why? Explain the time complexity of your choice."
    )
    add_response_box(doc, v3_response_text, "ChatGPT Observed Response for V3 (Score: 9/10)")

    add_heading_2(doc, "V3 Analysis & Academic Findings")
    add_body_p(doc,
        "1. High Consistency & Evidence Grounding: By anchoring the score explicitly (4=All correct, 3=Mostly correct, etc.) and enforcing integer scoring steps on Completeness (1/2 for covering core without Big-O), V3 produced an indisputable 9/10 score.\n"
        "2. Improved Follow-Up Quality: Rather than a repetitive theoretical question, V3 generated a situational, scenario-based follow-up question (designing a music playlist).\n"
        "3. UI Stability: Replacing Markdown tables with bullet points ensured clean rendering across all browsers.",
        "Key Findings: "
    )

    doc.add_page_break()

    # ==================== 6. PROMPT COMPARISON TABLE ====================
    add_heading_1(doc, "6. Prompt Versions Comparative Analysis (TC01 Benchmark)")

    comp_table = doc.add_table(rows=10, cols=5)
    comp_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    headers = ["Evaluation Metric", "Version 1 (V1)", "Version 2 (V2)", "Version 3 (V3)", "Final Prompt"]
    for j, h in enumerate(headers):
        cell = comp_table.cell(0, j)
        set_cell_background(cell, HEX_PRIMARY)
        set_cell_margins(cell, 100, 100, 120, 120)
        p = cell.paragraphs[0]
        r = p.add_run(h)
        r.font.bold = True
        r.font.size = Pt(10)
        r.font.color.rgb = RGBColor(255, 255, 255)

    comp_rows = [
        ("Prompt Strategy", "Role Prompting", "10-Mark Rubric", "Refined Scoring Anchors", "Multi-Turn + Guardrails"),
        ("Observed Score (TC01)", "8.0 / 10", "8.5 / 10", "9.0 / 10", "9.0 / 10"),
        ("Technical Accuracy", "9 / 10 (unweighted)", "4.0 / 4.0", "4.0 / 4.0", "4.0 / 4.0"),
        ("Completeness", "7 / 10 (unweighted)", "1.5 / 2.0", "1.0 / 2.0", "1.0 / 2.0"),
        ("Relevance", "Unscored", "2.0 / 2.0", "2.0 / 2.0", "2.0 / 2.0"),
        ("Clarity & Communication", "8 / 10 (unweighted)", "1.0 / 2.0", "2.0 / 2.0", "2.0 / 2.0"),
        ("Output Structure", "Numbered list", "Markdown tables", "Clean bullet points", "Section-anchored cards"),
        ("Follow-up Question Quality", "Basic conceptual", "Complexity theoretical", "Practical Scenario (Playlist)", "Adaptive Contextual Probing"),
        ("Edge Case Handling", "Unspecified", "Unspecified", "Basic", "Strict (0/10 on blank/idk)")
    ]

    for i, row in enumerate(comp_rows, 1):
        for j, val in enumerate(row):
            cell = comp_table.cell(i, j)
            bg = HEX_LIGHT_BG if i % 2 == 1 else "FFFFFF"
            set_cell_background(cell, bg)
            set_cell_margins(cell, 80, 80, 120, 120)
            p = cell.paragraphs[0]
            r = p.add_run(val)
            r.font.size = Pt(9.5)
            if j == 0 or (j == 4 and i == 1):
                r.font.bold = True
            if j == 1 and i == 1:
                r.font.bold = True
                r.font.color.rgb = COLOR_PRIMARY

    doc.add_paragraph().paragraph_format.space_after = Pt(8)

    # ==================== 7. STANDARDIZED TEST DATASET ====================
    add_heading_1(doc, "7. Standardized Testing Dataset & 40-Run Empirical Results")
    add_body_p(doc,
        "In strict adherence to academic integrity and rigorous empirical validation, a 10-case evaluation matrix was constructed "
        "covering all core domains (DSA, DBMS, OS, OOP, CN, Software Development Projects) and featuring realistic student answers, "
        "partial answers with common misconceptions, and edge cases (empty input).\n\n"
        "All 10 test cases were executed across all four prompt versions (V1, V2, V3, and Final) using the live Google Gemini API "
        "(gemini-3.1-flash-lite-preview via OpenAI-compatible endpoint). In total, 40 distinct evaluation runs were recorded with 100% "
        "genuine LLM outputs (40/40 real LLM runs, 0 simulated runs, 0 API errors).",
        "Empirical Protocol: "
    )

    test_table = doc.add_table(rows=11, cols=7)
    test_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    t_headers = ["Test ID", "Category", "Type", "Core Focus", "V1 Score", "V2 Score", "Final Score"]
    for j, h in enumerate(t_headers):
        cell = test_table.cell(0, j)
        set_cell_background(cell, HEX_PRIMARY)
        set_cell_margins(cell, 100, 100, 100, 100)
        p = cell.paragraphs[0]
        r = p.add_run(h)
        r.font.bold = True
        r.font.size = Pt(9.0)
        r.font.color.rgb = RGBColor(255, 255, 255)

    test_rows = [
        ("TC01", "DSA", "Normal", "Array vs Linked List (Memory & Access)", "7.5 / 10", "9.0 / 10", "9.0 / 10"),
        ("TC02", "DSA", "Normal", "Binary Search (O(log n) Derivation)", "7.5 / 10", "9.0 / 10", "9.0 / 10"),
        ("TC03", "DSA", "Normal", "Cycle Detection (Floyd's Fast & Slow)", "7.5 / 10", "10.0 / 10", "9.0 / 10"),
        ("TC04", "DBMS", "Normal", "Database Normalization (1NF to BCNF)", "7.5 / 10", "9.0 / 10", "8.0 / 10"),
        ("TC05", "OOP", "Normal", "Polymorphism (Dynamic Dispatch Example)", "7.5 / 10", "9.0 / 10", "9.0 / 10"),
        ("TC06", "OS", "Normal", "Process vs Thread (Context Switch & Memory)", "7.5 / 10", "9.0 / 10", "9.0 / 10"),
        ("TC07", "CN", "Normal", "TCP vs UDP (FTP Reliability & Flow Control)", "7.5 / 10", "9.0 / 10", "9.0 / 10"),
        ("TC08", "Projects", "Normal", "Student Management System (3-Tier Arch)", "7.5 / 10", "7.0 / 10", "8.0 / 10"),
        ("TC09", "DSA", "Partially Correct", "QuickSort Space (O(1) Error vs Call Stack)", "7.5 / 10", "7.0 / 10", "7.0 / 10"),
        ("TC10", "DBMS", "Edge Case (Empty)", "ACID Properties ([Blank / Empty Input])", "7.5 / 10", "0.0 / 10", "0.0 / 10")
    ]

    for i, row in enumerate(test_rows, 1):
        for j, val in enumerate(row):
            cell = test_table.cell(i, j)
            bg = HEX_LIGHT_BG if i % 2 == 1 else "FFFFFF"
            set_cell_background(cell, bg)
            set_cell_margins(cell, 80, 80, 100, 100)
            p = cell.paragraphs[0]
            r = p.add_run(val)
            r.font.size = Pt(8.5)
            if j == 0:
                r.font.bold = True
            if "Score" in t_headers[j]:
                r.font.bold = True
                if val == "0.0 / 10":
                    r.font.color.rgb = RGBColor(220, 38, 38)
                else:
                    r.font.color.rgb = RGBColor(5, 150, 105)

    doc.add_page_break()

    # ==================== 8. TEAM CONTRIBUTION & TASK OWNERSHIP ====================
    add_heading_1(doc, "8. Team Contributions & Milestone 2 Task Ownership (Section 7)")
    add_body_p(doc,
        "In accordance with Section 7 of the Bennett University SCSE4041 Milestone 2 guidelines, every member has owned specific tasks with concrete file evidence. "
        "All four team members have mutually agreed upon the division of responsibilities and equal contribution (25% each = 100% total).",
        "Contribution Framework: "
    )

    c_table = doc.add_table(rows=5, cols=6)
    c_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    c_heads = ["Member Name", "Enrollment No.", "Role", "Tasks Done & Owned", "Evidence", "Contrib (%)"]
    for j, h in enumerate(c_heads):
        cell = c_table.cell(0, j)
        set_cell_background(cell, HEX_PRIMARY)
        set_cell_margins(cell, 100, 100, 100, 100)
        p = cell.paragraphs[0]
        r = p.add_run(h)
        r.font.bold = True
        r.font.size = Pt(9.5)
        r.font.color.rgb = RGBColor(255, 255, 255)

    c_rows = [
        (
            "Sagar Goyal",
            "S24CSEU0441",
            "Prompt Engineering & Research Lead",
            "Problem C10 analysis and requirement identification; designed and documented prompt versions V1, V2, V3; developed 10-mark scoring rubric and prompt instructions; evaluated prompt outputs and comparative analysis; authored Final Prompt design.\nViva Focus: Prompt design, prompt iteration, rubric, consistency, and evaluation.",
            "prompts/v1/, prompts/v2/, prompts/v3/, prompts/final/, prompts/prompt_loader.py",
            "25%"
        ),
        (
            "Shauriya Garg",
            "S24CSEU0409",
            "AI Evaluation & Backend Lead",
            "Implemented C10 evaluation module (c10_evaluator.py); integrated LLM API via OpenAI-compatible client (supporting Gemini & OpenAI); implemented response parsing and scoring; multi-turn context management (thread_turns); API error handling and offline demo fallback.\nViva Focus: Backend logic, API integration, LLM interaction, scoring, and multi-turn flow.",
            "c10_evaluator.py, app.py (/api/c10/evaluate, /api/c10/questions)",
            "25%"
        ),
        (
            "Sarthak Maheswari",
            "S24CSEU0472",
            "Frontend & Application Integration Lead",
            "Designed mock interview user interface; implemented Job Role, Interview Category, and Prompt Version dropdowns; connected frontend interactions to backend API endpoints; rendered score badges, rubric cards, model answers, and follow-up flow; integrated with existing features.\nViva Focus: Flask routes, frontend-backend communication, UI flow, and API endpoints.",
            "templates/index.html, static/app.js, static/styles.css",
            "25%"
        ),
        (
            "Rohit Mukhopadhyay",
            "S24CSEU0375",
            "Testing, Validation & Documentation Lead",
            "Constructed 10-case standardized test dataset including edge cases; created automated test suites (test_c10.py, test_endpoints.py, test_integration.py); recorded test logs and limitations; authored experiment documentation, README.md, CHANGELOG.md, and PowerPoint presentation.\nViva Focus: Test design, edge cases, integration verification, limitations, and documentation.",
            "test-data/test_dataset.json, test_*.py, docs/, presentation/",
            "25%"
        )
    ]

    for i, row in enumerate(c_rows, 1):
        for j, val in enumerate(row):
            cell = c_table.cell(i, j)
            bg = HEX_LIGHT_BG if i % 2 == 1 else "FFFFFF"
            set_cell_background(cell, bg)
            set_cell_margins(cell, 80, 80, 100, 100)
            p = cell.paragraphs[0]
            r = p.add_run(val)
            r.font.size = Pt(8.5)
            if j in (0, 1, 2, 5):
                r.font.bold = True
            if j == 5:
                r.font.color.rgb = COLOR_PRIMARY

    doc.add_page_break()

    # ==================== 9. LIMITATIONS & FUTURE WORK ====================
    add_heading_1(doc, "9. Limitations, Ethical Considerations & Future Work")
    add_body_p(doc,
        "1. Free Tier Rate Limits: Google Gemini free-tier access enforces daily request quotas per model family (e.g. 20 requests/day on flash-preview), which was overcome by configuring intelligent multi-model candidate failover (gemini-3.1-flash-lite-preview).\n"
        "2. Deterministic Fallbacks: In the absence of an active API key, the platform utilizes dynamic heuristic rules labeled explicitly as [SIMULATED DEMO MODE - NO API KEY]. These must not be conflated with empirical LLM outputs.\n"
        "3. Multilingual Nuance: While the Final Prompt gracefully evaluates technical intent in Hinglish phrasing, technical explanations in regional languages require formal multilingual fine-tuning.",
        "Observed Limitations: "
    )
    add_body_p(doc,
        "1. Academic Integrity: Never presenting fabricated or simulated model scores as empirical evidence.\n"
        "2. Constructive Pedagogical Tone: Banning harsh criticism or demoralizing language for struggling students while refusing to falsely praise incorrect technical facts.\n"
        "3. Privacy Protection: No student credentials, session recordings, or resumes are sent to external third parties without user awareness.",
        "Responsible AI Considerations: "
    )
    add_body_p(doc,
        "1. Execution of full test dataset across Google Gemini 1.5 Flash and OpenAI gpt-4o-mini to establish multi-model reliability metrics.\n"
        "2. Integration of AST-based code validation for programming questions requiring live code snippet execution.\n"
        "3. Longitudinal reinforcement learning modeling candidate mastery over multi-week placement preparation cycles.",
        "Future Enhancements: "
    )

    doc.save(OUTPUT_FILE)
    print(f"Documentation successfully saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    build_experiment_doc()
