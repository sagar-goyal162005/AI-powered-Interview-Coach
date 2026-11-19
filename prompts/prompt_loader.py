"""
Prompt Loader Module for C10 AI-Powered Software Engineering Mock Interview Coach.
Loads versioned prompts (V1, V2, V3, and Final) from the prompts directory.
"""

from pathlib import Path
from typing import Dict, Optional

PROMPTS_DIR = Path(__file__).resolve().parent

PROMPT_FILES = {
    "v1": PROMPTS_DIR / "v1" / "evaluator_v1.txt",
    "v2": PROMPTS_DIR / "v2" / "evaluator_v2.txt",
    "v3": PROMPTS_DIR / "v3" / "evaluator_v3.txt",
    "final": PROMPTS_DIR / "final" / "evaluator_final.txt",
}


def get_prompt_template(version: str = "final") -> str:
    """Return the raw prompt template for a specific version ('v1', 'v2', 'v3', 'final')."""
    ver = version.lower().strip()
    path = PROMPT_FILES.get(ver, PROMPT_FILES["final"])
    if not path.exists():
        raise FileNotFoundError(f"Prompt template file not found at {path}")
    return path.read_text(encoding="utf-8")


def format_evaluation_prompt(
    version: str,
    job_role: str,
    interview_question: str,
    student_answer: str,
    interview_category: Optional[str] = None,
    interview_context: Optional[str] = None,
) -> str:
    """Format the specified prompt version with provided arguments."""
    template = get_prompt_template(version)
    context_str = interview_context if interview_context else "None (First turn of interview)"
    cat_str = interview_category if interview_category else "General Software Engineering"

    # Handle variables safely depending on version template
    replacements = {
        "{job_role}": job_role or "Software Developer",
        "{interview_question}": interview_question,
        "{student_answer}": student_answer if student_answer.strip() else "[No answer provided / blank]",
        "{interview_category}": cat_str,
        "{interview_context}": context_str,
    }

    formatted = template
    for key, val in replacements.items():
        formatted = formatted.replace(key, str(val))

    return formatted
