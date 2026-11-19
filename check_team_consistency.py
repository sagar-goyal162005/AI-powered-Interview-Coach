from pathlib import Path
from docx import Document
from pptx import Presentation

MEMBERS = [
    ("Sagar Goyal", "S24CSEU0441"),
    ("Shauriya Garg", "S24CSEU0409"),
    ("Sarthak Maheswari", "S24CSEU0472"),
    ("Rohit Mukhopadhyay", "S24CSEU0375")
]

def check_text(name, text):
    print(f"\nChecking: {name}")
    for person, roll in MEMBERS:
        has_person = person.lower() in text.lower()
        has_roll = roll.lower() in text.lower()
        status = "OK" if (has_person and has_roll) else ("MISSING ROLL" if has_person else "NOT FOUND")
        print(f"  {person} ({roll}): {status}")

def main():
    # 1. README.md
    check_text("README.md", Path("README.md").read_text(encoding="utf-8"))
    
    # 2. CHANGELOG.md
    check_text("CHANGELOG.md", Path("CHANGELOG.md").read_text(encoding="utf-8"))
    
    # 3. experiment_log.md
    check_text("experiment_log.md", Path("experiment-results/experiment_log.md").read_text(encoding="utf-8"))
    
    # 4. Prompt_Engineering_Experiment_Documentation.docx
    doc1 = Document("docs/Prompt_Engineering_Experiment_Documentation.docx")
    text1 = " ".join([p.text for p in doc1.paragraphs] + [c.text for t in doc1.tables for row in t.rows for c in row.cells])
    check_text("Prompt_Engineering_Experiment_Documentation.docx", text1)

    # 5. AI_Software_Interview_Coach_Final_Report.docx
    doc2 = Document("docs/AI_Software_Interview_Coach_Final_Report.docx")
    text2 = " ".join([p.text for p in doc2.paragraphs] + [c.text for t in doc2.tables for row in t.rows for c in row.cells])
    check_text("AI_Software_Interview_Coach_Final_Report.docx", text2)

    # 6. AI_Software_Interview_Coach_Final_Presentation.pptx
    prs = Presentation("presentation/AI_Software_Interview_Coach_Final_Presentation.pptx")
    text3 = " ".join([p.text for s in prs.slides for sh in s.shapes if sh.has_text_frame for p in sh.text_frame.paragraphs])
    check_text("AI_Software_Interview_Coach_Final_Presentation.pptx", text3)

if __name__ == "__main__":
    main()
