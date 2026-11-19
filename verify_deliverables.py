import sys
from pathlib import Path
from docx import Document
from pptx import Presentation

def verify_docx(path):
    p = Path(path)
    assert p.exists(), f"File does not exist: {path}"
    doc = Document(str(p))
    paras = [para.text for para in doc.paragraphs if para.text.strip()]
    tables = doc.tables
    total_words = sum(len(text.split()) for text in paras)
    for t in tables:
        for row in t.rows:
            for cell in row.cells:
                total_words += sum(len(para.text.split()) for para in cell.paragraphs)
    
    headings = [para.text for para in doc.paragraphs if para.text.startswith("Chapter ") or para.text.startswith("1.") or para.text.startswith("2.") or para.text.startswith("3.") or para.text.startswith("4.") or para.text.startswith("5.") or para.text.startswith("6.") or para.text.startswith("7.") or para.text.startswith("8.") or para.text.startswith("9.")]
    
    return {
        "file": p.name,
        "size_bytes": p.stat().st_size,
        "paragraphs_count": len(paras),
        "tables_count": len(tables),
        "total_words": total_words,
        "headings": headings[:15],
    }

def verify_pptx(path):
    p = Path(path)
    assert p.exists(), f"File does not exist: {path}"
    prs = Presentation(str(p))
    slide_count = len(prs.slides)
    slide_titles = []
    total_words = 0
    for idx, slide in enumerate(prs.slides, 1):
        texts = []
        for shape in slide.shapes:
            if shape.has_text_frame:
                for para in shape.text_frame.paragraphs:
                    if para.text.strip():
                        texts.append(para.text.strip())
                        total_words += len(para.text.split())
        title = texts[0] if texts else f"Slide {idx} (No Title)"
        slide_titles.append((idx, title))
    
    return {
        "file": p.name,
        "size_bytes": p.stat().st_size,
        "slide_count": slide_count,
        "total_words": total_words,
        "slide_titles": slide_titles
    }

def main():
    print("=== AUDIT: DOCUMENT DELIVERABLES ===")
    f1 = "docs/Prompt_Engineering_Experiment_Documentation.docx"
    r1 = verify_docx(f1)
    print(f"[OK] {r1['file']}: {r1['size_bytes']:,} bytes, {r1['paragraphs_count']} paragraphs, {r1['tables_count']} tables, ~{r1['total_words']:,} words.")
    
    f2 = "docs/AI_Software_Interview_Coach_Final_Report.docx"
    r2 = verify_docx(f2)
    print(f"[OK] {r2['file']}: {r2['size_bytes']:,} bytes, {r2['paragraphs_count']} paragraphs, {r2['tables_count']} tables, ~{r2['total_words']:,} words.")
    
    f3 = "presentation/AI_Software_Interview_Coach_Final_Presentation.pptx"
    r3 = verify_pptx(f3)
    print(f"[OK] {r3['file']}: {r3['size_bytes']:,} bytes, {r3['slide_count']} slides, ~{r3['total_words']:,} words.")
    print("\nSlide Breakdown:")
    for num, title in r3['slide_titles']:
        print(f"  Slide {num:02d}: {title[:75]}")

if __name__ == "__main__":
    main()
