import sys
import os

try:
    import fitz  # PyMuPDF
    doc = fitz.open(sys.argv[1])
    text = ""
    for i in range(min(3, len(doc))):
        text += doc[i].get_text()
    print(text[:3000])
except ImportError:
    try:
        import PyPDF2
        reader = PyPDF2.PdfReader(sys.argv[1])
        text = "".join(page.extract_text() for page in reader.pages[:3])
        print(text[:3000])
    except ImportError:
        print("Please install PyMuPDF or PyPDF2: pip install PyMuPDF PyPDF2")
