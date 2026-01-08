import pdfplumber
import docx
import re

def extract_text(file, file_type):
    text = ""

    if file_type == "pdf":
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                if page.extract_text():
                    text += page.extract_text() + " "

    elif file_type == "docx":
        doc = docx.Document(file)
        for para in doc.paragraphs:
            text += para.text + " "

    elif file_type == "txt":
        text = file.read().decode("utf-8", errors="ignore")

    # Clean extra whitespace
    text = re.sub(r"\s+", " ", text)
    return text.strip()
