#Extract plain text from PDF and TXT files

from io import BytesIO
import PyPDF2

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text from PDF bytes and return the full text."""
    reader = PyPDF2.PdfReader(BytesIO(file_bytes))
    pages_text = []
    for p in reader.pages:
        try:
            pages_text.append(p.extract_text() or "")  #Extract text if None, use empty string
        except Exception:
            pages_text.append("")                      #Add empty string if extraction fails
    full_text = "\n\n".join(pages_text)                #Join all pages with double newline to separate pages
    return full_text

def extract_text_from_txt(file_bytes: bytes) -> str:
    """Extract text from TXT bytes and return the full text."""
    return file_bytes.decode("utf-8", errors="ignore")

def extract_text_from_file(file_bytes: bytes, filename: str) -> str:
    """Detect file type by filename and extract text accordingly."""
    if filename.lower().endswith(".pdf"):
        return extract_text_from_pdf(file_bytes)
    else:
        return extract_text_from_txt(file_bytes)
