import PyPDF2
import re


def extract_text_from_pdf(file):
    """
    Extract text from a PDF file.

    Args:
        file (FileStorage): A file-like object representing the PDF file.

    Returns:
        str: The extracted text from the PDF.
    """
    text = ''
    # Use file.stream to access the file-like object
    reader = PyPDF2.PdfReader(file.stream)
    for page_num in range(len(reader.pages)):
        page = reader.pages[page_num]
        text += page.extract_text() or ''  # Handle cases where extract_text might return None
    
    text=clean_text(text)
    return text

def clean_text(text):
    """
    Clean the text by removing invalid Unicode characters.

    Args:
        text (str): The text to be cleaned.

    Returns:
        str: The cleaned text.
    """
    # Remove null bytes and other problematic characters
    text = text.replace('\u0000', '')
    
    # Remove any other non-printable characters if necessary
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Keeps only ASCII characters
    return text



# def extract_text_from_pdf(file):
#     with pdfplumber.open(file) as pdf:
#         text = ''
#         for page in pdf.pages:
#             text += page.extract_text()
#     return text