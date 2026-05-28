import fitz  # PyMuPDF


class PDFLoader:
    """
    Handles PDF document loading and text extraction.
    """

    def __init__(self):
        pass

    def extract_text(self, uploaded_file):
        """
        Extracts text from uploaded PDF file.

        Args:
            uploaded_file: Streamlit uploaded PDF file

        Returns:
            str: Extracted text
        """

        text = ""

        try:
            # Open PDF from uploaded file stream
            pdf_document = fitz.open(
                stream=uploaded_file.read(),
                filetype="pdf"
            )

            # Read all pages
            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]
                text += page.get_text()

            pdf_document.close()

            return text

        except Exception as e:
            print(f"Error reading PDF: {e}")
            return ""