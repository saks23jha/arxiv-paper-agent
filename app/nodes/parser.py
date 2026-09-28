import fitz


def parse_pdf(pdf_path: str) -> str:
    """
    Extract text from a PDF using PyMuPDF.

    Args:
        pdf_path: Local path to the PDF.

    Returns:
        Extracted text from the PDF.
    """

    try:
        document = fitz.open(pdf_path)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to open PDF: {exc}"
        ) from exc

    pages = []

    try:
        for page in document:
            text = page.get_text("text")

            if text.strip():
                pages.append(text.strip())
    finally:
        document.close()

    if not pages:
        raise RuntimeError(
            "No text could be extracted from the PDF. "
            "The paper may be scanned/image-based."
        )

    full_text = "\n\n".join(pages)

    return full_text