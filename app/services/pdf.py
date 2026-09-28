from pathlib import Path

from curl_cffi import requests


DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)


def download_pdf(pdf_url: str, arxiv_id: str) -> str:
    """
    Download an arXiv PDF and save it locally.

    Returns:
        Path to the downloaded PDF.
    """

    safe_id = arxiv_id.replace("/", "_")
    pdf_path = DATA_DIR / f"{safe_id}.pdf"

    if pdf_path.exists():
        return str(pdf_path)

    try:
        response = requests.get(
            pdf_url,
            impersonate="chrome",
            timeout=60,
        )
        response.raise_for_status()
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download PDF from arXiv: {exc}"
        ) from exc

    if not response.content:
        raise RuntimeError("Downloaded PDF is empty.")

    if not response.content.startswith(b"%PDF"):
        raise RuntimeError(
            "Downloaded content does not appear to be a valid PDF."
        )

    pdf_path.write_bytes(response.content)

    return str(pdf_path)