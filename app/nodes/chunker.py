from typing import List


def chunk_text(
    text: str,
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
) -> List[str]:
    """
    Split paper text into overlapping chunks.

    Args:
        text: Full extracted paper text.
        chunk_size: Approximate number of words per chunk.
        chunk_overlap: Number of words shared between consecutive chunks.

    Returns:
        List of text chunks.
    """

    if not text or not text.strip():
        raise ValueError("Cannot chunk empty text.")

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be smaller than chunk_size."
        )

    words = text.split()

    chunks = []
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))

        chunk = " ".join(words[start:end]).strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(words):
            break

        start = end - chunk_overlap

    if not chunks:
        raise RuntimeError("No chunks were created from the paper.")

    return chunks