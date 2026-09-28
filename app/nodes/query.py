import re
from typing import Dict


ARXIV_ID_PATTERN = re.compile(
    r"(?:arxiv\.org/(?:abs|pdf)/)?"
    r"(\d{4}\.\d{4,5}(?:v\d+)?)",
    re.IGNORECASE,
)


def understand_query(user_query: str) -> Dict:
    """
    Determine whether the user provided:
    - a specific arXiv paper ID/URL
    - or a research topic
    """

    if not user_query or not user_query.strip():
        raise ValueError("Query cannot be empty.")

    query = user_query.strip()

    match = ARXIV_ID_PATTERN.search(query)

    if match:
        return {
            "query_type": "paper",
            "arxiv_id": match.group(1),
        }

    return {
        "query_type": "topic",
        "arxiv_id": None,
    }