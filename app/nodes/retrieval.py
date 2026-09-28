from typing import Dict, List

from app.services.arxiv import search_arxiv
from curl_cffi import requests
import xml.etree.ElementTree as ET


ARXIV_API_URL = "https://export.arxiv.org/api/query"


def get_paper_by_id(arxiv_id: str) -> Dict:
    """
    Fetch a specific paper from arXiv using its ID.
    """

    params = {
        "id_list": arxiv_id,
    }

    try:
        response = requests.get(
            ARXIV_API_URL,
            params=params,
            impersonate="chrome",
            timeout=30,
        )
        response.raise_for_status()
    except Exception as exc:
        raise RuntimeError(
            f"Failed to retrieve arXiv paper {arxiv_id}: {exc}"
        ) from exc

    try:
        root = ET.fromstring(response.content)
    except ET.ParseError as exc:
        raise RuntimeError(
            f"Failed to parse arXiv response: {exc}"
        ) from exc

    namespace = {
        "atom": "http://www.w3.org/2005/Atom",
        "arxiv": "http://arxiv.org/schemas/atom",
    }

    entry = root.find("atom:entry", namespace)

    if entry is None:
        raise RuntimeError(
            f"No paper found on arXiv for ID: {arxiv_id}"
        )

    arxiv_url = entry.findtext(
        "atom:id",
        default="",
        namespaces=namespace,
    )

    authors = [
        author.findtext(
            "atom:name",
            default="",
            namespaces=namespace,
        )
        for author in entry.findall("atom:author", namespace)
    ]

    pdf_url = ""

    for link in entry.findall("atom:link", namespace):
        if link.attrib.get("title") == "pdf":
            pdf_url = link.attrib.get("href", "")
            break

    categories = [
        category.attrib.get("term", "")
        for category in entry.findall("atom:category", namespace)
    ]

    return {
        "title": entry.findtext(
            "atom:title",
            default="",
            namespaces=namespace,
        ).strip(),

        "authors": authors,

        "abstract": entry.findtext(
            "atom:summary",
            default="",
            namespaces=namespace,
        ).strip(),

        "arxiv_id": arxiv_url.rstrip("/").split("/")[-1],

        "published": entry.findtext(
            "atom:published",
            default="",
            namespaces=namespace,
        ),

        "pdf_url": pdf_url,

        "arxiv_url": arxiv_url,

        "categories": categories,
    }


def retrieve_papers(
    query_type: str,
    query: str,
    arxiv_id: str | None = None,
    max_results: int = 5,
) -> List[Dict]:
    """
    Retrieve papers either from a topic search or a specific arXiv ID.
    """

    if query_type == "paper":
        if not arxiv_id:
            raise ValueError(
                "arxiv_id is required when query_type is 'paper'."
            )

        return [get_paper_by_id(arxiv_id)]

    if query_type == "topic":
        return search_arxiv(
            query=query,
            max_results=max_results,
        )

    raise ValueError(
        f"Unsupported query type: {query_type}"
    )


def select_paper(papers: List[Dict]) -> Dict:
    """
    Select a paper from retrieved candidates.

    For now, use the first result returned by arXiv.
    Ranking can be added later if needed.
    """

    if not papers:
        raise RuntimeError(
            "No papers were retrieved from arXiv."
        )

    return papers[0]