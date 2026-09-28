from typing import List, Dict
import xml.etree.ElementTree as ET

from curl_cffi import requests


ARXIV_API_URL = "https://export.arxiv.org/api/query"


def search_arxiv(query: str, max_results: int = 5) -> List[Dict]:
    """
    Search papers using the official arXiv API.

    Args:
        query: Natural-language research topic.
        max_results: Maximum number of papers to retrieve.

    Returns:
        List of paper metadata dictionaries.
    """

    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
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
            f"Failed to retrieve papers from arXiv: {exc}"
        ) from exc

    try:
        root = ET.fromstring(response.content)

    except ET.ParseError as exc:
        raise RuntimeError(
            f"Failed to parse arXiv API response: {exc}"
        ) from exc

    namespace = {
        "atom": "http://www.w3.org/2005/Atom",
        "arxiv": "http://arxiv.org/schemas/atom",
    }

    papers = []

    for entry in root.findall("atom:entry", namespace):

        arxiv_url = entry.findtext(
            "atom:id",
            default="",
            namespaces=namespace,
        )

        arxiv_id = arxiv_url.rstrip("/").split("/")[-1]

        authors = [
            author.findtext(
                "atom:name",
                default="",
                namespaces=namespace,
            )
            for author in entry.findall(
                "atom:author",
                namespace,
            )
        ]

        pdf_url = ""

        for link in entry.findall("atom:link", namespace):
            if link.attrib.get("title") == "pdf":
                pdf_url = link.attrib.get("href", "")
                break

        categories = [
            category.attrib.get("term", "")
            for category in entry.findall(
                "atom:category",
                namespace,
            )
        ]

        papers.append(
            {
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

                "arxiv_id": arxiv_id,

                "published": entry.findtext(
                    "atom:published",
                    default="",
                    namespaces=namespace,
                ),

                "pdf_url": pdf_url,

                "arxiv_url": arxiv_url,

                "categories": categories,
            }
        )

    return papers