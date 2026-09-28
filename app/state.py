from typing import TypedDict, List, Dict, Optional


class AgentState(TypedDict, total=False):
    # User input
    user_query: str

    # Query understanding
    query_type: str  # "topic" or "paper"
    arxiv_id: Optional[str]

    # Retrieved paper metadata
    papers: List[Dict]
    selected_paper: Dict

    # PDF + parsed content
    pdf_path: Optional[str]
    paper_text: str

    # Chunking + retrieval
    chunks: List[str]
    embeddings: object
    vector_store_path: Optional[str]

    # Generated briefing
    briefing: Dict

    # QA
    question: Optional[str]
    retrieved_chunks: List[str]
    answer: Optional[str]
    conversation_history: List[Dict]

    # Error handling
    error: Optional[str]