from langgraph.graph import StateGraph, END

from app.state import AgentState
from app.nodes.query import understand_query
from app.nodes.retrieval import retrieve_papers, select_paper
from app.services.pdf import download_pdf
from app.nodes.parser import parse_pdf
from app.nodes.chunker import chunk_text
from app.services.vector_store import VectorStore
from app.nodes.summarizer import generate_briefing
from app.nodes.qa import answer_question


# ============================================================
# Query Understanding
# ============================================================

def query_node(state: AgentState) -> AgentState:
    try:
        result = understand_query(state["user_query"])

        return {
            **state,
            "query_type": result["query_type"],
            "arxiv_id": result["arxiv_id"],
            "error": None,
        }

    except Exception as exc:
        return {
            **state,
            "error": f"Query understanding failed: {exc}",
        }


# ============================================================
# arXiv Retrieval
# ============================================================

def retrieval_node(state: AgentState) -> AgentState:
    try:
        papers = retrieve_papers(
            query_type=state["query_type"],
            query=state["user_query"],
            arxiv_id=state.get("arxiv_id"),
            max_results=5,
        )

        selected_paper = select_paper(papers)

        return {
            **state,
            "papers": papers,
            "selected_paper": selected_paper,
            "error": None,
        }

    except Exception as exc:
        return {
            **state,
            "error": f"Paper retrieval failed: {exc}",
        }


# ============================================================
# PDF Download
# ============================================================

def download_node(state: AgentState) -> AgentState:
    try:
        paper = state["selected_paper"]

        pdf_path = download_pdf(
            pdf_url=paper["pdf_url"],
            arxiv_id=paper["arxiv_id"],
        )

        return {
            **state,
            "pdf_path": pdf_path,
            "error": None,
        }

    except Exception as exc:
        return {
            **state,
            "error": f"PDF download failed: {exc}",
        }


# ============================================================
# PDF Parsing
# ============================================================

def parse_node(state: AgentState) -> AgentState:
    try:
        paper_text = parse_pdf(state["pdf_path"])

        return {
            **state,
            "paper_text": paper_text,
            "error": None,
        }

    except Exception as exc:
        return {
            **state,
            "error": f"PDF parsing failed: {exc}",
        }


# ============================================================
# Chunking
# ============================================================

def chunk_node(state: AgentState) -> AgentState:
    try:
        chunks = chunk_text(state["paper_text"])

        return {
            **state,
            "chunks": chunks,
            "error": None,
        }

    except Exception as exc:
        return {
            **state,
            "error": f"Text chunking failed: {exc}",
        }


# ============================================================
# Vector Store
# ============================================================

def vector_node(state: AgentState) -> AgentState:
    try:
        paper = state["selected_paper"]

        store = VectorStore()
        store.build(state["chunks"])

        store_path = store.save(paper["arxiv_id"])

        return {
            **state,
            "vector_store_path": store_path,
            "error": None,
        }

    except Exception as exc:
        return {
            **state,
            "error": f"Vector store creation failed: {exc}",
        }


# ============================================================
# Summarization
# ============================================================

def summarizer_node(state: AgentState) -> AgentState:
    try:
        paper = state["selected_paper"]

        briefing = generate_briefing(
            paper=paper,
            paper_text=state["paper_text"],
        )

        return {
            **state,
            "briefing": briefing,
            "error": None,
        }

    except Exception as exc:
        return {
            **state,
            "error": f"Briefing generation failed: {exc}",
        }


# ============================================================
# QA
# ============================================================

def qa_node(state: AgentState) -> AgentState:
    question = state.get("question")

    if not question:
        return {
            **state,
            "answer": None,
        }

    try:
        answer = answer_question(
            question=question,
            vector_store_path=state["vector_store_path"],
            conversation_history=state.get(
                "conversation_history",
                [],
            ),
        )

        history = list(
            state.get(
                "conversation_history",
                [],
            )
        )

        history.append(
            {
                "role": "user",
                "content": question,
            }
        )

        history.append(
            {
                "role": "assistant",
                "content": answer,
            }
        )

        return {
            **state,
            "answer": answer,
            "retrieved_chunks": [],
            "conversation_history": history,
            "error": None,
        }

    except Exception as exc:
        return {
            **state,
            "error": f"Question answering failed: {exc}",
        }


# ============================================================
# Error Handler
# ============================================================

def error_handler_node(state: AgentState) -> AgentState:
    """
    Handles failures gracefully instead of allowing the
    graph to crash with an unhandled exception.
    """

    print("\n" + "=" * 70)
    print("AGENT ERROR")
    print("=" * 70)

    print(
        f"\nThe agent could not complete the request.\n"
        f"Reason: {state.get('error', 'Unknown error')}"
    )

    print("\nPlease check the input or try again.")

    return state


# ============================================================
# Routing Helpers
# ============================================================

def route_after_query(state: AgentState) -> str:
    if state.get("error"):
        return "error"

    return "continue"


def route_after_retrieval(state: AgentState) -> str:
    if state.get("error"):
        return "error"

    return "continue"


def route_after_download(state: AgentState) -> str:
    if state.get("error"):
        return "error"

    return "continue"


def route_after_parse(state: AgentState) -> str:
    if state.get("error"):
        return "error"

    return "continue"


def route_after_chunking(state: AgentState) -> str:
    if state.get("error"):
        return "error"

    return "continue"


def route_after_vector_store(state: AgentState) -> str:
    if state.get("error"):
        return "error"

    return "continue"


def route_after_summarizer(state: AgentState) -> str:
    if state.get("error"):
        return "error"

    if state.get("question"):
        return "qa"

    return "end"


# ============================================================
# Build LangGraph
# ============================================================

def build_graph():

    graph = StateGraph(AgentState)

    # --------------------------------------------------------
    # Register Nodes
    # --------------------------------------------------------

    graph.add_node(
        "query_understanding",
        query_node,
    )

    graph.add_node(
        "arxiv_retrieval",
        retrieval_node,
    )

    graph.add_node(
        "pdf_download",
        download_node,
    )

    graph.add_node(
        "pdf_parser",
        parse_node,
    )

    graph.add_node(
        "chunking",
        chunk_node,
    )

    graph.add_node(
        "vector_store",
        vector_node,
    )

    graph.add_node(
        "summarizer",
        summarizer_node,
    )

    graph.add_node(
        "qa",
        qa_node,
    )

    graph.add_node(
        "error_handler",
        error_handler_node,
    )

    # --------------------------------------------------------
    # Entry Point
    # --------------------------------------------------------

    graph.set_entry_point(
        "query_understanding"
    )

    # --------------------------------------------------------
    # Query Understanding -> Retrieval / Error
    # --------------------------------------------------------

    graph.add_conditional_edges(
        "query_understanding",
        route_after_query,
        {
            "continue": "arxiv_retrieval",
            "error": "error_handler",
        },
    )

    # --------------------------------------------------------
    # Retrieval -> Download / Error
    # --------------------------------------------------------

    graph.add_conditional_edges(
        "arxiv_retrieval",
        route_after_retrieval,
        {
            "continue": "pdf_download",
            "error": "error_handler",
        },
    )

    # --------------------------------------------------------
    # Download -> Parser / Error
    # --------------------------------------------------------

    graph.add_conditional_edges(
        "pdf_download",
        route_after_download,
        {
            "continue": "pdf_parser",
            "error": "error_handler",
        },
    )

    # --------------------------------------------------------
    # Parser -> Chunking / Error
    # --------------------------------------------------------

    graph.add_conditional_edges(
        "pdf_parser",
        route_after_parse,
        {
            "continue": "chunking",
            "error": "error_handler",
        },
    )

    # --------------------------------------------------------
    # Chunking -> Vector Store / Error
    # --------------------------------------------------------

    graph.add_conditional_edges(
        "chunking",
        route_after_chunking,
        {
            "continue": "vector_store",
            "error": "error_handler",
        },
    )

    # --------------------------------------------------------
    # Vector Store -> Summarizer / Error
    # --------------------------------------------------------

    graph.add_conditional_edges(
        "vector_store",
        route_after_vector_store,
        {
            "continue": "summarizer",
            "error": "error_handler",
        },
    )

    # --------------------------------------------------------
    # Summarizer -> QA / END / Error
    # --------------------------------------------------------

    graph.add_conditional_edges(
        "summarizer",
        route_after_summarizer,
        {
            "qa": "qa",
            "end": END,
            "error": "error_handler",
        },
    )

    # --------------------------------------------------------
    # QA -> END
    # --------------------------------------------------------

    graph.add_edge(
        "qa",
        END,
    )

    # --------------------------------------------------------
    # Error Handler -> END
    # --------------------------------------------------------

    graph.add_edge(
        "error_handler",
        END,
    )

    return graph.compile()