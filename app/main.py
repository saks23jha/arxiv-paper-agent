from app.graph import build_graph


def print_list_or_string(value) -> None:
    """
    Print a value that may be either:
    - a list of strings
    - a single string
    """
    if isinstance(value, str):
        print(f"- {value}")
    elif isinstance(value, list):
        for item in value:
            print(f"- {item}")
    else:
        print("- N/A")


def print_briefing(briefing: dict) -> None:
    print("\n" + "=" * 70)
    print("EXECUTIVE PAPER BRIEFING")
    print("=" * 70)

    print(f"\nTitle: {briefing.get('title', 'N/A')}")

    authors = briefing.get("authors", [])
    if isinstance(authors, list):
        authors_text = ", ".join(authors)
    else:
        authors_text = str(authors)

    print(f"Authors: {authors_text}")
    print(f"arXiv ID: {briefing.get('arxiv_id', 'N/A')}")
    print(f"Published: {briefing.get('published', 'N/A')}")
    print(f"Link: {briefing.get('link', 'N/A')}")

    print("\nWhy it matters:")
    print(briefing.get("why_it_matters", "N/A"))

    print("\nProblem statement:")
    print(briefing.get("problem_statement", "N/A"))

    print("\nMethod / approach:")
    print_list_or_string(briefing.get("method", []))

    print("\nKey results / claims:")
    print_list_or_string(briefing.get("key_results", []))

    print("\nLimitations:")
    print_list_or_string(briefing.get("limitations", []))

    print("\nSuggested follow-up questions:")
    print_list_or_string(briefing.get("follow_up_questions", []))

    print("\n" + "=" * 70)


def main():
    print("=" * 70)
    print("AUTONOMOUS arXiv PAPER DIGEST & QA AGENT")
    print("=" * 70)

    user_query = input(
        "\nEnter a research topic or arXiv ID/URL:\n> "
    ).strip()

    if not user_query:
        print("Please enter a valid topic or arXiv ID.")
        return

    graph = build_graph()

    initial_state = {
        "user_query": user_query,
        "conversation_history": [],
    }

    try:
        state = graph.invoke(initial_state)
    except Exception as exc:
        print(f"\nAgent failed: {exc}")
        return

    if state.get("error"):
        print(f"\nAgent error: {state['error']}")
        return

    briefing = state.get("briefing")

    if briefing:
        print_briefing(briefing)

    print("\nQA mode")
    print("Ask questions about the paper.")
    print("Type 'exit' to finish.")

    while True:
        question = input("\nQuestion:\n> ").strip()

        if question.lower() == "exit":
            print("\nSession ended.")
            break

        if not question:
            continue

        try:
            qa_state = {
                **state,
                "question": question,
            }

            from app.graph import qa_node

            state = qa_node(qa_state)

            if state.get("error"):
                print(f"\nQA error: {state['error']}")
                continue

            print("\nAnswer:")
            print(state.get("answer", "No answer generated."))

        except Exception as exc:
            print(f"\nQA error: {exc}")


if __name__ == "__main__":
    main()