from typing import List, Dict
import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

from app.services.vector_store import VectorStore

load_dotenv()

MODEL_NAME = "gemini-3-flash-preview"

def get_client():
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Add it to the .env file."
        )

    return genai.Client(api_key=api_key)


def answer_question(
    question: str,
    vector_store_path: str,
    conversation_history: List[Dict] | None = None,
    top_k: int = 5,
) -> str:

    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    store = VectorStore()

    # Load the FAISS index created during the graph pipeline
    store_name = os.path.basename(vector_store_path)
    store.load(store_name)

    # Retrieve the most relevant chunks from the paper
    results = store.search(question, top_k=top_k)

    if not results:
        return "The answer is not stated in the paper."

    retrieved_chunks = [
        chunk for chunk, score in results
    ]

    context = "\n\n--- PAPER CHUNK ---\n\n".join(retrieved_chunks)

    history_text = ""

    if conversation_history:
        history_text = "\n".join(
            f"{item['role']}: {item['content']}"
            for item in conversation_history[-6:]
        )

    prompt = f"""
You are a research paper question-answering assistant.

Answer the user's question using ONLY the paper excerpts provided below.

Rules:
- Do not use outside knowledge.
- Do not invent information.
- If the answer cannot be supported by the provided paper excerpts,
  respond exactly:
  "The answer is not stated in the paper."
- Keep the answer clear and concise.
- When possible, explain the answer using evidence from the paper.
- Previous conversation may help understand the question, but it must
  not be treated as evidence.

PREVIOUS CONVERSATION:
{history_text}

PAPER EXCERPTS:
{context}

USER QUESTION:
{question}
"""

    client = get_client()

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
               
            ),
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to generate QA answer: {exc}"
        ) from exc

    if not response.text:
        raise RuntimeError("The LLM returned an empty answer.")

    return response.text.strip()