import json
import os
import time
from typing import Dict

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

MODEL_NAME = "gemini-3-flash-preview"


def get_client():
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Add it to the .env file."
        )

    return genai.Client(api_key=api_key)


def generate_briefing(paper: Dict, paper_text: str) -> Dict:

    prompt = f"""
You are a research paper analysis assistant.

Analyze ONLY the paper content provided below.
Do not invent facts that are not supported by the paper.

Return valid JSON with exactly these fields:

{{
  "title": "...",
  "authors": ["..."],
  "arxiv_id": "...",
  "published": "...",
  "link": "...",
  "why_it_matters": "...",
  "problem_statement": "...",
  "method": ["...", "..."],
  "key_results": ["...", "..."],
  "limitations": ["...", "..."],
  "follow_up_questions": ["...", "..."]
}}

Requirements:
- why_it_matters: one clear plain-English paragraph.
- problem_statement: explain what problem the paper addresses.
- method: concise bullet-style items.
- key_results: important results or claims explicitly supported by the paper.
- limitations: explicitly mention limitations discussed by the paper.
  If the paper does not clearly state limitations, say:
  "The paper does not explicitly state a dedicated limitations section."
- follow_up_questions: 3 useful questions someone could ask about this paper.
- Do not use outside knowledge.
- Do not fabricate numerical results.
- Keep the briefing concise but informative.

PAPER METADATA:
{json.dumps(paper, indent=2)}

PAPER TEXT:
{paper_text}
"""

    client = get_client()

    max_attempts = 3

    for attempt in range(1, max_attempts + 1):

        try:
            print(
                f"\nGenerating paper briefing "
                f"(attempt {attempt}/{max_attempts})..."
            )

            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.2,
                    response_mime_type="application/json",
                ),
            )

            if not response.text:
                raise RuntimeError(
                    "The LLM returned an empty response."
                )

            try:
                briefing = json.loads(response.text)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    "The LLM response was not valid JSON."
                ) from exc

            return briefing

        except Exception as exc:

            error_text = str(exc)

            print(
                f"Briefing generation attempt {attempt} failed: "
                f"{error_text}"
            )

            if attempt == max_attempts:
                raise RuntimeError(
                    f"Gemini briefing generation failed after "
                    f"{max_attempts} attempts: {exc}"
                ) from exc

            # Wait before retrying temporary API failures.
            wait_time = attempt * 5

            print(
                f"Retrying in {wait_time} seconds..."
            )

            time.sleep(wait_time)

    raise RuntimeError(
        "Unable to generate paper briefing."
    )