# Autonomous arXiv Paper Digest & QA Agent

An AI-powered research assistant that retrieves papers from arXiv, analyzes their content, generates a concise executive briefing, and answers follow-up questions using information retrieved directly from the paper.

## Overview

The agent accepts either:

- A research topic
- An arXiv paper ID
- An arXiv paper URL

For a given paper, the system:

1. Retrieves the paper from arXiv.
2. Downloads and extracts text from the PDF.
3. Splits the paper into overlapping chunks.
4. Generates embeddings for the chunks.
5. Stores the embeddings in a FAISS vector index.
6. Generates an executive briefing using Gemini.
7. Allows the user to ask follow-up questions about the same paper.
8. Retrieves the most relevant paper chunks for each question.
9. Generates answers grounded in the retrieved paper content.

The system is implemented as a stateful graph using LangGraph, with separate nodes for each major stage of the workflow.

---

## Architecture

```text
                         User Input
                             |
                             v
                  +----------------------+
                  | Query Understanding  |
                  +----------------------+
                             |
                             v
                  +----------------------+
                  |   arXiv Retrieval    |
                  +----------------------+
                             |
                             v
                  +----------------------+
                  |     PDF Download     |
                  +----------------------+
                             |
                             v
                  +----------------------+
                  |      PDF Parser      |
                  +----------------------+
                             |
                             v
                  +----------------------+
                  |       Chunking       |
                  +----------------------+
                             |
                             v
                  +----------------------+
                  | Sentence Transformer |
                  |      Embeddings      |
                  +----------------------+
                             |
                             v
                  +----------------------+
                  |        FAISS         |
                  |     Vector Store     |
                  +----------------------+
                             |
                    +--------+--------+
                    |                 |
                    v                 v
             +-------------+    +-------------+
             |  Summarizer |    |     QA      |
             |   Gemini    |    |    Gemini   |
             +-------------+    +-------------+
                    |                 |
                    v                 v
              Executive         Grounded
               Briefing           Answer

The workflow is implemented as an explicit stateful LangGraph rather than a single monolithic LLM prompt.

Key Features
1. Flexible Paper Input

The agent can work with:

Research topics
arXiv IDs
arXiv URLs
2. Automatic Paper Processing

The selected paper is automatically:

Downloaded
Parsed
Chunked
Embedded
Indexed in FAISS
3. Executive Briefing

For each paper, the agent generates:

Title
Authors
arXiv ID
Publication date
Paper link
Why the paper matters
Problem statement
Method / approach
Key results / claims
Limitations
Suggested follow-up questions
4. Grounded Question Answering

Users can ask questions about the paper after the briefing is generated.

For every question:

User Question
      |
      v
Question Embedding
      |
      v
FAISS Similarity Search
      |
      v
Relevant Paper Chunks
      |
      v
Gemini
      |
      v
Grounded Answer

The QA prompt instructs the model to use only the retrieved paper content and avoid unsupported information.

If the answer cannot be supported by the retrieved content, the agent responds:

The answer is not stated in the paper.

5. Stateful Conversation

Conversation history is maintained in the shared agent state so that follow-up questions can be asked naturally during the same session.

Tech Stack
Component	Technology
Language	Python
Agent Framework	LangGraph
Paper Source	arXiv API
PDF Processing	PyMuPDF
Embeddings	Sentence Transformers
Embedding Model	all-MiniLM-L6-v2
Vector Store	FAISS
LLM	Google Gemini
Interface	CLI
Project Structure
arxiv-paper-agent/
│
├── app/
│   ├── __init__.py
│   ├── state.py
│   ├── graph.py
│   ├── main.py
│   │
│   ├── nodes/
│   │   ├── query.py
│   │   ├── retrieval.py
│   │   ├── parser.py
│   │   ├── chunker.py
│   │   ├── summarizer.py
│   │   └── qa.py
│   │
│   └── services/
│       ├── arxiv.py
│       ├── pdf.py
│       └── vector_store.py
│
├── data/
├── tests/
├── .env
├── .gitignore
├── requirements.txt
└── README.md
How the Agent Works
Step 1 — Query Understanding

The input is analyzed to determine whether it represents:

A topic
A specific arXiv paper
Step 2 — Paper Retrieval

The official arXiv API is used to retrieve paper metadata and the PDF URL.

Step 3 — PDF Processing

The PDF is downloaded and its text is extracted using PyMuPDF.

Step 4 — Chunking

The extracted text is divided into overlapping chunks.

Current configuration:

Chunk size: 1200 words
Overlap: 200 words
Step 5 — Embeddings and FAISS

Each chunk is converted into a vector using all-MiniLM-L6-v2.

The vectors are stored in a local FAISS index for similarity search.

Step 6 — Executive Briefing

Gemini receives the paper content and produces a structured briefing covering the important aspects of the paper.

Step 7 — Question Answering

When the user asks a question, the system:

Embeds the question.
Searches FAISS.
Retrieves the most relevant chunks.
Sends those chunks to Gemini.
Generates a grounded answer.
Installation
1. Clone the repository
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd arxiv-paper-agent
2. Create a virtual environment

For Windows PowerShell:

python -m venv .venv

Activate it:

.venv\Scripts\Activate.ps1
3. Install dependencies
pip install -r requirements.txt
4. Configure Gemini API

Create a .env file in the project root:

GEMINI_API_KEY=your_api_key_here

Do not commit the .env file to GitHub.

Running the Project

Start the application with:

python -m app.main

The application will ask:

Enter a research topic or arXiv ID/URL:
>

You can provide an arXiv ID, for example:

2306.04338

The agent will process the paper and display the executive briefing.

After the briefing, the application enters QA mode:

QA mode
Ask questions about the paper.
Type 'exit' to finish.

You can then ask multiple questions about the same paper.

Example

Example paper used during development and testing:

Title: Changing Data Sources in the Age of Machine Learning for Official Statistics

arXiv ID: 2306.04338

Example interaction:

Question:
> What problem does this paper address?

Answer:
[Answer generated using retrieved paper content]

Question:
> What approach does the paper use?

Answer:
[Answer generated using retrieved paper content]

Question:
> What are the limitations of the paper?

Answer:
[Answer generated using retrieved paper content]

The session can be ended using:

exit
Design Decisions
Why LangGraph?

The assessment requires an explicit stateful graph with nodes, edges, and shared state.

LangGraph allows the workflow to be divided into independent stages such as retrieval, parsing, chunking, summarization, and QA.

This makes the system easier to understand, test, and extend.

Why FAISS?

FAISS provides a lightweight local vector store suitable for a single-paper retrieval workflow.

It avoids the need for additional database infrastructure.

Why Sentence Transformers?

all-MiniLM-L6-v2 provides local embeddings without requiring a separate embedding API.

This keeps the retrieval pipeline simple and cost-effective.

Why Separate Retrieval from Generation?

The LLM is not given responsibility for searching the entire paper.

Instead:

Question
   ↓
FAISS Retrieval
   ↓
Relevant Chunks
   ↓
Gemini
   ↓
Answer

This makes the QA process more grounded and reduces the chance of generating unsupported information.

Why CLI?

The assessment focuses on the agent architecture, retrieval, grounding, and reasoning workflow rather than frontend development.

A CLI keeps the implementation small while making the complete workflow easy to demonstrate.

Error Handling

The application validates and handles several failure cases, including:

Empty user input
arXiv retrieval failures
Paper not found
PDF download failures
Invalid PDF content
Empty extracted text
Empty chunk lists
Missing vector store
Missing Gemini API key
Invalid LLM responses
Limitations

This is a focused prototype rather than a production research platform.

Current limitations include:

Topic search currently selects the first retrieved paper.
Retrieval is focused on the selected paper.
PDF parsing is text-based and does not provide OCR for scanned PDFs.
FAISS storage is local.
The application uses a CLI interface.
Only arXiv papers are supported.
Future Improvements

Potential improvements include:

Better ranking of papers for topic-based searches
Hybrid BM25 + vector retrieval
Section-aware chunking
Citation-aware answers
Support for multiple papers in one session
Improved extraction of tables and figures
OCR support
Streaming responses
More automated tests
Web-based interface
Security

The Gemini API key is loaded from the .env file.

The following files should not be committed:

.env
.venv/
__pycache__/
*.pyc

Generated PDFs and local FAISS indexes can also be excluded from Git if desired.

Assessment Alignment

This project addresses the main assessment requirements:

Stateful agent graph with explicit nodes and edges
arXiv paper retrieval
PDF parsing
Text chunking
Embeddings and vector retrieval
Executive paper briefing
Grounded question answering
Conversation history
Failure handling
CLI demonstration