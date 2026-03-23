# Multi-Agent Research System

This project runs a multi-step "research + summarize + critique" workflow:

- Breaks a user topic into subtopics
- Searches the web (Tavily) for each subtopic
- Optionally enriches results with local RAG from your `data/` documents (ChromaDB + HuggingFace embeddings)
- Summarizes and then runs one or more critique passes before producing the final research-style summary

## Prerequisites

- Python (3.10+ recommended)

## Setup

1. Install dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

2. Configure environment variables:

   - Copy `.env.example` to `.env`
   - Set at least one of these LLM provider keys:
     - `OPENAI_API_KEY`
     - `GOOGLE_API_KEY`
     - `GROQ_API_KEY`
   - Set `TAVILY_API_KEY` (required for web search)

   ChromaDB + embeddings are configured via:
   - `EMBEDDING_MODEL`
   - `CHROMA_COLLECTION_NAME`

## Local documents (optional but recommended)

Add your `.txt` and/or `.pdf` files into the `data/` folder at the repo root.

On startup, the app:
- reads all files in `data/`
- chunks them
- embeds them with the HuggingFace sentence-transformer model
- stores/reuses them in the persistent `./chroma_db` directory

## Run

From the repo root (`d:/P/Multi_Agent_System`):

```powershell
python src/main.py
```

Then enter a topic when prompted. The program will print the final summary after the workflow completes.


