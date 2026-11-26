# AI-Powered Research Assistant

Upload research papers or notes (PDF/TXT/MD) and ask questions. The app finds the most
relevant passages and asks an LLM to answer using only those passages, with citations.
This is Retrieval-Augmented Generation (RAG).

## How it works
1. **Ingest:** extract text from the file and split it into overlapping chunks.
2. **Embed:** turn each chunk into a vector with `BAAI/bge-small-en-v1.5` (fastembed, runs on CPU).
3. **Retrieve:** embed the question and pick the top 4 chunks by cosine similarity.
4. **Generate:** send those chunks plus the question to the LLM (Gemini) and return the answer with sources.

## Tech stack
Python, FastAPI, fastembed, NumPy, pypdf, Google Gemini API (free tier), plain HTML/JS frontend.

## Run locally
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export GEMINI_API_KEY=your_key_here              # Windows: set GEMINI_API_KEY=your_key_here
uvicorn main:app --reload
```
Open http://127.0.0.1:8000

## API
| Method | Path | Purpose |
|---|---|---|
| POST | /upload | Upload a document (multipart `file`) |
| POST | /ask | `{"question": "..."}` -> answer + sources |
| GET | /documents | List loaded documents |
| DELETE | /documents | Clear everything |

## Limitations
- Index is stored in memory, so it resets when the server restarts.
- Scanned (image-only) PDFs are not supported.
- Files are limited to 10 MB.

## Possible improvements
Persistent vector database, per-user sessions, OCR for scanned PDFs, streaming answers.

## Deploy on Render (free)
1. Push this folder to GitHub.
2. Render > New > Web Service > connect the repo.
3. Build command: `pip install -r requirements.txt`
4. Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
5. Environment: add `GEMINI_API_KEY` (and keep `.python-version` so Python 3.11 is used).
The first question after a cold start is slow because the embedding model downloads once.
