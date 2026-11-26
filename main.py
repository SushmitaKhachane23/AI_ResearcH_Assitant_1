from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import rag

app = FastAPI(title="AI Research Assistant")
store = rag.Store()
MAX_BYTES = 10 * 1024 * 1024


class Question(BaseModel):
    question: str


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".pdf", ".txt", ".md")):
        raise HTTPException(400, "Upload a .pdf, .txt or .md file.")
    data = await file.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "File is larger than 10 MB.")
    text = rag.extract_text(file.filename, data)
    n = store.add_document(file.filename, text)
    if n == 0:
        raise HTTPException(422, "No readable text found (scanned PDFs are not supported).")
    return {"file": file.filename, "chunks": n, "documents": store.documents()}


@app.post("/ask")
def ask(body: Question):
    q = body.question.strip()
    if not q:
        raise HTTPException(400, "Question is empty.")
    sources = store.search(q)
    if not sources:
        raise HTTPException(400, "Upload a document first.")
    try:
        answer = rag.ask_llm(q, sources)
    except Exception as e:
        raise HTTPException(502, f"LLM call failed: {e}")
    return {"answer": answer, "sources": sources}


@app.get("/documents")
def documents():
    return {"documents": store.documents()}


@app.delete("/documents")
def clear():
    store.clear()
    return {"documents": []}


@app.get("/")
def home():
    return FileResponse("static/index.html")


app.mount("/static", StaticFiles(directory="static"), name="static")
