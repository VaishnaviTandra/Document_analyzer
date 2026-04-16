from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from regex import match
import uvicorn
import os
import re
from rag_pipeline import *

app = FastAPI(title="RAG API", version="1.0")

# 🔹 Enable CORS (for React)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🔹 Request schema
class QueryRequest(BaseModel):
    query: str

# 🔹 Global variables (store current session data)
vector_store = None
chunks = None

# 🔹 Ensure data folder exists
os.makedirs("data", exist_ok=True)

# =========================
# 📤 UPLOAD PDF ENDPOINT
# =========================
@app.post("/upload/pdf")
async def upload_pdf(file: UploadFile = File(...)):
    global vector_store, chunks

    file_path = f"data/{file.filename}"

    # Save file
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # 🔹 Run full pipeline
    docs = load_data("pdf", file_path)
    processed_docs = preprocess_documents(docs)
    chunks = split_documents(processed_docs)
    vector_store = vectorize_documents(chunks)

    return {"message": "PDF processed successfully ✅"}

# =========================
# 🌐 WEB URL INGESTION
# =========================
@app.post("/upload/web")
async def upload_web(url: str):
    global vector_store, chunks

    docs = load_data("web", url)
    processed_docs = preprocess_documents(docs)
    chunks = split_documents(processed_docs)
    vector_store = vectorize_documents(chunks)

    return {"message": "Web content processed successfully ✅"}

# =========================
# ▶️ YOUTUBE INGESTION
# =========================
@app.post("/upload/youtube")
async def upload_youtube(video_id: str):
    global vector_store, chunks

    docs = load_data("youtube", video_id)
    processed_docs = preprocess_documents(docs)
    chunks = split_documents(processed_docs)
    vector_store = vectorize_documents(chunks)

    return {"message": "YouTube transcript processed successfully ✅"}

# =========================
# ❓ ASK QUESTION
# =========================
@app.post("/ask")
async def ask(request: QueryRequest):
    global vector_store, chunks

    print("Incoming query:", request.query)  # 👈 debug

    if vector_store is None or chunks is None:
        raise HTTPException(
            status_code=400,
            detail="No data uploaded. Please upload a document first."
        )

    result = rag_pipeline(request.query, vector_store, chunks)

    return {
        "answer": result["answer"],
        "sources": [doc.metadata for doc in result["sources"]]
    }
from fastapi import Form

from typing import Optional

@app.post("/ask-all")
async def ask_all(
    
    query: str = Form(""),
    file: UploadFile = File(None),
    url: Optional[str] = Form(None),
    youtube: Optional[str] = Form(None)
):
    global vector_store, chunks
    print("FILE RECEIVED:", file)
    print("URL:", url)
    print("YOUTUBE:", youtube)
    documents = []

    # 📄 PDF
    if file:
        file_path = f"data/{file.filename}"
        with open(file_path, "wb") as f:
            f.write(await file.read())

        documents.extend(load_data("pdf", file_path))

    # 🌐 WEB
    if url:
        documents.extend(load_data("web", url))

    # ▶️ YOUTUBE
    if youtube:
        match = re.search(r"v=([^&]+)", youtube)
        video_id = match.group(1) if match else youtube
        documents.extend(load_data("youtube", video_id))

    # 🔹 Process if new data
    if documents:
        processed_docs = preprocess_documents(documents)
        chunks = split_documents(processed_docs)
        vector_store = vectorize_documents(chunks)

    if vector_store is None:
        return {
        "answer": "⚠️ Please upload a PDF, website URL, or YouTube link first.",
        "sources": []
        }

    result = rag_pipeline(query, vector_store, chunks)

    return {
        "answer": result["answer"],
        "sources": [doc.metadata for doc in result["sources"]]
    }
# =========================
# 🏠 HOME
# =========================
@app.get("/")
def home():
    return {"message": "RAG Backend Running 🚀"}

# =========================
# ▶️ RUN SERVER
# =========================
if __name__ == "__main__":
    uvicorn.run("app:app", reload=True)