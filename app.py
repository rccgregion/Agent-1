
import os
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI
from rag import VectorIndex, load_json_qa, synthesize_answer
from fastapi.responses import HTMLResponse, FileResponse

load_dotenv()

DEBUG = os.getenv("DEBUG", "0") == "1"

app = FastAPI(title="Dr. E | RiteBridge | Champions QA Agent", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_PATH = os.getenv("KB_PATH", "data/knowledge.json")

class Ask(BaseModel):
    question: str
    top_k: int = 5
    temperature: float = 0.2

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
VI: Optional[VectorIndex] = None

def build_index():
    global VI
    qa_chunks = load_json_qa(DATA_PATH)
    VI = VectorIndex(client)
    VI.build_from_chunks(qa_chunks)

@app.on_event("startup")
def startup():
    if not os.path.exists(DATA_PATH):
        raise RuntimeError(f"Knowledge base not found at {DATA_PATH}.")
    build_index()

@app.post("/ask")
def ask(payload: Ask):
    if VI is None:
        raise HTTPException(status_code=500, detail="Index not ready")
    hits = VI.query(payload.question, k=payload.top_k)
    result = synthesize_answer(payload.question, hits, client, temperature=payload.temperature)
    return result

@app.post("/reload")
def reload_index():
    build_index()
    return {"status": "ok", "message": "Index rebuilt."}

@app.get("/", response_class=HTMLResponse)
def home():
    return FileResponse("static/index.html")
