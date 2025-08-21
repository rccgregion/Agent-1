
import os
import json
import faiss
import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass
from openai import OpenAI
import tiktoken
from pathlib import Path

EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-large")
CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")

@dataclass
class Chunk:
    id: str
    title: str
    text: str
    source: str

def _chunk_text(text: str, max_tokens: int = 600) -> List[str]:
    enc = tiktoken.get_encoding("cl100k_base")
    toks = enc.encode(text)
    out, cur = [], []
    for t in toks:
        cur.append(t)
        if len(cur) >= max_tokens:
            out.append(enc.decode(cur))
            cur = []
    if cur:
        out.append(enc.decode(cur))
    return out

def load_json_qa(qa_json_path: str) -> List[Chunk]:
    """
    Load a JSON structured like:
    {
      "SectionName": {
        "about": [
          {"question": "...", "answer":"..."},
          ...
        ],
        "faqs": [ ... ]
      },
      ...
    }
    Returns list of chunks (Q + A concatenated for better retrieval).
    """
    with open(qa_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    chunks: List[Chunk] = []
    for section, buckets in data.items():
        if isinstance(buckets, dict):
            for bucket_name, items in buckets.items():
                if not isinstance(items, list): 
                    continue
                for i, qa in enumerate(items):
                    q = qa.get("question","").strip()
                    a = qa.get("answer","").strip()
                    content = f"Q: {q}\nA: {a}"
                    title = f"{section} | {bucket_name} | {q[:80]}"
                    chunks.append(Chunk(
                        id=f"{section}/{bucket_name}/{i}",
                        title=title,
                        text=content,
                        source=f"{section}.{bucket_name}"
                    ))
        elif isinstance(buckets, list):
            for i, qa in enumerate(buckets):
                q = qa.get("question","").strip()
                a = qa.get("answer","").strip()
                content = f"Q: {q}\nA: {a}"
                title = f"{section} | {q[:80]}"
                chunks.append(Chunk(
                    id=f"{section}/{i}",
                    title=title,
                    text=content,
                    source=f"{section}"
                ))
    return chunks

class VectorIndex:
    def __init__(self, client: OpenAI):
        self.client = client
        self.index = None
        self.meta: List[Chunk] = []

    def build_from_chunks(self, chunks: List[Chunk]):
        texts = []
        metas = []
        for ch in chunks:
            parts = _chunk_text(ch.text, 600)
            for j, piece in enumerate(parts):
                texts.append(piece)
                metas.append(Chunk(
                    id=f"{ch.id}#part{j}",
                    title=ch.title,
                    text=piece,
                    source=ch.source
                ))
        if not texts:
            raise ValueError("No texts to index.")
        emb = self.client.embeddings.create(model=EMBED_MODEL, input=texts).data
        vectors = np.array([e.embedding for e in emb], dtype="float32")
        faiss.normalize_L2(vectors)
        dim = vectors.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(vectors)
        self.meta = metas

    def query(self, question: str, k: int = 5) -> List[Tuple[float, Chunk]]:
        if self.index is None:
            raise RuntimeError("Index not built.")
        emb = self.client.embeddings.create(model=EMBED_MODEL, input=[question]).data[0].embedding
        qv = np.array([emb], dtype="float32")
        faiss.normalize_L2(qv)
        D, I = self.index.search(qv, k)
        results: List[Tuple[float, Chunk]] = []
        for rank in range(len(I[0])):
            idx = I[0][rank]
            score = float(D[0][rank])
            results.append((score, self.meta[idx]))
        return results

def synthesize_answer(question: str, hits: List[Tuple[float, Chunk]], client: OpenAI, temperature: float = 0.2) -> Dict[str, str]:
    context_lines = []
    used_titles = []
    for score, meta in hits:
        context_lines.append(f"- {meta.title} :: {meta.text}")
        used_titles.append(meta.title)
    sys_prompt = (
        "You are an expert concierge for Dr. Ekaette Joseph-Isang, RiteBridge Consulting, and Champions Coaching Academy. "
        "Answer strictly from the provided context when questions concern these entities. "
        "For general domain questions in Dr. E's fields, answer clearly but note if the context lacks specifics. "
        "Never invent facts or numbers. Use a formal yet friendly tone. "
        "If a question asks for private or unavailable info, say you don't have it. "
        "For any medical guidance, include a brief disclaimer to consult a qualified professional."
    )
    user_prompt = f"Question: {question}\n\nContext:\n" + "\n".join(context_lines[:8])
    resp = client.chat.completions.create(
        model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=temperature
    )
    answer = resp.choices[0].message.content.strip()
    return {
        "answer": answer,
        "sources": list(dict.fromkeys(used_titles))
    }
