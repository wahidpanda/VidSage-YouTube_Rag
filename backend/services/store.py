"""Local vector store: sentence-transformers embeddings + FAISS index per video.

Everything here runs on the local machine — no paid API involved.
The embedding model is lazy-loaded so the server boots instantly.
"""
import json
from pathlib import Path

import numpy as np

from backend.config import VECTOR_DIR, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, TOP_K

_model = None


def _embedder():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def _store_path(video_id: str) -> Path:
    return VECTOR_DIR / video_id


def store_exists(video_id: str) -> bool:
    return (_store_path(video_id) / "index.faiss").exists()


def build_chunks(segments: list[dict]) -> list[dict]:
    """Merge caption segments into overlapping chunks, keeping the start
    timestamp of each chunk so answers can cite exact moments."""
    chunks, buf, start = [], "", None
    for seg in segments:
        if start is None:
            start = seg["start"]
        buf += (" " if buf else "") + seg["text"]
        if len(buf) >= CHUNK_SIZE:
            chunks.append({"text": buf, "start": start})
            buf = buf[-CHUNK_OVERLAP:] if CHUNK_OVERLAP else ""
            start = seg["start"]
    if buf.strip():
        chunks.append({"text": buf, "start": start or 0.0})
    return chunks


def create_store(video_id: str, segments: list[dict]) -> int:
    import faiss

    chunks = build_chunks(segments)
    texts = [c["text"] for c in chunks]
    vectors = _embedder().encode(texts, normalize_embeddings=True, show_progress_bar=False)
    vectors = np.asarray(vectors, dtype="float32")

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    path = _store_path(video_id)
    path.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(path / "index.faiss"))
    (path / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False))
    (path / "segments.json").write_text(json.dumps(segments, ensure_ascii=False))
    return len(chunks)


def search(video_id: str, query: str, k: int = TOP_K) -> list[dict]:
    import faiss

    path = _store_path(video_id)
    index = faiss.read_index(str(path / "index.faiss"))
    chunks = json.loads((path / "chunks.json").read_text())

    q = _embedder().encode([query], normalize_embeddings=True, show_progress_bar=False)
    q = np.asarray(q, dtype="float32")
    scores, ids = index.search(q, min(k, index.ntotal))

    results = []
    for score, i in zip(scores[0], ids[0]):
        if i == -1:
            continue
        c = dict(chunks[int(i)])
        c["score"] = float(score)
        results.append(c)
    return results


def load_segments(video_id: str) -> list[dict]:
    return json.loads((_store_path(video_id) / "segments.json").read_text())


def full_text(video_id: str, limit: int) -> str:
    segs = load_segments(video_id)
    text = " ".join(s["text"] for s in segs)
    return text[:limit]
