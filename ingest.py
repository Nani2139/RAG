"""Embed parsed records and store them in a local cosine Chroma collection.

Dealer columns are embedded as one vector. A brochure page is longer than
the embedding window, so its vector is the mean of the sentence embeddings
while the stored text stays the full page.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import chromadb
from langchain_huggingface import HuggingFaceEmbeddings

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CHROMA_DIR = ROOT / "chroma_db"
COLLECTION = "royalenfield"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def page_segments(text: str, max_chars: int = 400) -> list[str]:
    pieces = re.split(r"\n+|(?<=[.!?])\s+", text)
    segments = []
    for piece in pieces:
        piece = piece.strip()
        if not piece:
            continue
        while len(piece) > max_chars:
            segments.append(piece[:max_chars])
            piece = piece[max_chars:].strip()
        if piece:
            segments.append(piece)
    return segments or [text]


def metadata_for_chroma(metadata: dict) -> dict:
    clean = {}
    for key, value in metadata.items():
        if value is None or value == "":
            continue
        if isinstance(value, (str, int, float, bool)):
            clean[key] = value
        else:
            clean[key] = str(value)
    return clean


def main() -> None:
    dealers = load_jsonl(DATA_DIR / "dealers.jsonl")
    brochure = load_jsonl(DATA_DIR / "brochure.jsonl")
    embedder = HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        encode_kwargs={"normalize_embeddings": True},
    )
    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict] = []
    embeddings: list[list[float]] = []

    dealer_texts = [row["text"] for row in dealers]
    dealer_vectors = embedder.embed_documents(dealer_texts) if dealer_texts else []
    for row, vector in zip(dealers, dealer_vectors):
        ids.append(row["id"])
        documents.append(row["text"])
        metadatas.append(metadata_for_chroma(row["metadata"]))
        embeddings.append(vector)

    for row in brochure:
        # Spec sheets share the same field names, so one averaged vector
        # makes every model look alike. Each segment is embedded with the
        # model name, and every segment points at the full model text.
        title = row["metadata"].get("title", "")
        segments = page_segments(row["text"])
        labeled = [f"{title}. {segment}" if title else segment for segment in segments]
        vectors = embedder.embed_documents(labeled)
        for index, vector in enumerate(vectors):
            metadata = metadata_for_chroma(row["metadata"])
            metadata["segment"] = index
            ids.append(f"{row['id']}-{index}")
            documents.append(row["text"])
            metadatas.append(metadata)
            embeddings.append(vector)

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    existing = {collection.name for collection in client.list_collections()}
    if COLLECTION in existing:
        client.delete_collection(COLLECTION)
    collection = client.create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )
    batch = 128
    for start in range(0, len(ids), batch):
        collection.add(
            ids=ids[start : start + batch],
            documents=documents[start : start + batch],
            metadatas=metadatas[start : start + batch],
            embeddings=embeddings[start : start + batch],
        )
    print(f"stored {collection.count()} vectors in {CHROMA_DIR}")


if __name__ == "__main__":
    main()
