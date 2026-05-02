"""
rag_service.py - Retrieval-Augmented Generation service.

Responsibilities:
  - Parse and chunk FAQ text file
  - Generate sentence-transformer embeddings
  - Persist/load a Chroma vector store
  - Retrieve the top-k most relevant FAQ chunks for any query
"""

import logging
import re
from pathlib import Path
from typing import List, Dict, Any

from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import (
    EMBEDDING_MODEL,
    VECTOR_STORE_DIR,
    FAQ_FILE,
    RAG_TOP_K,
    RAG_CHROMA_PROBE_K,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)

logger = logging.getLogger("supportplus.rag")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Split text into overlapping chunks of approximately `chunk_size` characters.
    Tries to split on sentence boundaries first to preserve semantic coherence.
    """
    # Normalise whitespace
    text = re.sub(r"\n{3,}", "\n\n", text.strip())

    # Split on sentence endings to get natural segments
    sentences = re.split(r"(?<=[.?!])\s+", text)

    chunks: List[str] = []
    current = ""

    for sentence in sentences:
        # If adding this sentence stays within limit, accumulate
        if len(current) + len(sentence) + 1 <= chunk_size:
            current = (current + " " + sentence).strip()
        else:
            if current:
                chunks.append(current)
            # If a single sentence exceeds chunk_size, hard-split it
            if len(sentence) > chunk_size:
                for i in range(0, len(sentence), chunk_size - overlap):
                    chunks.append(sentence[i : i + chunk_size])
            else:
                current = sentence

    if current:
        chunks.append(current)

    # Build overlapping window: carry last `overlap` chars into next chunk
    if overlap > 0 and len(chunks) > 1:
        overlapped: List[str] = [chunks[0]]
        for i in range(1, len(chunks)):
            tail = chunks[i - 1][-overlap:]
            overlapped.append(tail + " " + chunks[i])
        return overlapped

    return chunks


def _parse_faq_file(faq_path: Path) -> List[Dict[str, str]]:
    """
    Parse the FAQ text file into a list of {"question": ..., "answer": ...} dicts.
    Also includes section headers as metadata.
    """
    if not faq_path.exists():
        logger.error("FAQ file not found: %s", faq_path)
        return []

    content = faq_path.read_text(encoding="utf-8")
    records: List[Dict[str, str]] = []
    current_section = "General"

    for block in re.split(r"\n{2,}", content):
        block = block.strip()
        if not block:
            continue

        # Detect SECTION header
        section_match = re.match(r"SECTION:\s*(.+)", block)
        if section_match:
            current_section = section_match.group(1).strip()
            continue

        # Detect Q: / A: pairs
        qa_match = re.match(r"Q:\s*(.+?)\nA:\s*(.+)", block, re.DOTALL)
        if qa_match:
            question = qa_match.group(1).strip()
            answer = qa_match.group(2).strip()
            records.append({
                "question": question,
                "answer": answer,
                "section": current_section,
                "text": f"Q: {question}\nA: {answer}",
            })

    logger.info("Parsed %d FAQ entries from %s", len(records), faq_path)
    return records


def _similarity_from_cosine_distance(distance: float) -> float:
    """
    Chroma returns cosine *distance* for the cosine space (0 = identical).
    Map to a 0–1 similarity score for thresholds and logging.
    """
    try:
        d = float(distance)
    except (TypeError, ValueError):
        return 0.0
    # similarity ≈ 1 - distance for normalized embedding cosine setups
    return max(0.0, min(1.0, 1.0 - d))


def _parse_qa_from_chunk(document: str) -> tuple[str, str]:
    """Split a stored FAQ chunk into (question, answer) when it follows Q:/A: layout."""
    m = re.match(r"Q:\s*(.+?)\nA:\s*(.+)", document.strip(), re.DOTALL)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return "", document.strip()


# ---------------------------------------------------------------------------
# RAG Service class
# ---------------------------------------------------------------------------

class RAGService:
    """
    Manages embedding creation and similarity-based retrieval against the FAQ
    knowledge base using Chroma as the vector store.
    """

    COLLECTION_NAME = "supportplus_faqs"

    def __init__(self) -> None:
        logger.info("Initialising RAGService (model=%s)…", EMBEDDING_MODEL)
        self._encoder = SentenceTransformer(EMBEDDING_MODEL)

        # Persistent Chroma client
        self._client = chromadb.PersistentClient(
            path=str(VECTOR_STORE_DIR),
            settings=ChromaSettings(anonymized_telemetry=False),
        )

        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

        # Index FAQs if the collection is empty
        if self._collection.count() == 0:
            logger.info("Vector store is empty – indexing FAQs…")
            self._index_faqs()
        else:
            logger.info("Vector store loaded (%d chunks).", self._collection.count())

    # ------------------------------------------------------------------ #
    #  Indexing                                                            #
    # ------------------------------------------------------------------ #

    def _index_faqs(self) -> None:
        """Parse FAQ file, chunk entries, embed, and upsert into Chroma."""
        faq_records = _parse_faq_file(FAQ_FILE)
        if not faq_records:
            logger.warning("No FAQ records found – vector store will be empty.")
            return

        documents: List[str] = []
        metadatas: List[Dict[str, Any]] = []
        ids: List[str] = []

        for idx, record in enumerate(faq_records):
            # Chunk based on combined Q+A text
            chunks = _chunk_text(record["text"])
            for c_idx, chunk in enumerate(chunks):
                chunk_id = f"faq_{idx}_{c_idx}"
                documents.append(chunk)
                metadatas.append({
                    "section": record["section"],
                    "question": record["question"][:200],
                    "source": "faq",
                    "faq_idx": idx,
                    "chunk_idx": c_idx,
                })
                ids.append(chunk_id)

        # Generate embeddings in one batch for efficiency
        logger.info("Generating embeddings for %d chunks…", len(documents))
        embeddings = self._encoder.encode(documents, show_progress_bar=False).tolist()

        # Upsert in batches of 500 to respect Chroma limits
        batch_size = 500
        for start in range(0, len(documents), batch_size):
            end = min(start + batch_size, len(documents))
            self._collection.upsert(
                ids=ids[start:end],
                documents=documents[start:end],
                metadatas=metadatas[start:end],
                embeddings=embeddings[start:end],
            )

        logger.info("Indexed %d chunks into vector store.", len(documents))

    def reindex(self) -> None:
        """Force a full re-index of the FAQ file (useful after updates)."""
        logger.info("Re-indexing FAQ knowledge base…")
        self._client.delete_collection(self.COLLECTION_NAME)
        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        self._index_faqs()

    # ------------------------------------------------------------------ #
    #  Retrieval                                                           #
    # ------------------------------------------------------------------ #

    def retrieve(self, query: str, top_k: int = RAG_TOP_K) -> List[Dict[str, Any]]:
        """
        Find the top-k **FAQ entries** (deduplicated) most similar to ``query``.

        Over-fetches chunks from Chroma, keeps the **best** match per ``faq_idx``,
        then returns the top ``top_k`` entries sorted by similarity (high first).

        Each dict includes:
          ``text``, ``section``, ``question``, ``answer``, ``score``, ``faq_idx``, ``source``.
        """
        if self._collection.count() == 0:
            logger.warning("Vector store is empty – returning no FAQ results.")
            return []

        try:
            n_total = self._collection.count()
            # Ask Chroma for more hits than we need so split chunks don't hide the right FAQ.
            probe = min(max(RAG_CHROMA_PROBE_K, top_k * 4, top_k), n_total)
            query_embedding = self._encoder.encode([query], show_progress_bar=False).tolist()
            results = self._collection.query(
                query_embeddings=query_embedding,
                n_results=probe,
                include=["documents", "metadatas", "distances"],
            )

            docs = results.get("documents", [[]])[0]
            metas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]

            # Best (lowest distance) chunk per logical FAQ row
            best: Dict[int, tuple[float, str, Dict[str, Any]]] = {}
            for doc, meta, dist in zip(docs, metas, distances):
                try:
                    faq_idx = int(meta.get("faq_idx", -1))
                except (TypeError, ValueError):
                    faq_idx = -1
                d = float(dist)
                if faq_idx not in best or d < best[faq_idx][0]:
                    best[faq_idx] = (d, doc, dict(meta))

            merged: List[Dict[str, Any]] = []
            for faq_idx, (dist, doc, meta) in best.items():
                if faq_idx < 0:
                    continue
                score = _similarity_from_cosine_distance(dist)
                q_parsed, a_parsed = _parse_qa_from_chunk(doc)
                q_meta = (meta.get("question") or "").strip()
                question = q_parsed or q_meta
                answer = a_parsed if a_parsed else doc.strip()
                merged.append({
                    "text": doc,
                    "section": meta.get("section", "") or "",
                    "question": question,
                    "answer": answer,
                    "score": round(score, 4),
                    "faq_idx": faq_idx,
                    "source": "faq",
                })

            merged.sort(key=lambda r: r["score"], reverse=True)
            # Filter low confidence matches (irrelevant chunks)
            merged = [m for m in merged if m["score"] >= 0.25]
            retrieved = merged[: max(1, top_k)]

            logger.debug(
                "RAG merged %d unique FAQs → returning top %d for query: '%s'",
                len(merged),
                len(retrieved),
                query[:80],
            )
            return retrieved

        except Exception as exc:
            logger.error("RAG retrieval failed: %s", exc, exc_info=True)
            return []

    def health(self) -> Dict[str, Any]:
        """Return health info for the /health endpoint."""
        return {
            "status": "ok",
            "chunks_indexed": self._collection.count(),
            "embedding_model": EMBEDDING_MODEL,
            "vector_store": str(VECTOR_STORE_DIR),
        }


# ---------------------------------------------------------------------------
# Module-level singleton (initialised once at import time)
# ---------------------------------------------------------------------------
_rag_service: RAGService | None = None


def get_rag_service() -> RAGService:
    """Return the module-level RAGService singleton, creating it if needed."""
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
    return _rag_service
