"""
vector_store.py

Collection schema 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
id        : "{photo_id}_{face_index}"
embedding : 512-dimensional float vector (L2-normalised)
metadata  : {
    "photo_id"          : str   (UUID of the parent photo)
    "face_index"        : int   (0-based index within the photo)
    "original_filename" : str   (human-readable original filename)
}
"""

from __future__ import annotations

from typing import List

import chromadb
from chromadb.config import Settings

from app.config import CHROMA_DIR, SIMILARITY_THRESHOLD, TOP_K

COLLECTION_NAME = "face_embeddings"


class VectorStore:
    """ChromaDB-backed vector store for face embeddings."""

    def __init__(self) -> None:
        self._client = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False),
        )
        self._col = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    # Indexing

    def index_faces(
        self,
        photo_id: str,
        original_filename: str,
        embeddings: List[List[float]],
    ) -> int:
        """
        Store all face embeddings for a single photo.

        Parameters
        ----------
        photo_id :
            Unique identifier for the photo (UUID string).
        original_filename :
            Human-readable name, stored in metadata for display.
        embeddings :
            List of 512-d vectors, one per detected face.

        Returns
        -------
        int
            Number of faces indexed.
        """
        if not embeddings:
            return 0

        ids       = [f"{photo_id}_{i}" for i in range(len(embeddings))]
        metadatas = [
            {
                "photo_id":          photo_id,
                "face_index":        i,
                "original_filename": original_filename,
            }
            for i in range(len(embeddings))
        ]

        self._col.upsert(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        return len(embeddings)

    # Querying

    def search(self, query_embedding: List[float]) -> List[str]:
        """
        Find all photos containing a face similar to *query_embedding*.

        Parameters
        ----------
        query_embedding :
            512-d vector for the probe face.

        Returns
        -------
        list[str]
            Deduplicated list of photo_ids ordered by best-match distance,
            filtered by SIMILARITY_THRESHOLD.
        """
        total_docs = self._col.count()
        if total_docs == 0:
            return []

        n_results = min(TOP_K, total_docs)

        result = self._col.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["metadatas", "distances"],
        )

        metadatas: list = result["metadatas"][0]   # type: ignore[index]
        distances: list = result["distances"][0]   # type: ignore[index]

        # Deduplicate: keep best distance per photo_id
        best: dict[str, float] = {}
        for meta, dist in zip(metadatas, distances):
            pid = meta["photo_id"]
            # ChromaDB cosine distance ∈ [0, 2]; lower = more similar
            if dist <= SIMILARITY_THRESHOLD:
                if pid not in best or dist < best[pid]:
                    best[pid] = dist

        # Sort by ascending distance (best match first)
        return sorted(best.keys(), key=lambda p: best[p])

    # Maintenance

    def delete_photo(self, photo_id: str) -> None:
        """Remove all face vectors associated with *photo_id*."""
        self._col.delete(where={"photo_id": photo_id})

    def count(self) -> int:
        """Return total number of indexed face vectors."""
        return self._col.count()


# Module-level singleton
_store: VectorStore | None = None


def get_store() -> VectorStore:
    global _store
    if _store is None:
        _store = VectorStore()
    return _store
