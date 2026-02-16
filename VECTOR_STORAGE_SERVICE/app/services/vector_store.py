import uuid
import logging
import chromadb


from typing import List, Dict, Optional
from ..config import PERSIST_DIR, COLLECTION_NAME
from .utils import normalize_metadata
from .embedder import VECTOR_DIMENSION


class VectorStore:
    def __init__(self, expected_dimension: int = VECTOR_DIMENSION):
        if expected_dimension is None:
            raise ValueError("Vector dimension must be defined before initializing VectorStore.")

        self.client = chromadb.PersistentClient(path=PERSIST_DIR)
        self.collection = self.client.get_or_create_collection(name=COLLECTION_NAME)
        self.expected_dimension = expected_dimension

        logging.info(f"🔒 VectorStore initialized with dimension {self.expected_dimension}")


    # -------------------------
    # STORE VECTOR (UNCHANGED)
    # -------------------------
    def store_vector(self, vector: List[float], metadata: Dict):
        if not isinstance(metadata, dict):
            raise ValueError("Metadata must be a dictionary.")

        if len(vector) != self.expected_dimension:
            raise ValueError(
                f"Vector length mismatch. Expected {self.expected_dimension}, got {len(vector)}"
            )
        metadata = normalize_metadata(metadata)

        # Ensure confidence always exists
        metadata["confidence"] = float(metadata.get("confidence", 0.0))

        doc_id = str(uuid.uuid4())

        self.collection.add(
            ids=[doc_id],
            embeddings=[vector],
            metadatas=[metadata],
            documents=[metadata.get("text", "")]
        )

        logging.info(f"✅ Vector stored with ID {doc_id}")
        return doc_id

    # -------------------------
    # GET ALL
    # -------------------------
    def get_all(self, limit: Optional[int] = None):
        return self.collection.get(limit=limit)
    
    # -------------------------
    # GET BY ID (ENCAPSULATED)
    # -------------------------
    def get_by_id(self, doc_id: str):
        return self.collection.get(ids=[doc_id])
    

    # -------------------------
    # UPDATE METADATA (ENCAPSULATED)
    # -------------------------
    def update_metadata(self, doc_id: str, metadata: Dict):
        metadata = normalize_metadata(metadata)
        metadata["confidence"] = float(metadata.get("confidence", 0.0))

        self.collection.update(
            ids=[doc_id],
            metadatas=[metadata]
        )

        logging.info(f"🔄 Metadata updated for {doc_id}")



    # -----------------------------------
    # QDRANT-LIKE SEARCH (CHROMA-CORRECT)
    # -----------------------------------
    def search(
        self,
        query_vector: List[float],
        top_k: int = 3,
        domain: Optional[str] = None,
        entity_type: Optional[str] = None,
        min_confidence: float = 0.6
    ):
        if len(query_vector) != self.expected_dimension:
            raise ValueError(
                f"Query vector length mismatch. Expected {self.expected_dimension}"
            )

        filters = []

        # Always enforce confidence filter
        filters.append({"confidence": {"$gte": min_confidence}})

        if domain:
            filters.append({"domain": domain})

        if entity_type:
            filters.append({"entity_type": entity_type})

        # Chroma requires EXACTLY one top-level operator
        where = {"$and": filters} if len(filters) > 1 else (filters[0] if filters else None)

        return self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            where=where  # type: ignore
        )

# Global instance - initialized with default VECTOR_DIMENSION
vector_store = VectorStore(expected_dimension=VECTOR_DIMENSION)
