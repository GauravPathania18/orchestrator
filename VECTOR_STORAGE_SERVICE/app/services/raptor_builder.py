import uuid
import logging
from typing import List, Dict, Any
from sklearn.cluster import KMeans
import numpy as np

from .embedder import get_embedding
from .vector_store import VectorStore
from .chunker import TextChunker, ChunkingConfig


class RaptorBuilder:
    def __init__(self, vector_store: VectorStore, cluster_size: int = 4, chunk_size: int = 500):
        self.vector_store = vector_store
        self.cluster_size = cluster_size
        
        # Initialize smart chunker with sentence awareness
        chunking_config = ChunkingConfig(
            max_chunk_size=chunk_size,
            chunk_overlap=50,
            min_chunk_size=50,
            respect_sentences=True
        )
        self.chunker = TextChunker(chunking_config)
        
        logging.info(f"🌲 RaptorBuilder initialized with cluster_size={cluster_size}, chunk_size={chunk_size}")

    def ingest(self, documents: List[str]):
        """Main ingestion pipeline for RAPTOR hierarchical clustering"""
        logging.info(f"📥 Starting RAPTOR ingestion for {len(documents)} documents")
        
        # Step 1: Chunk documents
        chunks = self._chunk_documents(documents)
        logging.info(f"🔪 Created {len(chunks)} chunks")
        
        # Step 2: Generate embeddings for chunks
        embeddings = self._get_embeddings(chunks)
        logging.info(f"🔢 Generated embeddings for {len(embeddings)} chunks")
        
        # Step 3: Cluster embeddings
        clusters = self._cluster_embeddings(embeddings)
        logging.info(f"🎯 Clustered chunks into {len(set(clusters['labels']))} clusters")
        
        # Step 4: Store chunks with cluster metadata
        chunk_ids = self._store_chunks(chunks, embeddings, clusters)
        logging.info(f"💾 Stored {len(chunk_ids)} chunks in vector store")
        
        # Step 5: Build and store summaries
        summaries = self._build_summaries(chunks, clusters)
        summary_ids = self._store_summaries(summaries)
        logging.info(f"📝 Created and stored {len(summary_ids)} summaries")
        
        logging.info("✅ RAPTOR ingestion completed successfully")
        return {
            "chunk_ids": chunk_ids,
            "summary_ids": summary_ids,
            "num_clusters": len(set(clusters['labels']))
        }

    def _chunk_documents(self, docs: List[str]) -> List[str]:
        """Split documents into semantically coherent chunks using sentence-based chunking."""
        return self.chunker.chunk_documents(docs, strategy="sentences")

    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for a list of texts"""
        embeddings = []
        for text in texts:
            try:
                embedding = get_embedding(text)
                embeddings.append(embedding)
            except Exception as e:
                logging.error(f"Failed to get embedding for text: {e}")
                # Use zero vector as fallback
                embeddings.append([0.0] * 768)  # Default dimension
        return embeddings

    def _cluster_embeddings(self, embeddings: List[List[float]]) -> Dict[str, Any]:
        """Cluster embeddings using K-means"""
        if len(embeddings) < 2:
            return {"labels": [0], "kmeans": None}
        
        # Convert to numpy array
        embedding_matrix = np.array(embeddings)
        
        # Determine optimal number of clusters
        k = max(1, len(embeddings) // self.cluster_size)
        k = min(k, len(embeddings))  # Don't exceed number of embeddings
        
        try:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(embedding_matrix)
            return {"labels": labels.tolist(), "kmeans": kmeans}
        except Exception as e:
            logging.error(f"Clustering failed: {e}")
            # Fallback: assign all to cluster 0
            return {"labels": [0] * len(embeddings), "kmeans": None}

    def _store_chunks(self, chunks: List[str], embeddings: List[List[float]], clusters: Dict[str, Any]) -> List[str]:
        """Store chunks in vector store with cluster metadata"""
        chunk_ids = []
        
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            chunk_id = str(uuid.uuid4())
            chunk_ids.append(chunk_id)
            
            metadata = {
                "type": "chunk",
                "level": 0,
                "cluster_id": int(clusters["labels"][i]),
                "text": chunk,
                "confidence": 1.0
            }
            
            self.vector_store.collection.add(
                ids=[chunk_id],
                documents=[chunk],
                embeddings=[embedding],
                metadatas=[metadata]
            )
        
        return chunk_ids

    def _build_summaries(self, chunks: List[str], clusters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build summaries for each cluster"""
        cluster_map = {}
        
        # Group chunks by cluster
        for idx, label in enumerate(clusters["labels"]):
            cluster_map.setdefault(label, []).append(chunks[idx])
        
        summaries = []
        
        for cluster_id, texts in cluster_map.items():
            combined = " ".join(texts)
            summary = self._summarize(combined)
            
            summaries.append({
                "cluster_id": cluster_id,
                "text": summary,
                "source_chunks": len(texts)
            })
        
        return summaries

    def _summarize(self, text: str) -> str:
        """Summarize text - placeholder for LLM-based summarization"""
        # For now, use simple truncation
        # TODO: Replace with actual LLM summarization
        if len(text) <= 200:
            return text
        return text[:200] + "..."

    def _store_summaries(self, summaries: List[Dict[str, Any]]) -> List[str]:
        """Store summaries in vector store"""
        summary_ids = []
        
        for summary in summaries:
            summary_id = str(uuid.uuid4())
            summary_ids.append(summary_id)
            
            # Get embedding for summary
            embedding = self._get_embeddings([summary["text"]])[0]
            
            metadata = {
                "type": "summary",
                "level": 1,
                "cluster_id": summary["cluster_id"],
                "text": summary["text"],
                "source_chunks": summary["source_chunks"],
                "confidence": 1.0
            }
            
            self.vector_store.collection.add(
                ids=[summary_id],
                documents=[summary["text"]],
                embeddings=[embedding],
                metadatas=[metadata]
            )
        
        return summary_ids
