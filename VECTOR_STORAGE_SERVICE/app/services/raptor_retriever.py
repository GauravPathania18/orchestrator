import logging
from typing import List, Dict, Any, Optional

from .embedder import get_embedding
from .vector_store import VectorStore


class RaptorRetriever:
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        logging.info("🔍 RaptorRetriever initialized")

    def retrieve(
        self, 
        query: str, 
        k_summary: int = 3, 
        k_chunks: int = 10,
        min_confidence: float = 0.0,
        max_distance: float = 1.0
    ) -> Dict[str, Any]:
        """
        Retrieve documents using RAPTOR hierarchical approach:
        1. Get summaries first
        2. Extract cluster IDs from summaries
        3. Get chunks from those clusters
        """
        logging.info(f"🎯 Starting RAPTOR retrieval for query: '{query[:50]}...'")
        
        # Get query embedding
        try:
            q_emb = get_embedding(query)
        except Exception as e:
            logging.error(f"Failed to get query embedding: {e}")
            return {"summaries": [], "chunks": [], "error": str(e)}
        
        # Step 1: Retrieve summaries
        summaries = self._retrieve_summaries(
            q_emb, k_summary, min_confidence, max_distance
        )
        logging.info(f"📝 Retrieved {len(summaries['documents'])} summaries")
        
        # Step 2: Extract cluster IDs from summaries
        cluster_ids = self._extract_cluster_ids(summaries)
        logging.info(f"🏷️  Found cluster IDs: {cluster_ids}")
        
        # Step 3: Retrieve chunks from those clusters
        chunks = self._retrieve_chunks_from_clusters(
            q_emb, cluster_ids, k_chunks, min_confidence, max_distance
        )
        logging.info(f"🔪 Retrieved {len(chunks['documents'])} chunks from relevant clusters")
        
        result = {
            "summaries": summaries["documents"],
            "summary_metadatas": summaries.get("metadatas", []),
            "chunks": chunks["documents"],
            "chunk_metadatas": chunks.get("metadatas", []),
            "cluster_ids": cluster_ids
        }
        
        logging.info("✅ RAPTOR retrieval completed successfully")
        return result

    def _retrieve_summaries(
        self, 
        query_embedding: List[float], 
        k: int, 
        min_confidence: float,
        max_distance: float
    ) -> Dict[str, Any]:
        """Retrieve summary documents"""
        try:
            results = self.vector_store.collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                where={
                    "$and": [
                        {"type": "summary"},
                        {"confidence": {"$gte": min_confidence}}
                    ]
                }
            )
            
            # Filter by distance threshold
            if "distances" in results and results["distances"]:
                filtered_results = self._filter_by_distance(results, max_distance)
                return filtered_results
            
            return results
            
        except Exception as e:
            logging.error(f"Error retrieving summaries: {e}")
            return {"documents": [], "metadatas": [], "distances": [], "ids": []}

    def _retrieve_chunks_from_clusters(
        self,
        query_embedding: List[float],
        cluster_ids: List[int],
        k: int,
        min_confidence: float,
        max_distance: float
    ) -> Dict[str, Any]:
        """Retrieve chunks from specific clusters"""
        if not cluster_ids:
            logging.warning("No cluster IDs provided for chunk retrieval")
            return {"documents": [], "metadatas": [], "distances": [], "ids": []}
        
        try:
            results = self.vector_store.collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                where={
                    "$and": [
                        {"type": "chunk"},
                        {"cluster_id": {"$in": cluster_ids}},
                        {"confidence": {"$gte": min_confidence}}
                    ]
                }
            )
            
            # Filter by distance threshold
            if "distances" in results and results["distances"]:
                filtered_results = self._filter_by_distance(results, max_distance)
                return filtered_results
            
            return results
            
        except Exception as e:
            logging.error(f"Error retrieving chunks: {e}")
            return {"documents": [], "metadatas": [], "distances": [], "ids": []}

    def _extract_cluster_ids(self, summaries: Dict[str, Any]) -> List[int]:
        """Extract unique cluster IDs from summary metadata"""
        cluster_ids = set()
        
        if "metadatas" in summaries and summaries["metadatas"]:
            for metadata_list in summaries["metadatas"]:
                if isinstance(metadata_list, list):
                    for metadata in metadata_list:
                        if "cluster_id" in metadata:
                            cluster_ids.add(metadata["cluster_id"])
                elif isinstance(metadata_list, dict) and "cluster_id" in metadata_list:
                    cluster_ids.add(metadata_list["cluster_id"])
        
        return list(cluster_ids)

    def _filter_by_distance(self, results: Dict[str, Any], max_distance: float) -> Dict[str, Any]:
        """Filter results by distance threshold"""
        filtered = {
            "ids": [],
            "documents": [],
            "metadatas": [],
            "distances": []
        }
        
        # Get results with proper null checks
        ids_list = results.get("ids", [])
        distances_list = results.get("distances", [])
        documents_list = results.get("documents", [])
        metadatas_list = results.get("metadatas", [])
        
        # Check that all required lists exist and have elements
        if ids_list and distances_list and documents_list and metadatas_list:
            first_ids = ids_list[0] if ids_list else []
            first_distances = distances_list[0] if distances_list else []
            first_documents = documents_list[0] if documents_list else []
            first_metadatas = metadatas_list[0] if metadatas_list else []
            
            for i, dist in enumerate(first_distances):
                if dist is not None and dist <= max_distance:
                    filtered["ids"].append(first_ids[i])
                    filtered["documents"].append(first_documents[i])
                    filtered["metadatas"].append(first_metadatas[i])
                    filtered["distances"].append(dist)
        
        return filtered

    def get_cluster_info(self) -> Dict[str, Any]:
        """Get information about clusters in the vector store"""
        try:
            # Get all summaries to see cluster distribution
            all_summaries = self.vector_store.collection.get(
                where={"type": "summary"}
            )
            
            cluster_counts = {}
            if "metadatas" in all_summaries:
                for metadata in all_summaries["metadatas"]:
                    cluster_id = metadata.get("cluster_id", 0)
                    cluster_counts[cluster_id] = cluster_counts.get(cluster_id, 0) + 1
            
            # Get all chunks to see cluster distribution
            all_chunks = self.vector_store.collection.get(
                where={"type": "chunk"}
            )
            
            chunk_cluster_counts = {}
            if "metadatas" in all_chunks:
                for metadata in all_chunks["metadatas"]:
                    cluster_id = metadata.get("cluster_id", 0)
                    chunk_cluster_counts[cluster_id] = chunk_cluster_counts.get(cluster_id, 0) + 1
            
            return {
                "summary_clusters": cluster_counts,
                "chunk_clusters": chunk_cluster_counts,
                "total_summaries": len(all_summaries.get("ids", [])),
                "total_chunks": len(all_chunks.get("ids", []))
            }
            
        except Exception as e:
            logging.error(f"Error getting cluster info: {e}")
            return {"error": str(e)}
