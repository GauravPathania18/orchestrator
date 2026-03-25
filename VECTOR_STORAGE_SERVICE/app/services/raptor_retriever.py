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
        max_distance: float = 2.0
    ) -> Dict[str, Any]:
        """
        Retrieve documents using RAPTOR hierarchical approach:
        1. Get summaries first
        2. Extract cluster IDs from summaries
        3. Get chunks from those clusters
        """
        print(f"🎯 RAPTOR RETRIEVER CALLED: query='{query}', max_distance={max_distance}")
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
        if not cluster_ids:
            logging.warning("_extract_cluster_ids returned empty — check if summaries retrieved any results")
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
            logging.info(f"🔍 Retrieving summaries with min_confidence={min_confidence}, max_distance={max_distance}")
            
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
            
            logging.info(f"📊 Raw summary query results: {len(results.get('ids', [[]])[0]) if results.get('ids') else 0} documents")
            
            # Filter by distance threshold
            if "distances" in results and results["distances"]:
                filtered_results = self._filter_by_distance(results, max_distance)
                logging.info(f"🎯 After distance filter: {len(filtered_results['documents'])} documents")
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
        """Extract unique cluster IDs from summary metadata."""
        cluster_ids = set()

        metadatas = summaries.get("metadatas", [])

        for metadata in metadatas:
            # After _filter_by_distance, metadatas is always a flat list of dicts
            if isinstance(metadata, dict):
                cluster_id = metadata.get("cluster_id")
                if cluster_id is not None:
                    cluster_ids.add(int(cluster_id))
            elif isinstance(metadata, list):
                # Defensive: handle raw ChromaDB list-of-lists if called without filtering
                for m in metadata:
                    if isinstance(m, dict):
                        cluster_id = m.get("cluster_id")
                        if cluster_id is not None:
                            cluster_ids.add(int(cluster_id))

        logging.info(f"Extracted cluster IDs: {cluster_ids}")
        return list(cluster_ids)

    def _filter_by_distance(self, results: Dict[str, Any], max_distance: float) -> Dict[str, Any]:
        """Filter results by distance threshold"""
        logging.info(f"🔍 Filtering by distance: max_distance={max_distance}")
        
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
        
        logging.info(f"📊 Raw results: ids={len(ids_list)}, distances={len(distances_list)}, docs={len(documents_list)}, metas={len(metadatas_list)}")
        
        # Check that all required lists exist and have elements
        if ids_list and distances_list and documents_list and metadatas_list:
            first_ids = ids_list[0] if ids_list else []
            first_distances = distances_list[0] if distances_list else []
            first_documents = documents_list[0] if documents_list else []
            first_metadatas = metadatas_list[0] if metadatas_list else []
            
            logging.info(f"📋 First batch: {len(first_ids)} items, distances: {first_distances}")
            
            for i, dist in enumerate(first_distances):
                logging.info(f"  Item {i}: dist={dist}, max_dist={max_distance}, pass={dist is not None and dist <= max_distance}")
                if dist is not None and dist <= max_distance:
                    filtered["ids"].append(first_ids[i])
                    filtered["documents"].append(first_documents[i])
                    filtered["metadatas"].append(first_metadatas[i])
                    filtered["distances"].append(dist)
        
        logging.info(f"✅ Filtered results: {len(filtered['documents'])} documents")
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
