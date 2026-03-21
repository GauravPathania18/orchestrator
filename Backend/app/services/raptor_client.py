import httpx
import logging
from typing import Dict, Any, List

# Configuration
VECTOR_STORAGE_URL = "http://vector-storage:8001"

class RaptorClient:
    """Client for interacting with RAPTOR-based vector storage service"""
    
    def __init__(self, base_url: str = VECTOR_STORAGE_URL):
        self.base_url = base_url.rstrip('/')
        self.logger = logging.getLogger(__name__)
    
    async def ingest_documents(self, documents: List[str], cluster_size: int = 4) -> Dict[str, Any]:
        """Ingest documents using RAPTOR hierarchical clustering"""
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.base_url}/raptor/index",
                    json={
                        "documents": documents,
                        "cluster_size": cluster_size
                    },
                    timeout=60.0
                )
                response.raise_for_status()
                return response.json()
            except httpx.HTTPError as e:
                self.logger.error(f"RAPTOR ingestion failed: {e}")
                raise
    
    async def raptor_query(
        self, 
        query: str, 
        k_summary: int = 3, 
        k_chunks: int = 10,
        top_k_final: int = 5
    ) -> Dict[str, Any]:
        """Query using RAPTOR hierarchical retrieval pipeline"""
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.base_url}/raptor/retrieve",
                    json={
                        "query": query,
                        "k_summary": k_summary,
                        "k_chunks": k_chunks,
                        "top_k_final": top_k_final
                    },
                    timeout=30.0
                )
                response.raise_for_status()
                return response.json()
            except httpx.HTTPError as e:
                self.logger.error(f"RAPTOR query failed: {e}")
                raise
    
    async def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get RAPTOR pipeline statistics"""
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(f"{self.base_url}/raptor/status")
                response.raise_for_status()
                return response.json()
            except httpx.HTTPError as e:
                self.logger.error(f"Failed to get pipeline stats: {e}")
                raise

# Global instance
raptor_client = RaptorClient()
