import httpx
import logging
from typing import Dict, Any, List

from app.core.config import VECTOR_SERVICE_URL


def _check_response(data: dict, service_name: str = "RAPTOR") -> dict:
    """Validate standard contract response and extract data payload."""
    if data.get("status") != "success":
        error = data.get("error", {})
        raise RuntimeError(f"{service_name} error: {error.get('message', 'Unknown error')}")
    return data.get("data", {})


class RaptorClient:
    """Client for interacting with RAPTOR-based vector storage service"""

    def __init__(self, base_url: str = VECTOR_SERVICE_URL):
        self.base_url = base_url.rstrip('/')
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"RaptorClient initialized → targeting: {self.base_url}")

    async def ingest_documents(self, documents: List[str], cluster_size: int = 4) -> Dict[str, Any]:
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.base_url}/raptor/index",
                    json={"documents": documents, "cluster_size": cluster_size},
                    timeout=60.0
                )
                response.raise_for_status()
                data = response.json()
                return _check_response(data, "RAPTOR Ingest")
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
                data = response.json()
                return _check_response(data, "RAPTOR Query")
            except httpx.HTTPError as e:
                self.logger.error(f"RAPTOR query failed: {e}")
                raise

    # NEW METHOD (IMPORTANT)
    async def retrieve(self, query: str, top_k: int = 5):
        """Unified retrieval interface for evaluation"""
        data = await self.raptor_query(
            query=query,
            k_summary=2,
            k_chunks=8,
            top_k_final=top_k
        )

        # Correct key is "final_documents" from vector store's RAPTOR pipeline
        final_docs = data.get("final_documents", [])
        
        if not final_docs:
            self.logger.warning(f"RAPTOR retrieve returned 0 docs. Response keys: {list(data.keys())}")

        return [{"text": doc} for doc in final_docs]

    async def get_pipeline_stats(self) -> Dict[str, Any]:
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(f"{self.base_url}/raptor/status")
                response.raise_for_status()
                data = response.json()
                return _check_response(data, "RAPTOR Stats")
            except httpx.HTTPError as e:
                self.logger.error(f"Failed to get pipeline stats: {e}")
                raise


raptor_client = RaptorClient()