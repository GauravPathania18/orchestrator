import logging
import asyncio
from typing import List, Dict, Any, Optional

from .raptor_retriever import RaptorRetriever
from .reranker import Reranker
from .vector_store import VectorStore


class RetrievalPipeline:
    """
    Main orchestration pipeline for RAPTOR-based RAG retrieval
    
    Pipeline Flow:
    1. User Query → Embedding
    2. Summary Retrieval → Cluster Filter → Chunk Retrieval
    3. Rerank → Final Context
    """
    
    def __init__(
        self, 
        vector_store: VectorStore,
        k_summary: int = 3,
        k_chunks: int = 10,
        top_k_final: int = 5,
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        min_confidence: float = 0.0,
        max_distance: float = 2.0
    ):
        """
        Initialize the retrieval pipeline
        
        Args:
            vector_store: VectorStore instance
            k_summary: Number of summaries to retrieve initially
            k_chunks: Number of chunks to retrieve from relevant clusters
            top_k_final: Number of final documents after reranking
            reranker_model: Name of the cross-encoder model for reranking
            min_confidence: Minimum confidence threshold for retrieval
            max_distance: Maximum distance threshold for similarity
        """
        self.vector_store = vector_store
        self.k_summary = k_summary
        self.k_chunks = k_chunks
        self.top_k_final = top_k_final
        self.min_confidence = min_confidence
        self.max_distance = max_distance
        
        # Initialize components
        self.retriever = RaptorRetriever(vector_store)
        self.reranker = Reranker(model_name=reranker_model)
        
        logging.info("🚀 RetrievalPipeline initialized")
        logging.info(f"📊 Config: k_summary={k_summary}, k_chunks={k_chunks}, top_k_final={top_k_final}")

    def run(
        self, 
        query: str,
        return_intermediate: bool = False,
        return_scores: bool = False,
        timeout: float = 15.0
    ) -> Dict[str, Any]:
        """
        Run the complete retrieval pipeline with timeout protection
        
        Args:
            query: User query
            return_intermediate: Whether to return intermediate results
            return_scores: Whether to return reranking scores
            timeout: Maximum time for pipeline execution
            
        Returns:
            Dictionary containing final context and optionally intermediate results
        """
        logging.info(f"🎯 Running RetrievalPipeline for query: '{query[:50]}...'")
        
        try:
            # Check if we're in an event loop
            try:
                loop = asyncio.get_running_loop()
                # We're in an event loop, use create_task
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(self._run_sync, query, return_intermediate, return_scores, timeout)
                    result = future.result(timeout=timeout)
                return result
            except RuntimeError:
                # No event loop, use asyncio.run
                result = asyncio.run(self._run_async_with_timeout(
                    query, return_intermediate, return_scores, timeout
                ))
                return result
            
        except Exception as e:
            logging.error(f"❌ Pipeline failed: {e}")
            return {
                "context": "",
                "query": query,
                "num_retrieved": 0,
                "num_final": 0,
                "error": str(e)
            }

    def _run_sync(
        self, 
        query: str, 
        return_intermediate: bool, 
        return_scores: bool,
        timeout: float
    ) -> Dict[str, Any]:
        """Synchronous pipeline execution"""
        try:
            # Step 1: RAPTOR retrieval
            retrieval_result = self.retriever.retrieve(query, self.k_summary, self.k_chunks, self.min_confidence, self.max_distance)
            
            # Combine summaries and chunks
            combined_docs = retrieval_result["summaries"] + retrieval_result["chunks"]
            logging.info(f"📚 Combined {len(retrieval_result['summaries'])} summaries + {len(retrieval_result['chunks'])} chunks = {len(combined_docs)} total docs")
            
            if not combined_docs:
                logging.warning("⚠️  No documents retrieved from RAPTOR")
                result = {
                    "context": "",
                    "query": query,
                    "num_retrieved": 0,
                    "num_final": 0
                }
                if return_intermediate:
                    result["retrieval_result"] = retrieval_result
                return result
            
            # Step 2: Rerank combined documents
            final_docs, scores = self._rerank_with_fallback(query, combined_docs, return_scores)
            
            # Step 3: Create final context
            context = "\n\n".join(final_docs)
            
            logging.info(f"✅ Pipeline completed: {len(combined_docs)} → {len(final_docs)} final docs")
            
            # Build result
            result = {
                "context": context,
                "query": query,
                "num_retrieved": len(combined_docs),
                "num_final": len(final_docs),
                "final_documents": final_docs
            }
            
            if return_scores:
                result["rerank_scores"] = scores
            
            if return_intermediate:
                result["retrieval_result"] = retrieval_result
                result["combined_documents"] = combined_docs
            
            return result
            
        except Exception as e:
            logging.error(f"❌ Sync pipeline failed: {e}")
            return {
                "context": "",
                "query": query,
                "num_retrieved": 0,
                "num_final": 0,
                "error": str(e)
            }

    async def _run_async_with_timeout(
        self, 
        query: str, 
        return_intermediate: bool, 
        return_scores: bool,
        timeout: float
    ) -> Dict[str, Any]:
        """Async pipeline execution with timeout"""
        
        # Step 1: RAPTOR retrieval (async)
        retrieval_task = asyncio.create_task(
            asyncio.to_thread(self.retriever.retrieve, query, self.k_summary, self.k_chunks, self.min_confidence, self.max_distance)
        )
        
        # Wait for retrieval with timeout
        retrieval_result = await asyncio.wait_for(retrieval_task, timeout=timeout * 0.6)
        
        # Combine summaries and chunks
        combined_docs = retrieval_result["summaries"] + retrieval_result["chunks"]
        logging.info(f"📚 Combined {len(retrieval_result['summaries'])} summaries + {len(retrieval_result['chunks'])} chunks = {len(combined_docs)} total docs")
        
        if not combined_docs:
            logging.warning("⚠️  No documents retrieved from RAPTOR")
            result = {
                "context": "",
                "query": query,
                "num_retrieved": 0,
                "num_final": 0
            }
            if return_intermediate:
                result["retrieval_result"] = retrieval_result
            return result
        
        # Step 2: Rerank combined documents (async with remaining timeout)
        rerank_task = asyncio.create_task(
            asyncio.to_thread(self._rerank_with_fallback, query, combined_docs, return_scores)
        )
        
        try:
            final_docs, scores = await asyncio.wait_for(rerank_task, timeout=timeout * 0.4)
        except asyncio.TimeoutError:
            logging.warning("⚠️  Reranking timeout, using original order")
            final_docs = combined_docs[:self.top_k_final]
            scores = [0.0] * len(final_docs)
        
        # Step 3: Create final context
        context = "\n\n".join(final_docs)
        
        logging.info(f"✅ Pipeline completed: {len(combined_docs)} → {len(final_docs)} final docs")
        
        # Build result
        result = {
            "context": context,
            "query": query,
            "num_retrieved": len(combined_docs),
            "num_final": len(final_docs),
            "final_documents": final_docs
        }
        
        if return_scores:
            result["rerank_scores"] = scores
        
        if return_intermediate:
            result["retrieval_result"] = retrieval_result
            result["combined_documents"] = combined_docs
        
        return result

    def _rerank_with_fallback(self, query: str, docs: List[str], return_scores: bool = False) -> tuple:
        """Rerank with fallback if CrossEncoder fails"""
        try:
            if return_scores:
                final_docs, scores = self.reranker.rerank(query, docs, self.top_k_final, return_scores=True)
                return final_docs, scores
            else:
                final_docs = self.reranker.rerank(query, docs, self.top_k_final, return_scores=False)
                return final_docs, [0.0] * len(final_docs)
        except Exception as e:
            logging.warning(f"⚠️  Reranking failed: {e}, using original order")
            final_docs = docs[:self.top_k_final]
            scores = [0.0] * len(final_docs)
            return final_docs, scores

    def batch_run(
        self, 
        queries: List[str],
        return_intermediate: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Run pipeline for multiple queries
        
        Args:
            queries: List of queries
            return_intermediate: Whether to return intermediate results
            
        Returns:
            List of pipeline results
        """
        results = []
        for query in queries:
            result = self.run(query, return_intermediate=return_intermediate)
            results.append(result)
        return results

    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get statistics about the pipeline components"""
        try:
            # Get cluster info from retriever
            cluster_info = self.retriever.get_cluster_info()
            
            # Get reranker info
            reranker_info = self.reranker.get_model_info()
            
            return {
                "pipeline_config": {
                    "k_summary": self.k_summary,
                    "k_chunks": self.k_chunks,
                    "top_k_final": self.top_k_final,
                    "min_confidence": self.min_confidence,
                    "max_distance": self.max_distance
                },
                "cluster_info": cluster_info,
                "reranker_info": reranker_info
            }
            
        except Exception as e:
            logging.error(f"Error getting pipeline stats: {e}")
            return {"error": str(e)}

    def update_config(
        self,
        k_summary: Optional[int] = None,
        k_chunks: Optional[int] = None,
        top_k_final: Optional[int] = None,
        min_confidence: Optional[float] = None,
        max_distance: Optional[float] = None
    ):
        """Update pipeline configuration"""
        if k_summary is not None:
            self.k_summary = k_summary
        if k_chunks is not None:
            self.k_chunks = k_chunks
        if top_k_final is not None:
            self.top_k_final = top_k_final
        if min_confidence is not None:
            self.min_confidence = min_confidence
        if max_distance is not None:
            self.max_distance = max_distance
        
        logging.info("🔧 Pipeline configuration updated")

    def format_for_llm(self, query: str, context: Optional[str] = None) -> str:
        """
        Format query and context for LLM prompt
        
        Args:
            query: User query
            context: Retrieved context (if None, will run pipeline)
            
        Returns:
            Formatted prompt string
        """
        if context is None:
            pipeline_result = self.run(query)
            context = pipeline_result["context"]
        
        if not context.strip():
            return f"""SYSTEM: You are a helpful AI assistant.

USER: {query}

ASSISTANT: """
        
        return f"""SYSTEM: You are a helpful AI assistant. Use the following retrieved context to answer the user's question.

RETRIEVED CONTEXT:
{context}

USER: {query}

ASSISTANT: """
