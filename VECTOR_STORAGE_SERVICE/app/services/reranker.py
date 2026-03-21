import logging
from typing import List, Tuple
import numpy as np

try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CROSS_ENCODER_AVAILABLE = False
    logging.warning("sentence-transformers not available. Reranker will use fallback scoring.")


class Reranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize reranker with CrossEncoder model
        
        Args:
            model_name: Name of the cross-encoder model to use
        """
        self.model_name = model_name
        self.model = None
        
        if CROSS_ENCODER_AVAILABLE:
            try:
                self.model = CrossEncoder(model_name)
                logging.info(f"🔥 Reranker initialized with model: {model_name}")
            except Exception as e:
                logging.error(f"Failed to load CrossEncoder model {model_name}: {e}")
                logging.info("🔄 Falling back to simple TF-IDF reranking")
        else:
            logging.warning("🔄 CrossEncoder not available, using fallback reranking")

    def rerank(
        self, 
        query: str, 
        documents: List[str], 
        top_k: int = 5,
        return_scores: bool = False
    ) -> List[str] | Tuple[List[str], List[float]]:
        """
        Rerank documents based on relevance to query
        
        Args:
            query: User query
            documents: List of documents to rerank
            top_k: Number of top documents to return
            return_scores: Whether to return scores along with documents
            
        Returns:
            List of reranked documents (and optionally scores)
        """
        if not documents:
            return [] if not return_scores else ([], [])
        
        if self.model is not None:
            return self._cross_encoder_rerank(query, documents, top_k, return_scores)
        else:
            return self._fallback_rerank(query, documents, top_k, return_scores)

    def _cross_encoder_rerank(
        self, 
        query: str, 
        documents: List[str], 
        top_k: int,
        return_scores: bool
    ) -> List[str] | Tuple[List[str], List[float]]:
        """Use CrossEncoder for reranking"""
        try:
            # Create query-document pairs
            pairs = [(query, doc) for doc in documents]
            
            # Get relevance scores
            scores = self.model.predict(pairs)
            
            # Sort documents by score (descending)
            ranked_docs_with_scores = sorted(
                zip(documents, scores),
                key=lambda x: x[1],
                reverse=True
            )
            
            # Extract top-k documents and scores
            top_docs = [doc for doc, _ in ranked_docs_with_scores[:top_k]]
            top_scores = [score for _, score in ranked_docs_with_scores[:top_k]]
            
            logging.info(f"🔥 CrossEncoder reranked {len(documents)} docs to top {len(top_docs)}")
            
            if return_scores:
                return top_docs, top_scores
            return top_docs
            
        except Exception as e:
            logging.error(f"CrossEncoder reranking failed: {e}")
            return self._fallback_rerank(query, documents, top_k, return_scores)

    def _fallback_rerank(
        self, 
        query: str, 
        documents: List[str], 
        top_k: int,
        return_scores: bool
    ) -> List[str] | Tuple[List[str], List[float]]:
        """Fallback reranking using simple lexical overlap"""
        try:
            # Simple lexical overlap scoring
            query_terms = set(query.lower().split())
            scores = []
            
            for doc in documents:
                doc_terms = set(doc.lower().split())
                # Jaccard similarity
                intersection = len(query_terms & doc_terms)
                union = len(query_terms | doc_terms)
                score = intersection / union if union > 0 else 0.0
                scores.append(score)
            
            # Sort by score
            ranked_docs_with_scores = sorted(
                zip(documents, scores),
                key=lambda x: x[1],
                reverse=True
            )
            
            top_docs = [doc for doc, _ in ranked_docs_with_scores[:top_k]]
            top_scores = [score for _, score in ranked_docs_with_scores[:top_k]]
            
            logging.info(f"🔄 Fallback reranking completed for {len(documents)} docs")
            
            if return_scores:
                return top_docs, top_scores
            return top_docs
            
        except Exception as e:
            logging.error(f"Fallback reranking failed: {e}")
            # Return original documents if all else fails
            if return_scores:
                return documents[:top_k], [0.0] * min(len(documents), top_k)
            return documents[:top_k]

    def batch_rerank(
        self, 
        queries: List[str], 
        documents_list: List[List[str]], 
        top_k: int = 5
    ) -> List[List[str]]:
        """
        Batch rerank multiple queries
        
        Args:
            queries: List of queries
            documents_list: List of document lists (one per query)
            top_k: Number of top documents to return per query
            
        Returns:
            List of reranked document lists
        """
        if len(queries) != len(documents_list):
            raise ValueError("Number of queries must match number of document lists")
        
        results = []
        for query, documents in zip(queries, documents_list):
            reranked = self.rerank(query, documents, top_k)
            results.append(reranked)
        
        return results

    def get_model_info(self) -> dict:
        """Get information about the loaded model"""
        return {
            "model_name": self.model_name,
            "is_cross_encoder": self.model is not None,
            "cross_encoder_available": CROSS_ENCODER_AVAILABLE
        }
