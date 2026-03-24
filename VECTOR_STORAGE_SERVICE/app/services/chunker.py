"""Smart document chunking with sentence awareness and overlap.

This module provides chunking strategies that preserve semantic meaning
by chunking at sentence boundaries instead of arbitrary character positions.
"""

import re
import logging
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class ChunkingConfig:
    """Configuration for chunking strategies."""
    max_chunk_size: int = 500  # Maximum characters per chunk
    chunk_overlap: int = 50    # Overlap between chunks in characters
    min_chunk_size: int = 50   # Minimum characters to consider a valid chunk
    respect_sentences: bool = True  # Whether to break at sentence boundaries


class TextChunker:
    """Intelligent text chunker with sentence awareness."""
    
    def __init__(self, config: Optional[ChunkingConfig] = None):
        self.config = config or ChunkingConfig()
        self.logger = logging.getLogger(__name__)
    
    def clean_text(self, text: str) -> str:
        """Clean text before chunking."""
        # Replace newlines and multiple spaces with single space
        text = re.sub(r'[\n\r\t]+', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def split_sentences(self, text: str) -> List[str]:
        """Split text into sentences using regex."""
        # Handle common abbreviations to avoid false splits
        text = re.sub(r'(?<!\w)(Dr|Mr|Mrs|Ms|Prof|Sr|Jr|vs|vol|Vol|inc|Inc)\.', r'\1<ABBR>', text)
        
        # Split on sentence endings followed by space and capital
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        
        # Restore abbreviations
        sentences = [s.replace('<ABBR>', '.') for s in sentences]
        
        # Clean and filter
        return [s.strip() for s in sentences if len(s.strip()) > 10]
    
    def chunk_with_overlap(
        self, 
        text: str, 
        chunk_size: int,
        overlap: int
    ) -> List[str]:
        """Create chunks with overlap between them."""
        if not text or len(text) <= chunk_size:
            return [text] if len(text) >= self.config.min_chunk_size else []
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            if end >= len(text):
                # Final chunk
                chunk = text[start:].strip()
                if len(chunk) >= self.config.min_chunk_size:
                    chunks.append(chunk)
                break
            
            # Find the best break point (prefer sentence ending)
            chunk_text = text[start:end]
            
            # Look for sentence boundary in overlap region
            overlap_start = max(start, end - overlap)
            overlap_region = text[overlap_start:end]
            
            # Find last sentence end in overlap region
            last_sentence_end = -1
            for match in re.finditer(r'[.!?]\s+', overlap_region):
                last_sentence_end = match.end()
            
            if last_sentence_end > 0:
                # Break at sentence boundary
                actual_end = overlap_start + last_sentence_end
                chunk = text[start:actual_end].strip()
                next_start = actual_end
            else:
                # Fallback: break at word boundary in overlap region
                space_idx = overlap_region.rfind(' ')
                if space_idx > 0:
                    actual_end = overlap_start + space_idx
                    chunk = text[start:actual_end].strip()
                    next_start = actual_end + 1  # Skip the space
                else:
                    # Hard break
                    chunk = text[start:end].strip()
                    next_start = end
            
            if len(chunk) >= self.config.min_chunk_size:
                chunks.append(chunk)
            
            start = next_start
        
        return chunks
    
    def chunk_sentences(
        self, 
        text: str, 
        max_tokens: Optional[int] = None
    ) -> List[str]:
        """
        Chunk text by grouping sentences together.
        
        Each chunk contains complete sentences up to max_tokens size.
        """
        max_size = max_tokens or self.config.max_chunk_size
        
        # Clean and split
        text = self.clean_text(text)
        sentences = self.split_sentences(text)
        
        if not sentences:
            # No proper sentences found, use word-based chunking
            return self._chunk_by_words(text, max_size)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_len = len(sentence)
            
            # If single sentence is too long, split it further
            if sentence_len > max_size:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = []
                    current_length = 0
                
                # Split long sentence by words
                sub_chunks = self._chunk_by_words(sentence, max_size)
                chunks.extend(sub_chunks)
                continue
            
            # Check if adding this sentence exceeds limit
            if current_length + sentence_len + (1 if current_chunk else 0) > max_size:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = []
                    current_length = 0
            
            current_chunk.append(sentence)
            current_length += sentence_len + (1 if len(current_chunk) > 1 else 0)
        
        # Don't forget the last chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        # Filter by minimum size
        return [c for c in chunks if len(c) >= self.config.min_chunk_size]
    
    def _chunk_by_words(self, text: str, max_size: int) -> List[str]:
        """Fallback chunking by words when sentences aren't available."""
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            word_len = len(word)
            
            if current_length + word_len + (1 if current_chunk else 0) > max_size:
                if current_chunk:
                    chunk_text = " ".join(current_chunk)
                    if len(chunk_text) >= self.config.min_chunk_size:
                        chunks.append(chunk_text)
                    current_chunk = []
                    current_length = 0
            
            current_chunk.append(word)
            current_length += word_len + (1 if current_chunk else 0)
        
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            if len(chunk_text) >= self.config.min_chunk_size:
                chunks.append(chunk_text)
        
        return chunks
    
    def chunk_documents(
        self, 
        documents: List[str],
        strategy: str = "sentences"
    ) -> List[str]:
        """
        Chunk multiple documents using specified strategy.
        
        Args:
            documents: List of documents to chunk
            strategy: "sentences" (default) or "overlap"
        
        Returns:
            List of all chunks from all documents
        """
        all_chunks = []
        
        for doc in documents:
            if not doc or not isinstance(doc, str):
                continue
            
            if strategy == "sentences":
                chunks = self.chunk_sentences(doc)
            elif strategy == "overlap":
                cleaned = self.clean_text(doc)
                chunks = self.chunk_with_overlap(
                    cleaned,
                    self.config.max_chunk_size,
                    self.config.chunk_overlap
                )
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            
            all_chunks.extend(chunks)
        
        self.logger.info(
            f"Chunked {len(documents)} documents into {len(all_chunks)} chunks "
            f"(strategy={strategy})"
        )
        return all_chunks


# Convenience function for backward compatibility
def chunk_documents(
    documents: List[str],
    max_chunk_size: int = 500,
    chunk_overlap: int = 50,
    min_chunk_size: int = 50,
    strategy: str = "sentences"
) -> List[str]:
    """
    Chunk documents with sentence awareness.
    
    Args:
        documents: List of documents to chunk
        max_chunk_size: Maximum characters per chunk
        chunk_overlap: Overlap between chunks (for overlap strategy)
        min_chunk_size: Minimum characters for a valid chunk
        strategy: "sentences" or "overlap"
    
    Returns:
        List of chunks
    """
    config = ChunkingConfig(
        max_chunk_size=max_chunk_size,
        chunk_overlap=chunk_overlap,
        min_chunk_size=min_chunk_size
    )
    chunker = TextChunker(config)
    return chunker.chunk_documents(documents, strategy=strategy)
