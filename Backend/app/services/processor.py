from typing import List
import re
import numpy as np
from app.services.embedding_client import get_embedding
from app.services.utils import cosine_similarity

def split_sentences(text: str) -> List[str]:
    """Simple sentence splitter using regex."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


async def semantic_chunk_text(text: str, threshold_percentile: float = 60.0) -> List[str]:
    """
    Split text into chunks based on semantic similarity between sentences.
    """
    sentences = split_sentences(text)
    if len(sentences) <= 1:
        return sentences

    # Get embeddings for all sentences
    embeddings = []
    for sentence in sentences:
        emb = await get_embedding(sentence)
        embeddings.append(emb)

    # Calculate similarities between consecutive sentences
    similarities = []
    for i in range(len(embeddings) - 1):
        sim = cosine_similarity(embeddings[i], embeddings[i+1])
        similarities.append(sim)

    # Determine threshold for splitting
    # Lower similarity means a potential topic shift
    if not similarities:
        return [text]
    
    threshold = np.percentile(similarities, 100 - threshold_percentile)
    
    chunks = []
    current_chunk = [sentences[0]]
    
    for i in range(len(similarities)):
        if similarities[i] < threshold:
            # Topic shift detected
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentences[i+1]]
        else:
            current_chunk.append(sentences[i+1])
            
    if current_chunk:
        chunks.append(" ".join(current_chunk))
        
    return chunks
