"""
Ollama LLM client for generating responses.

Requires Ollama to be running locally (default: http://localhost:11434)
or set OLLAMA_BASE_URL environment variable.
"""
import os
import httpx
from typing import List, Dict, Optional, AsyncGenerator

# Configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:1b")


async def generate_response(
    prompt: str,
    model: str = DEFAULT_MODEL,
    system_prompt: Optional[str] = None,
    temperature: float = 0.7,
    stream: bool = False
) -> str:
    """
    Generate a response from Ollama LLM.
    
    Args:
        prompt: The user prompt
        model: Ollama model to use (default: llama3.2)
        system_prompt: Optional system instructions
        temperature: Sampling temperature (0.0-1.0)
        stream: Whether to stream the response
        
    Returns:
        Generated response text
    """
    url = f"{OLLAMA_BASE_URL}/api/generate"
    
    payload = {
        "model": model,
        "prompt": prompt,
        "temperature": temperature,
        "stream": stream
    }
    
    if system_prompt:
        payload["system"] = system_prompt
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload, timeout=120.0)
            response.raise_for_status()
            
            if stream:
                # Handle streaming response
                full_response = ""
                async for line in response.aiter_lines():
                    if line:
                        import json
                        try:
                            data = json.loads(line)
                            if "response" in data:
                                full_response += data["response"]
                            if data.get("done", False):
                                break
                        except json.JSONDecodeError:
                            continue
                return full_response
            else:
                # Non-streaming response
                data = response.json()
                return data.get("response", "")
                
    except httpx.ConnectError:
        raise ConnectionError(
            f"Cannot connect to Ollama at {OLLAMA_BASE_URL}. "
            "Please ensure Ollama is running."
        )
    except httpx.HTTPStatusError as e:
        raise RuntimeError(f"Ollama API error: {e.response.text}")
    except Exception as e:
        raise RuntimeError(f"Error calling Ollama: {e}")


async def chat_completion(
    messages: List[Dict[str, str]],
    model: str = DEFAULT_MODEL,
    temperature: float = 0.7
) -> str:
    """
    Generate a chat completion using Ollama's chat API.
    
    Args:
        messages: List of message dicts with "role" and "content"
        model: Ollama model to use
        temperature: Sampling temperature
        
    Returns:
        Generated response text
    """
    url = f"{OLLAMA_BASE_URL}/api/chat"
    
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "stream": False
    }
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url, json=payload, timeout=120.0)
            response.raise_for_status()
            data = response.json()
            return data.get("message", {}).get("content", "")
            
    except httpx.ConnectError:
        raise ConnectionError(
            f"Cannot connect to Ollama at {OLLAMA_BASE_URL}. "
            "Please ensure Ollama is running."
        )
    except Exception as e:
        raise RuntimeError(f"Error calling Ollama chat API: {e}")


async def list_models() -> List[str]:
    """List available Ollama models."""
    url = f"{OLLAMA_BASE_URL}/api/tags"
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=10.0)
            response.raise_for_status()
            data = response.json()
            models = data.get("models", [])
            return [m.get("name", m.get("model")) for m in models]
    except Exception as e:
        raise RuntimeError(f"Error fetching Ollama models: {e}")


def create_rag_prompt(query: str, retrieved_docs: List[Dict]) -> str:
    """
    Create a RAG prompt with context from retrieved documents.
    
    Args:
        query: User query
        retrieved_docs: List of retrieved documents with text/content
        
    Returns:
        Formatted prompt for the LLM
    """
    context_parts = []
    for i, doc in enumerate(retrieved_docs[:5], 1):  # Limit to top 5 docs
        text = doc.get("document") or doc.get("text") or doc.get("content", "")
        if text:
            context_parts.append(f"Document {i}:\n{text}\n")
    
    context = "\n".join(context_parts) if context_parts else "No relevant documents found."
    
    prompt = f"""You are a helpful AI assistant with access to a knowledge base. Use the provided context to answer the user's question. If the context doesn't contain relevant information, say so clearly.

Context:
{context}

User Question: {query}

Please provide a clear, accurate answer based on the context above. If the context is insufficient, say "I don't have enough information to answer that question."

Answer:"""
    
    return prompt
