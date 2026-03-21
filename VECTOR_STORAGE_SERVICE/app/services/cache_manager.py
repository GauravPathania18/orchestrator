"""
Advanced caching system for RAPTOR services
"""

import time
import hashlib
import json
import logging
from typing import Any, Optional, Dict, List, Union
from dataclasses import dataclass
from threading import Lock, RLock
from collections import OrderedDict
import pickle

@dataclass
class CacheConfig:
    """Cache configuration"""
    max_size: int = 10000
    ttl_seconds: int = 3600  # 1 hour
    cleanup_interval: int = 300  # 5 minutes
    enable_compression: bool = True
    enable_persistence: bool = False
    persist_file: str = "cache_data.pkl"

class CacheEntry:
    """Individual cache entry with TTL"""
    
    def __init__(self, value: Any, ttl: int):
        self.value = value
        self.created_at = time.time()
        self.expires_at = self.created_at + ttl
        self.access_count = 0
        self.last_accessed = self.created_at
        self.size = self._calculate_size(value)
    
    def _calculate_size(self, value: Any) -> int:
        """Calculate approximate size of cached value"""
        try:
            return len(pickle.dumps(value))
        except:
            return len(str(value))
    
    def is_expired(self) -> bool:
        """Check if cache entry has expired"""
        return time.time() > self.expires_at
    
    def access(self) -> Any:
        """Access the cached value and update stats"""
        self.access_count += 1
        self.last_accessed = time.time()
        return self.value

class LRUCache:
    """Thread-safe LRU cache with TTL support"""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.lock = RLock()
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "total_requests": 0
        }
        self._last_cleanup = time.time()
    
    def _generate_key(self, key: Union[str, Dict, List]) -> str:
        """Generate consistent cache key from various input types"""
        if isinstance(key, str):
            return key
        elif isinstance(key, (dict, list)):
            # Create deterministic key from complex objects
            key_str = json.dumps(key, sort_keys=True, default=str)
            return hashlib.md5(key_str.encode()).hexdigest()
        else:
            return hashlib.md5(str(key).encode()).hexdigest()
    
    def get(self, key: Union[str, Dict, List]) -> Optional[Any]:
        """Get value from cache with thread safety"""
        cache_key = self._generate_key(key)
        
        with self.lock:
            self.stats["total_requests"] += 1
            self._cleanup_expired()
            
            if cache_key in self.cache:
                entry = self.cache[cache_key]
                
                if entry.is_expired():
                    del self.cache[cache_key]
                    self.stats["misses"] += 1
                    return None
                
                # Move to end (LRU)
                self.cache.move_to_end(cache_key)
                self.stats["hits"] += 1
                return entry.access()
            else:
                self.stats["misses"] += 1
                return None
    
    def set(self, key: Union[str, Dict, List], value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache with thread safety"""
        cache_key = self._generate_key(key)
        ttl = ttl or self.config.ttl_seconds
        
        with self.lock:
            self._cleanup_expired()
            
            # Check if we need to evict
            while len(self.cache) >= self.config.max_size:
                self._evict_lru()
            
            entry = CacheEntry(value, ttl)
            self.cache[cache_key] = entry
            
            return True
    
    def delete(self, key: Union[str, Dict, List]) -> bool:
        """Delete key from cache"""
        cache_key = self._generate_key(key)
        
        with self.lock:
            if cache_key in self.cache:
                del self.cache[cache_key]
                return True
            return False
    
    def clear(self):
        """Clear all cache entries"""
        with self.lock:
            self.cache.clear()
            self.stats = {
                "hits": 0,
                "misses": 0,
                "evictions": 0,
                "total_requests": 0
            }
    
    def _cleanup_expired(self):
        """Remove expired entries"""
        current_time = time.time()
        expired_keys = []
        
        for key, entry in self.cache.items():
            if entry.is_expired():
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.cache[key]
        
        self._last_cleanup = current_time
    
    def _evict_lru(self):
        """Evict least recently used entry"""
        if self.cache:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
            self.stats["evictions"] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self.lock:
            hit_rate = (self.stats["hits"] / self.stats["total_requests"] * 100 
                       if self.stats["total_requests"] > 0 else 0)
            
            return {
                "size": len(self.cache),
                "max_size": self.config.max_size,
                "hit_rate": round(hit_rate, 2),
                "hits": self.stats["hits"],
                "misses": self.stats["misses"],
                "evictions": self.stats["evictions"],
                "total_requests": self.stats["total_requests"],
                "memory_usage": sum(entry.size for entry in self.cache.values())
            }

class CacheManager:
    """Multi-cache manager with different cache types"""
    
    def __init__(self, config: Optional[CacheConfig] = None):
        self.config = config or CacheConfig()
        
        # Different cache types for different use cases
        self.embedding_cache = LRUCache(CacheConfig(
            max_size=5000,
            ttl_seconds=7200,  # 2 hours
            enable_compression=True
        ))
        
        self.raptor_cache = LRUCache(CacheConfig(
            max_size=1000,
            ttl_seconds=3600,  # 1 hour
            enable_compression=True
        ))
        
        self.query_cache = LRUCache(CacheConfig(
            max_size=2000,
            ttl_seconds=1800,  # 30 minutes
            enable_compression=True
        ))
        
        self.metadata_cache = LRUCache(CacheConfig(
            max_size=10000,
            ttl_seconds=86400,  # 24 hours
            enable_compression=False
        ))
    
    def cache_embedding(self, text: str, embedding: List[float]) -> None:
        """Cache text embedding"""
        self.embedding_cache.set(text, embedding)
    
    def get_cached_embedding(self, text: str) -> Optional[List[float]]:
        """Get cached text embedding"""
        return self.embedding_cache.get(text)
    
    def cache_raptor_result(self, query: str, result: Dict[str, Any]) -> None:
        """Cache RAPTOR query result"""
        cache_key = {
            "query": query,
            "k_summary": result.get("k_summary", 3),
            "k_chunks": result.get("k_chunks", 10),
            "top_k_final": result.get("top_k_final", 5)
        }
        self.raptor_cache.set(cache_key, result)
    
    def get_cached_raptor_result(self, query: str, k_summary: int = 3, k_chunks: int = 10, top_k_final: int = 5) -> Optional[Dict[str, Any]]:
        """Get cached RAPTOR result"""
        cache_key = {
            "query": query,
            "k_summary": k_summary,
            "k_chunks": k_chunks,
            "top_k_final": top_k_final
        }
        return self.raptor_cache.get(cache_key)
    
    def cache_query_result(self, query: str, context: str, metadata: Dict[str, Any]) -> None:
        """Cache complete query result"""
        cache_key = {
            "query": query,
            "context_hash": hashlib.md5(context.encode()).hexdigest()
        }
        self.query_cache.set(cache_key, {
            "context": context,
            "metadata": metadata,
            "cached_at": time.time()
        })
    
    def get_cached_query_result(self, query: str, context: str) -> Optional[Dict[str, Any]]:
        """Get cached query result"""
        cache_key = {
            "query": query,
            "context_hash": hashlib.md5(context.encode()).hexdigest()
        }
        return self.query_cache.get(cache_key)
    
    def cache_metadata(self, doc_id: str, metadata: Dict[str, Any]) -> None:
        """Cache document metadata"""
        self.metadata_cache.set(doc_id, metadata)
    
    def get_cached_metadata(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get cached document metadata"""
        return self.metadata_cache.get(doc_id)
    
    def invalidate_query_cache(self) -> None:
        """Invalidate query cache (useful after data updates)"""
        self.query_cache.clear()
        logging.info("Query cache invalidated")
    
    def invalidate_embedding_cache(self) -> None:
        """Invalidate embedding cache"""
        self.embedding_cache.clear()
        logging.info("Embedding cache invalidated")
    
    def get_all_stats(self) -> Dict[str, Any]:
        """Get statistics for all caches"""
        return {
            "embedding_cache": self.embedding_cache.get_stats(),
            "raptor_cache": self.raptor_cache.get_stats(),
            "query_cache": self.query_cache.get_stats(),
            "metadata_cache": self.metadata_cache.get_stats(),
            "total_memory_usage": (
                self.embedding_cache.get_stats()["memory_usage"] +
                self.raptor_cache.get_stats()["memory_usage"] +
                self.query_cache.get_stats()["memory_usage"] +
                self.metadata_cache.get_stats()["memory_usage"]
            )
        }
    
    def cleanup_all(self) -> None:
        """Clean up all caches"""
        self.embedding_cache._cleanup_expired()
        self.raptor_cache._cleanup_expired()
        self.query_cache._cleanup_expired()
        self.metadata_cache._cleanup_expired()
    
    def clear_all(self) -> None:
        """Clear all caches"""
        self.embedding_cache.clear()
        self.raptor_cache.clear()
        self.query_cache.clear()
        self.metadata_cache.clear()
        logging.info("All caches cleared")

# Global cache manager instance
cache_manager = CacheManager()
