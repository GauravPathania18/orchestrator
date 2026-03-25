"""
Centralized configuration management for RAPTOR services
"""

import os
import logging
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path

from .config import DEFAULT_MODEL

@dataclass
class ServiceConfig:
    """Configuration for individual services"""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    timeout: int = 30
    log_level: str = "INFO"

@dataclass
class DatabaseConfig:
    """Database configuration"""
    persist_dir: str = "./chroma_store"
    collection_name: str = "personal_llm"
    max_connections: int = 10
    connection_timeout: int = 30

@dataclass
class EmbeddingConfig:
    """Embedding service configuration"""
    model_name: str = "all-mpnet-base-v2"
    vector_dimension: int = 768
    batch_size: int = 32
    cache_size: int = 10000
    normalize: bool = True
    url: str = "http://localhost:8000/embed"

@dataclass
class RaptorConfig:
    """RAPTOR-specific configuration"""
    cluster_size: int = 4
    k_summary: int = 3
    k_chunks: int = 10
    top_k_final: int = 5
    min_confidence: float = 0.0
    max_distance: float = 1.5
    chunk_size: int = 300

@dataclass
class RerankerConfig:
    """Reranker configuration"""
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    batch_size: int = 16
    max_length: int = 512

@dataclass
class LLMConfig:
    """LLM service configuration"""
    url: str = "http://localhost:11434"
    model: str = DEFAULT_MODEL
    temperature: float = 0.7
    max_tokens: int = 2048

@dataclass
class SystemConfig:
    """Complete system configuration"""
    service: ServiceConfig = field(default_factory=ServiceConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    raptor: RaptorConfig = field(default_factory=RaptorConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    vector_store_url: str = "http://localhost:8001"
    
    # System-wide settings
    debug: bool = False
    environment: str = "development"
    log_format: str = "%(asctime)s - %(levelname)s - %(message)s"

class ConfigManager:
    """Centralized configuration manager"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or ".env"
        self.config = SystemConfig()
        self._load_environment()
        self._validate_config()
    
    def _load_environment(self):
        """Load configuration from environment variables"""
        
        # Service configuration
        self.config.service.host = os.getenv("HOST", self.config.service.host)
        self.config.service.port = int(os.getenv("PORT", str(self.config.service.port)))
        self.config.service.workers = int(os.getenv("WORKERS", str(self.config.service.workers)))
        self.config.service.timeout = int(os.getenv("TIMEOUT", str(self.config.service.timeout)))
        self.config.service.log_level = os.getenv("LOG_LEVEL", self.config.service.log_level)
        
        # Database configuration
        self.config.database.persist_dir = os.getenv("PERSIST_DIR", self.config.database.persist_dir)
        self.config.database.collection_name = os.getenv("COLLECTION_NAME", self.config.database.collection_name)
        self.config.database.max_connections = int(os.getenv("MAX_CONNECTIONS", str(self.config.database.max_connections)))
        
        # Embedding configuration
        self.config.embedding.model_name = os.getenv("EMBEDDING_MODEL", self.config.embedding.model_name)
        self.config.embedding.vector_dimension = int(os.getenv("VECTOR_DIMENSION", str(self.config.embedding.vector_dimension)))
        self.config.embedding.batch_size = int(os.getenv("EMBEDDING_BATCH_SIZE", str(self.config.embedding.batch_size)))
        self.config.embedding.cache_size = int(os.getenv("EMBEDDING_CACHE_SIZE", str(self.config.embedding.cache_size)))
        
        # RAPTOR configuration
        self.config.raptor.cluster_size = int(os.getenv("RAPTOR_CLUSTER_SIZE", str(self.config.raptor.cluster_size)))
        self.config.raptor.k_summary = int(os.getenv("RAPTOR_K_SUMMARY", str(self.config.raptor.k_summary)))
        self.config.raptor.k_chunks = int(os.getenv("RAPTOR_K_CHUNKS", str(self.config.raptor.k_chunks)))
        self.config.raptor.top_k_final = int(os.getenv("RAPTOR_TOP_K_FINAL", str(self.config.raptor.top_k_final)))
        
        # Reranker configuration
        self.config.reranker.model_name = os.getenv("RERANKER_MODEL", self.config.reranker.model_name)
        self.config.reranker.batch_size = int(os.getenv("RERANKER_BATCH_SIZE", str(self.config.reranker.batch_size)))
        
        # LLM configuration
        self.config.llm.url = os.getenv("OLLAMA_URL", self.config.llm.url)
        self.config.llm.model = os.getenv("LLM_MODEL", self.config.llm.model)
        self.config.llm.temperature = float(os.getenv("LLM_TEMPERATURE", str(self.config.llm.temperature)))
        
        # System configuration
        self.config.debug = os.getenv("DEBUG", "false").lower() in ("true", "1", "yes")
        self.config.environment = os.getenv("ENVIRONMENT", self.config.environment)
        
        # Service-specific URLs
        self.config.embedding.url = os.getenv("EMBEDDER_URL", "http://localhost:8000/embed")
        self.config.vector_store_url = os.getenv("VECTOR_STORE_URL", "http://localhost:8001")
    
    def _validate_config(self):
        """Validate configuration values"""
        
        # Validate ports
        if not (1 <= self.config.service.port <= 65535):
            raise ValueError(f"Invalid port number: {self.config.service.port}")
        
        # Validate directories
        persist_dir = Path(self.config.database.persist_dir)
        if not persist_dir.exists():
            logging.info(f"Creating persist directory: {persist_dir}")
            persist_dir.mkdir(parents=True, exist_ok=True)
        
        # Validate numeric values
        if self.config.embedding.vector_dimension <= 0:
            raise ValueError("Vector dimension must be positive")
        
        if self.config.raptor.cluster_size <= 0:
            raise ValueError("RAPTOR cluster size must be positive")
        
        if not (0.0 <= self.config.raptor.min_confidence <= 1.0):
            raise ValueError("Min confidence must be between 0.0 and 1.0")
        
        if not (0.0 <= self.config.raptor.max_distance <= 1.5):
            raise ValueError("Max distance must be between 0.0 and 1.5")
        
        if not (0.0 <= self.config.llm.temperature <= 2.0):
            raise ValueError("LLM temperature must be between 0.0 and 2.0")
    
    def get_service_config(self) -> ServiceConfig:
        """Get service configuration"""
        return self.config.service
    
    def get_database_config(self) -> DatabaseConfig:
        """Get database configuration"""
        return self.config.database
    
    def get_embedding_config(self) -> EmbeddingConfig:
        """Get embedding configuration"""
        return self.config.embedding
    
    def get_raptor_config(self) -> RaptorConfig:
        """Get RAPTOR configuration"""
        return self.config.raptor
    
    def get_reranker_config(self) -> RerankerConfig:
        """Get reranker configuration"""
        return self.config.reranker
    
    def get_llm_config(self) -> LLMConfig:
        """Get LLM configuration"""
        return self.config.llm
    
    def get_config_dict(self) -> Dict[str, Any]:
        """Get complete configuration as dictionary"""
        return {
            "service": {
                "host": self.config.service.host,
                "port": self.config.service.port,
                "workers": self.config.service.workers,
                "timeout": self.config.service.timeout,
                "log_level": self.config.service.log_level
            },
            "database": {
                "persist_dir": self.config.database.persist_dir,
                "collection_name": self.config.database.collection_name,
                "max_connections": self.config.database.max_connections
            },
            "embedding": {
                "model_name": self.config.embedding.model_name,
                "vector_dimension": self.config.embedding.vector_dimension,
                "batch_size": self.config.embedding.batch_size,
                "cache_size": self.config.embedding.cache_size,
                "normalize": self.config.embedding.normalize
            },
            "raptor": {
                "cluster_size": self.config.raptor.cluster_size,
                "k_summary": self.config.raptor.k_summary,
                "k_chunks": self.config.raptor.k_chunks,
                "top_k_final": self.config.raptor.top_k_final,
                "min_confidence": self.config.raptor.min_confidence,
                "max_distance": self.config.raptor.max_distance,
                "chunk_size": self.config.raptor.chunk_size
            },
            "reranker": {
                "model_name": self.config.reranker.model_name,
                "batch_size": self.config.reranker.batch_size,
                "max_length": self.config.reranker.max_length
            },
            "llm": {
                "url": self.config.llm.url,
                "model": self.config.llm.model,
                "temperature": self.config.llm.temperature,
                "max_tokens": self.config.llm.max_tokens
            },
            "system": {
                "debug": self.config.debug,
                "environment": self.config.environment,
                "log_format": self.config.log_format
            }
        }
    
    def update_config(self, updates: Dict[str, Any]):
        """Update configuration with new values"""
        for section, values in updates.items():
            if hasattr(self.config, section):
                section_config = getattr(self.config, section)
                for key, value in values.items():
                    if hasattr(section_config, key):
                        setattr(section_config, key, value)
                    else:
                        logging.warning(f"Unknown config key: {section}.{key}")
            else:
                logging.warning(f"Unknown config section: {section}")
        
        # Re-validate after update
        self._validate_config()
    
    def save_config_template(self, file_path: str = ".env.template"):
        """Save configuration template file"""
        template_content = f"""# RAPTOR RAG System Configuration Template

# Service Configuration
HOST={self.config.service.host}
PORT={self.config.service.port}
WORKERS={self.config.service.workers}
TIMEOUT={self.config.service.timeout}
LOG_LEVEL={self.config.service.log_level}

# Database Configuration
PERSIST_DIR={self.config.database.persist_dir}
COLLECTION_NAME={self.config.database.collection_name}
MAX_CONNECTIONS={self.config.database.max_connections}

# Embedding Configuration
EMBEDDING_MODEL={self.config.embedding.model_name}
VECTOR_DIMENSION={self.config.embedding.vector_dimension}
EMBEDDING_BATCH_SIZE={self.config.embedding.batch_size}
EMBEDDING_CACHE_SIZE={self.config.embedding.cache_size}

# RAPTOR Configuration
RAPTOR_CLUSTER_SIZE={self.config.raptor.cluster_size}
RAPTOR_K_SUMMARY={self.config.raptor.k_summary}
RAPTOR_K_CHUNKS={self.config.raptor.k_chunks}
RAPTOR_TOP_K_FINAL={self.config.raptor.top_k_final}

# Reranker Configuration
RERANKER_MODEL={self.config.reranker.model_name}
RERANKER_BATCH_SIZE={self.config.reranker.batch_size}

# LLM Configuration
OLLAMA_URL={self.config.llm.url}
LLM_MODEL={self.config.llm.model}
LLM_TEMPERATURE={self.config.llm.temperature}

# Service URLs
EMBEDDER_URL={self.config.embedding.url}
VECTOR_STORE_URL={self.config.vector_store_url}

# System Configuration
DEBUG={str(self.config.debug).lower()}
ENVIRONMENT={self.config.environment}
"""
        
        with open(file_path, 'w') as f:
            f.write(template_content)
        
        logging.info(f"Configuration template saved to {file_path}")

# Global configuration instance
config_manager = ConfigManager()
