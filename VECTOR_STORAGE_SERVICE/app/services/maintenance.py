import os
import shutil
import logging
from datetime import datetime
from .vector_store import get_vector_store
from .embedder import VECTOR_DIMENSION
from ..config import PERSIST_DIR

def backup_collection():
    """
    Creates a backup of the entire ChromaDB persistence directory.
    """
    try:
        if not os.path.exists(PERSIST_DIR):
            logging.warning(f"Persistence directory {PERSIST_DIR} does not exist. Nothing to backup.")
            return False
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"{PERSIST_DIR}_backup_{timestamp}"
        
        shutil.copytree(PERSIST_DIR, backup_path)
        logging.info(f"✅ Backup created at {backup_path}")
        return True
    except Exception as e:
        logging.error(f"❌ Backup failed: {e}")
        return False

def get_db_stats():
    """
    Returns basic statistics about the vector store.
    """
    try:
        vs = get_vector_store(VECTOR_DIMENSION)
        count = vs.collection.count()
        return {
            "count": count,
            "persist_dir": PERSIST_DIR,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logging.error(f"❌ Failed to get DB stats: {e}")
        return None
