import os
from dotenv import load_dotenv

load_dotenv()

EMBEDDING_SERVICE_URL = os.getenv(
    "EMBEDDING_SERVICE_URL",
    "http://localhost:8000"
)
VECTOR_SERVICE_URL = os.getenv(
    "VECTOR_SERVICE_URL",
    "http://localhost:8003"
)
