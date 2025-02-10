import os
from typing import Optional, List
from functools import lru_cache
from dataclasses import dataclass
from datetime import datetime

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

from zazu_bot.settings import settings


@dataclass
class Memory:
    """
    Represents a single memory entry stored in the vector database.
    
    Stores the text content, associated metadata, and optional similarity score.
    Provides convenient access to memory ID and timestamp.
    """

    text: str  # Actual content of the memory
    metadata: dict  # Additional information about the memory
    score: Optional[float] = None  # Similarity score when retrieved

    @property
    def id(self) -> Optional[str]:
        """Retrieve the unique identifier for this memory from metadata."""
        return self.metadata.get("id")

    @property
    def timestamp(self) -> Optional[datetime]:
        """
        Convert and return the timestamp from metadata.
        Converts ISO format string to datetime object if available.
        """
        ts = self.metadata.get("timestamp")
        return datetime.fromisoformat(ts) if ts else None


class VectorStore:
    """
    Manages long-term memory storage using Qdrant vector database.
    
    Implements a singleton pattern to ensure a single database connection.
    Uses sentence transformers for generating text embeddings.
    Supports storing, searching, and deduplicating memories.
    """

    # Configuration constants for vector storage
    REQUIRED_ENV_VARS = ["QDRANT_URL", "QDRANT_API_KEY"]  # Environment variables needed
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Sentence transformer model for embeddings
    COLLECTION_NAME = "long_term_memory"  # Name of the Qdrant collection
    SIMILARITY_THRESHOLD = 0.9  # Cosine similarity threshold for memory deduplication

    # Singleton pattern implementation variables
    _instance: Optional["VectorStore"] = None
    _initialized: bool = False

    def __new__(cls) -> "VectorStore":
        """
        Ensure only one instance of VectorStore is created.
        
        Returns the existing instance or creates a new one if not exists.
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """
        Initialize the vector store if not already initialized.
        
        Sets up sentence transformer model and Qdrant client.
        Validates required environment variables.
        """
        if not self._initialized:
            self._validate_env_vars()
            self.model = SentenceTransformer(self.EMBEDDING_MODEL)
            self.client = QdrantClient(
                url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY
            )
            self._initialized = True

    def _validate_env_vars(self) -> None:
        """
        Check that all required environment variables are set.
        
        Raises a ValueError if any required variables are missing.
        """
        missing_vars = [var for var in self.REQUIRED_ENV_VARS if not os.getenv(var)]
        if missing_vars:
            raise ValueError(
                f"Missing required environment variables: {', '.join(missing_vars)}"
            )

    def _collection_exists(self) -> bool:
        """
        Check if the memory collection already exists in Qdrant.
        
        Returns True if collection is present, False otherwise.
        """
        collections = self.client.get_collections().collections
        return any(col.name == self.COLLECTION_NAME for col in collections)

    def _create_collection(self) -> None:
        """
        Create a new vector collection in Qdrant for storing memories.
        
        Uses a sample embedding to determine vector size and configures cosine distance.
        """
        sample_embedding = self.model.encode("sample text")
        self.client.create_collection(
            collection_name=self.COLLECTION_NAME,
            vectors_config=VectorParams(
                size=len(sample_embedding),
                distance=Distance.COSINE,
            ),
        )

    def find_similar_memory(self, text: str) -> Optional[Memory]:
        """
        Search for a memory similar to the given text.
        
        Args:
            text: Input text to compare against existing memories
        
        Returns:
            Memory object if a highly similar memory exists, None otherwise
        """
        results = self.search_memories(text, k=1)
        if results and results[0].score >= self.SIMILARITY_THRESHOLD:
            return results[0]
        return None

    def store_memory(self, text: str, metadata: dict) -> None:
        """
        Store a new memory or update an existing similar memory.
        
        Checks for similar memories before storing to avoid duplicates.
        Generates an embedding for the text and stores it in Qdrant.
        
        Args:
            text: Content of the memory
            metadata: Additional information about the memory
        """
        if not self._collection_exists():
            self._create_collection()

        # Check if similar memory exists
        similar_memory = self.find_similar_memory(text)
        if similar_memory and similar_memory.id:
            metadata["id"] = similar_memory.id  # Keep same ID for update

        embedding = self.model.encode(text)
        point = PointStruct(
            id=metadata.get("id", hash(text)),
            vector=embedding.tolist(),
            payload={
                "text": text,
                **metadata,
            },
        )

        self.client.upsert(
            collection_name=self.COLLECTION_NAME,
            points=[point],
        )

    def search_memories(self, query: str, k: int = 5) -> List[Memory]:
        """
        Search for similar memories in the vector store.
        
        Generates an embedding for the query and finds similar memories.
        
        Args:
            query: Text to search for similar memories
            k: Maximum number of results to return
        
        Returns:
            List of Memory objects sorted by similarity
        """
        if not self._collection_exists():
            return []

        query_embedding = self.model.encode(query)
        results = self.client.search(
            collection_name=self.COLLECTION_NAME,
            query_vector=query_embedding.tolist(),
            limit=k,
        )

        return [
            Memory(
                text=hit.payload["text"],
                metadata={k: v for k, v in hit.payload.items() if k != "text"},
                score=hit.score,
            )
            for hit in results
        ]


@lru_cache
def get_vector_store() -> VectorStore:
    """
    Cached function to retrieve or create the VectorStore singleton.
    
    Uses lru_cache to memoize and efficiently return the same instance.
    
    Returns:
        Singleton VectorStore instance
    """
    return VectorStore()