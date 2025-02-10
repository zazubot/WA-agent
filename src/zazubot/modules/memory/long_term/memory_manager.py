import logging
import uuid
from datetime import datetime
from typing import List, Optional

from langchain_core.messages import BaseMessage
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

from zazu_bot.core.prompts import MEMORY_ANALYSIS_PROMPT
from zazu_bot.modules.memory.long_term.vector_store import get_vector_store
from zazu_bot.settings import settings


class MemoryAnalysis(BaseModel):
    """
    Represents the result of analyzing a message for memory storage.
    
    Determines if a message contains important information 
    and provides a formatted memory if applicable.
    """

    is_important: bool = Field(
        ...,
        description="Whether the message contains significant information to store",
    )
    formatted_memory: Optional[str] = Field(
        ..., description="Processed and structured memory content"
    )


class MemoryManager:
    """
    Manages long-term memory operations for the AI system.
    
    Handles memory extraction, storage, and retrieval using 
    a vector store and language model for analysis.
    """

    def __init__(self):
        """
        Initialize memory management components:
        - Vector store for persistent memory storage
        - Logger for tracking memory operations
        - Language model for memory analysis
        """
        self.vector_store = get_vector_store()
        self.logger = logging.getLogger(__name__)
        self.llm = ChatGroq(
            model=settings.SMALL_TEXT_MODEL_NAME,
            api_key=settings.GROQ_API_KEY,
            temperature=0.1,  # Low temperature for consistent analysis
            max_retries=2,
        ).with_structured_output(MemoryAnalysis)

    async def _analyze_memory(self, message: str) -> MemoryAnalysis:
        """
        Analyze a message to determine its importance for long-term memory.
        
        Uses a language model to evaluate and format the message.
        
        Args:
            message: Input message text to analyze
        
        Returns:
            MemoryAnalysis with importance and formatted memory
        """
        prompt = MEMORY_ANALYSIS_PROMPT.format(message=message)
        return await self.llm.ainvoke(prompt)

    async def extract_and_store_memories(self, message: BaseMessage) -> None:
        """
        Extract and store important memories from incoming messages.
        
        Skips non-human messages and checks for memory uniqueness.
        Stores unique, important memories in the vector store.
        
        Args:
            message: Message to potentially convert into a memory
        """
        if message.type != "human":
            return

        # Analyze the message for importance and formatting
        analysis = await self._analyze_memory(message.content)
        if analysis.is_important and analysis.formatted_memory:
            # Check if similar memory exists
            similar = self.vector_store.find_similar_memory(analysis.formatted_memory)
            if similar:
                # Skip storage if we already have a similar memory
                self.logger.info(
                    f"Similar memory already exists: '{analysis.formatted_memory}'"
                )
                return

            # Store new memory
            self.logger.info(f"Storing new memory: '{analysis.formatted_memory}'")
            self.vector_store.store_memory(
                text=analysis.formatted_memory,
                metadata={
                    "id": str(uuid.uuid4()),  # Generate unique identifier
                    "timestamp": datetime.now().isoformat(),  # Record creation time
                },
            )

    def get_relevant_memories(self, context: str) -> List[str]:
        """
        Retrieve memories most relevant to the given context.
        
        Searches vector store and logs retrieved memories.
        
        Args:
            context: Text to find relevant memories for
        
        Returns:
            List of memory texts sorted by relevance
        """
        memories = self.vector_store.search_memories(context, k=settings.MEMORY_TOP_K)
        if memories:
            for memory in memories:
                self.logger.debug(
                    f"Memory: '{memory.text}' (score: {memory.score:.2f})"
                )
        return [memory.text for memory in memories]

    def format_memories_for_prompt(self, memories: List[str]) -> str:
        """
        Convert memory list into a formatted string for prompting.
        
        Args:
            memories: List of memory texts
        
        Returns:
            Memories formatted as bullet-point list, or empty string
        """
        if not memories:
            return ""
        return "\n".join(f"- {memory}" for memory in memories)


def get_memory_manager() -> MemoryManager:
    """
    Singleton-like function to retrieve Memory Manager instance.
    
    Returns:
        MemoryManager for handling long-term memory operations
    """
    return MemoryManager()