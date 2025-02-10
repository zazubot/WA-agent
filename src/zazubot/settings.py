from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application configuration management using Pydantic Settings.
    
    Loads environment variables from .env file, with support for 
    API keys, model names, and memory management settings.
    """

    # Configuration for environment variable loading
    model_config = SettingsConfigDict(
        env_file=".env",  # Specify .env file location
        extra="ignore",  # Ignore undefined environment variables
        env_file_encoding="utf-8"  # Use UTF-8 encoding for .env file
    )

    # API Keys for various services
    GROQ_API_KEY: str  # API key for Groq language model service
    ELEVENLABS_API_KEY: str  # API key for ElevenLabs text-to-speech
    ELEVENLABS_VOICE_ID: str  # Specific voice identifier for TTS
    TOGETHER_API_KEY: str  # API key for Together AI services

    # Qdrant vector database configuration
    QDRANT_API_KEY: str | None  # Optional API key for Qdrant
    QDRANT_URL: str  # Base URL for Qdrant service
    QDRANT_PORT: str = "6333"  # Default Qdrant port
    QDRANT_HOST: str | None = None  # Optional host specification

    # Model name configurations
    TEXT_MODEL_NAME: str = "llama-3.3-70b-versatile"  # Primary text generation model
    SMALL_TEXT_MODEL_NAME: str = "gemma2-9b-it"  # Smaller text model
    STT_MODEL_NAME: str = "whisper-large-v3-turbo"  # Speech-to-text model
    TTS_MODEL_NAME: str = "eleven_flash_v2_5"  # Text-to-speech model
    TTI_MODEL_NAME: str = "black-forest-labs/FLUX.1-schnell-Free"  # Text-to-image model
    ITT_MODEL_NAME: str = "llama-3.2-90b-vision-preview"  # Image-to-text model

    # Memory and conversation management settings
    MEMORY_TOP_K: int = 3  # Number of top memories to retrieve
    ROUTER_MESSAGES_TO_ANALYZE: int = 3  # Messages to analyze for routing
    TOTAL_MESSAGES_SUMMARY_TRIGGER: int = 20  # Trigger point for conversation summary
    TOTAL_MESSAGES_AFTER_SUMMARY: int = 5  # Messages to keep after summary

    # Storage path for short-term memory database
    SHORT_TERM_MEMORY_DB_PATH: str = "/app/data/memory.db"


# Create a singleton settings instance
settings = Settings()