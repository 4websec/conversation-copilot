"""Configuration management for Conversation Copilot."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
else:
    # Try local .env
    load_dotenv()


class Config:
    # API Keys
    DEEPGRAM_API_KEY: str = os.getenv("DEEPGRAM_API_KEY", "")
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")

    # Server
    HOST: str = os.getenv("HOST", "127.0.0.1")
    PORT: int = int(os.getenv("PORT", "8765"))

    # LLM
    LLM_MODEL: str = os.getenv("LLM_MODEL", "claude-sonnet-4-5")
    LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "512"))
    LLM_TRIGGER_SILENCE_MS: int = int(os.getenv("LLM_TRIGGER_SILENCE_MS", "1500"))
    LLM_MIN_WORDS_TRIGGER: int = int(os.getenv("LLM_MIN_WORDS_TRIGGER", "3"))

    # Transcript
    TRANSCRIPT_BUFFER_MINUTES: int = int(os.getenv("TRANSCRIPT_BUFFER_MINUTES", "3"))

    # Speaker Calibration
    CALIBRATION_DURATION_SEC: int = int(os.getenv("CALIBRATION_DURATION_SEC", "5"))

    @classmethod
    def validate(cls):
        errors = []
        if not cls.DEEPGRAM_API_KEY or cls.DEEPGRAM_API_KEY == "your_deepgram_api_key_here":
            errors.append("DEEPGRAM_API_KEY not configured")
        if not cls.ANTHROPIC_API_KEY or cls.ANTHROPIC_API_KEY == "your_anthropic_api_key_here":
            errors.append("ANTHROPIC_API_KEY not configured")
        return errors
