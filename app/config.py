import os

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "qwen3:8b")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
DB_PATH = os.getenv("SECOND_BRAIN_DB", "second_brain.db")
DMN_INTERVAL_MINUTES = int(os.getenv("DMN_INTERVAL_MINUTES", "30"))
DMN_ENABLED = os.getenv("DMN_ENABLED", "true").lower() == "true"
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")
