import os

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "qwen3:8b")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
DB_PATH = os.getenv("SECOND_BRAIN_DB", "second_brain.db")
DMN_INTERVAL_MINUTES = int(os.getenv("DMN_INTERVAL_MINUTES", "30"))
DMN_ENABLED = os.getenv("DMN_ENABLED", "true").lower() == "true"
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")

# Autonomous external information collection.
EXTERNAL_COLLECTION_ENABLED = os.getenv("EXTERNAL_COLLECTION_ENABLED", "true").lower() == "true"
EXTERNAL_COLLECTION_INTERVAL_MINUTES = int(os.getenv("EXTERNAL_COLLECTION_INTERVAL_MINUTES", "30"))
EXTERNAL_ITEMS_PER_FEED = int(os.getenv("EXTERNAL_ITEMS_PER_FEED", "20"))

# Comma-separated RSS/Atom feeds can be added without changing code.
# Defaults intentionally span general news, technology, research and developer news.
DEFAULT_EXTERNAL_FEEDS = [
    ("Google News Japan", "https://news.google.com/rss?hl=ja&gl=JP&ceid=JP:ja"),
    ("BBC World", "https://feeds.bbci.co.uk/news/world/rss.xml"),
    ("BBC Technology", "https://feeds.bbci.co.uk/news/technology/rss.xml"),
    ("arXiv AI", "https://export.arxiv.org/rss/cs.AI"),
    ("Hacker News", "https://hnrss.org/frontpage"),
]

_extra_feeds = os.getenv("SECOND_BRAIN_FEEDS", "").strip()
EXTERNAL_FEEDS = list(DEFAULT_EXTERNAL_FEEDS)
if _extra_feeds:
    for raw in _extra_feeds.split(","):
        url = raw.strip()
        if url:
            EXTERNAL_FEEDS.append((url, url))
