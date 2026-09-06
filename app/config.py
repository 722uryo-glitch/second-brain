import os

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "qwen3:8b")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
DB_PATH = os.getenv("SECOND_BRAIN_DB", "second_brain.db")
DMN_INTERVAL_MINUTES = int(os.getenv("DMN_INTERVAL_MINUTES", "30"))
DMN_ENABLED = os.getenv("DMN_ENABLED", "true").lower() == "true"
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")

# V1 global intelligence collector
EXTERNAL_COLLECTION_ENABLED = os.getenv("EXTERNAL_COLLECTION_ENABLED", "true").lower() == "true"
EXTERNAL_COLLECTION_INTERVAL_MINUTES = int(os.getenv("EXTERNAL_COLLECTION_INTERVAL_MINUTES", "15"))
EXTERNAL_ITEMS_PER_FEED = int(os.getenv("EXTERNAL_ITEMS_PER_FEED", "100"))
EXTERNAL_CONCURRENCY = int(os.getenv("EXTERNAL_CONCURRENCY", "12"))
FACTCHECK_ENABLED = os.getenv("FACTCHECK_ENABLED", "true").lower() == "true"
FACTCHECK_INTERVAL_SECONDS = int(os.getenv("FACTCHECK_INTERVAL_SECONDS", "20"))
FACTCHECK_BATCH_SIZE = int(os.getenv("FACTCHECK_BATCH_SIZE", "20"))
GDELT_ENABLED = os.getenv("GDELT_ENABLED", "true").lower() == "true"
GITHUB_ENABLED = os.getenv("GITHUB_ENABLED", "true").lower() == "true"
X_ENABLED = os.getenv("X_ENABLED", "true").lower() == "true"
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "").strip()
X_BEARER_TOKEN = os.getenv("X_BEARER_TOKEN", "").strip()

# Google News country/language editions. Failures are tolerated per-locale.
GLOBAL_NEWS_LOCALES = [
    ("US","en-US","en"),("GB","en-GB","en"),("CA","en-CA","en"),("CA","fr-CA","fr"),
    ("AU","en-AU","en"),("NZ","en-NZ","en"),("IE","en-IE","en"),("IN","en-IN","en"),
    ("IN","hi","hi"),("JP","ja","ja"),("KR","ko","ko"),("TW","zh-TW","zh"),("HK","zh-TW","zh"),
    ("SG","en-SG","en"),("MY","en-MY","en"),("ID","id","id"),("TH","th","th"),("VN","vi","vi"),
    ("PH","en-PH","en"),("PK","en-PK","en"),("BD","bn","bn"),("LK","en-LK","en"),
    ("DE","de","de"),("FR","fr","fr"),("ES","es","es"),("IT","it","it"),("PT","pt-PT","pt"),
    ("NL","nl","nl"),("BE","nl","nl"),("BE","fr","fr"),("CH","de","de"),("CH","fr","fr"),
    ("AT","de","de"),("SE","sv","sv"),("NO","no","no"),("DK","da","da"),("FI","fi","fi"),
    ("PL","pl","pl"),("CZ","cs","cs"),("SK","sk","sk"),("HU","hu","hu"),("RO","ro","ro"),
    ("BG","bg","bg"),("GR","el","el"),("UA","uk","uk"),("TR","tr","tr"),("IL","he","he"),
    ("AE","ar","ar"),("SA","ar","ar"),("EG","ar","ar"),("ZA","en-ZA","en"),("NG","en-NG","en"),
    ("KE","en-KE","en"),("BR","pt-BR","pt"),("MX","es-419","es"),("AR","es-419","es"),
    ("CL","es-419","es"),("CO","es-419","es"),("PE","es-419","es"),
]

# Broad GDELT query matrix. Each query can return up to 250 fresh articles.
GDELT_QUERIES = [
    "politics OR election OR government", "economy OR inflation OR markets", "war OR conflict OR military",
    "technology OR software OR semiconductor", "artificial intelligence OR machine learning", "cybersecurity OR hack OR vulnerability",
    "science OR research OR discovery", "health OR disease OR medicine", "climate OR environment OR energy",
    "space OR satellite OR astronomy", "business OR startup OR company", "culture OR media OR entertainment",
]

GITHUB_EVENT_PAGES = int(os.getenv("GITHUB_EVENT_PAGES", "5"))
X_QUERIES = [q.strip() for q in os.getenv(
    "X_QUERIES",
    "AI,technology,cybersecurity,markets,politics,science,breaking news,war,climate,startup,open source"
).split(",") if q.strip()]

DEFAULT_EXTERNAL_FEEDS = [
    ("BBC World", "https://feeds.bbci.co.uk/news/world/rss.xml"),
    ("BBC Technology", "https://feeds.bbci.co.uk/news/technology/rss.xml"),
    ("arXiv AI", "https://export.arxiv.org/rss/cs.AI"),
    ("arXiv ML", "https://export.arxiv.org/rss/cs.LG"),
    ("Hacker News", "https://hnrss.org/newest?points=50"),
]
_extra_feeds = os.getenv("SECOND_BRAIN_FEEDS", "").strip()
EXTERNAL_FEEDS = list(DEFAULT_EXTERNAL_FEEDS)
if _extra_feeds:
    for raw in _extra_feeds.split(","):
        url = raw.strip()
        if url:
            EXTERNAL_FEEDS.append((url, url))
