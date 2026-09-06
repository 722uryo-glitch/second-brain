import os


def _csv_env(name: str, default: str):
    return [x.strip() for x in os.getenv(name, default).split(",") if x.strip()]


OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "qwen3:8b")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

# AI Router: keep embeddings/local fallback on Ollama, offload public/bulk reasoning to UnoRouter.
UNOROUTER_ENABLED = os.getenv("UNOROUTER_ENABLED", "true").lower() == "true"
UNOROUTER_BASE_URL = os.getenv("UNOROUTER_BASE_URL", "https://api.unorouter.com/v1")
UNOROUTER_API_KEY = os.getenv("UNOROUTER_API_KEY", "").strip()
UNOROUTER_PRIVATE_CHAT = os.getenv("UNOROUTER_PRIVATE_CHAT", "false").lower() == "true"
UNOROUTER_TIMEOUT_SECONDS = int(os.getenv("UNOROUTER_TIMEOUT_SECONDS", "120"))
UNOROUTER_CHAT_MODELS = _csv_env(
    "UNOROUTER_CHAT_MODELS",
    "glm-5.3-flash-thinking:free,qwen3.8-flash-next:free,ling-3.0-flash-fin:free",
)
UNOROUTER_VERIFY_MODELS = _csv_env(
    "UNOROUTER_VERIFY_MODELS",
    "glm-5.3-flash-think-search:free,glm-5.3-flash-thinking:free,qwen3.8-flash-next:free",
)

# Executive layer: planner -> retrieval/research -> draft -> critic -> optional revision.
EXECUTIVE_ENABLED = os.getenv("EXECUTIVE_ENABLED", "true").lower() == "true"
EXECUTIVE_REVIEW_ENABLED = os.getenv("EXECUTIVE_REVIEW_ENABLED", "true").lower() == "true"
EXECUTIVE_MAX_RESEARCH_QUERIES = int(os.getenv("EXECUTIVE_MAX_RESEARCH_QUERIES", "3"))
EXECUTIVE_MAX_REVISIONS = int(os.getenv("EXECUTIVE_MAX_REVISIONS", "1"))
EXECUTIVE_HISTORY_TURNS = int(os.getenv("EXECUTIVE_HISTORY_TURNS", "8"))
EXECUTIVE_SIMPLE_MAX_TOKENS = int(os.getenv("EXECUTIVE_SIMPLE_MAX_TOKENS", "520"))
EXECUTIVE_LONG_MAX_TOKENS = int(os.getenv("EXECUTIVE_LONG_MAX_TOKENS", "2200"))

DB_PATH = os.getenv("SECOND_BRAIN_DB", "second_brain.db")
DMN_INTERVAL_MINUTES = int(os.getenv("DMN_INTERVAL_MINUTES", "30"))
DMN_ENABLED = os.getenv("DMN_ENABLED", "true").lower() == "true"
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")

EXTERNAL_COLLECTION_ENABLED = os.getenv("EXTERNAL_COLLECTION_ENABLED", "true").lower() == "true"
EXTERNAL_COLLECTION_INTERVAL_MINUTES = int(os.getenv("EXTERNAL_COLLECTION_INTERVAL_MINUTES", "15"))
EXTERNAL_ITEMS_PER_FEED = int(os.getenv("EXTERNAL_ITEMS_PER_FEED", "100"))
EXTERNAL_CONCURRENCY = int(os.getenv("EXTERNAL_CONCURRENCY", "16"))
DOCUMENT_FETCH_ENABLED = os.getenv("DOCUMENT_FETCH_ENABLED", "true").lower() == "true"
DOCUMENT_FETCH_BATCH_SIZE = int(os.getenv("DOCUMENT_FETCH_BATCH_SIZE", "120"))
DOCUMENT_FETCH_CONCURRENCY = int(os.getenv("DOCUMENT_FETCH_CONCURRENCY", "12"))
DOCUMENT_FETCH_INTERVAL_SECONDS = int(os.getenv("DOCUMENT_FETCH_INTERVAL_SECONDS", "10"))

FACTCHECK_ENABLED = os.getenv("FACTCHECK_ENABLED", "true").lower() == "true"
FACTCHECK_INTERVAL_SECONDS = int(os.getenv("FACTCHECK_INTERVAL_SECONDS", "5"))
FACTCHECK_BATCH_SIZE = int(os.getenv("FACTCHECK_BATCH_SIZE", "30"))
FACTCHECK_MAX_BATCH_SIZE = int(os.getenv("FACTCHECK_MAX_BATCH_SIZE", "80"))

GDELT_ENABLED = os.getenv("GDELT_ENABLED", "true").lower() == "true"
GITHUB_ENABLED = os.getenv("GITHUB_ENABLED", "true").lower() == "true"
X_ENABLED = os.getenv("X_ENABLED", "true").lower() == "true"
BLUESKY_ENABLED = os.getenv("BLUESKY_ENABLED", "true").lower() == "true"
REDDIT_ENABLED = os.getenv("REDDIT_ENABLED", "true").lower() == "true"
MASTODON_ENABLED = os.getenv("MASTODON_ENABLED", "true").lower() == "true"

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "").strip()
X_BEARER_TOKEN = os.getenv("X_BEARER_TOKEN", "").strip()

OBSIDIAN_ENABLED = os.getenv("OBSIDIAN_ENABLED", "true").lower() == "true"
OBSIDIAN_VAULT_PATH = os.getenv("OBSIDIAN_VAULT_PATH", "obsidian_vault")
OBSIDIAN_EXPORT_INTERVAL_MINUTES = int(os.getenv("OBSIDIAN_EXPORT_INTERVAL_MINUTES", "30"))
OBSIDIAN_MAX_CLAIMS = int(os.getenv("OBSIDIAN_MAX_CLAIMS", "300"))
OBSIDIAN_MAX_EXTERNAL = int(os.getenv("OBSIDIAN_MAX_EXTERNAL", "500"))
OBSIDIAN_MAX_MEMORIES = int(os.getenv("OBSIDIAN_MAX_MEMORIES", "200"))
OBSIDIAN_EXPORT_LIMIT = int(os.getenv("OBSIDIAN_EXPORT_LIMIT", str(max(OBSIDIAN_MAX_CLAIMS, OBSIDIAN_MAX_EXTERNAL, OBSIDIAN_MAX_MEMORIES))))

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
    ("CL","es-419","es"),("CO","es-419","es"),("PE","es-419","es"),("RU","ru","ru"),
]

GDELT_QUERIES = [
    "politics OR election OR government", "economy OR inflation OR markets", "war OR conflict OR military",
    "technology OR software OR semiconductor", "artificial intelligence OR machine learning", "cybersecurity OR hack OR vulnerability",
    "science OR research OR discovery", "health OR disease OR medicine", "climate OR environment OR energy",
    "space OR satellite OR astronomy", "business OR startup OR company", "culture OR media OR entertainment",
    "earthquake OR flood OR wildfire OR disaster", "law OR court OR regulation", "crypto OR blockchain OR bitcoin",
]

GITHUB_EVENT_PAGES = int(os.getenv("GITHUB_EVENT_PAGES", "5"))
GITHUB_SEARCH_QUERIES = [q.strip() for q in os.getenv(
    "GITHUB_SEARCH_QUERIES",
    "artificial intelligence,agent,llm,cybersecurity,open source,robotics,computer vision,developer tools"
).split(",") if q.strip()]

X_QUERIES = [q.strip() for q in os.getenv(
    "X_QUERIES",
    "AI,technology,cybersecurity,markets,politics,science,breaking news,war,climate,startup,open source"
).split(",") if q.strip()]

SOCIAL_QUERIES = [q.strip() for q in os.getenv(
    "SOCIAL_QUERIES",
    "AI,technology,cybersecurity,science,markets,politics,war,climate,open source,startup"
).split(",") if q.strip()]

MASTODON_INSTANCES = [x.strip().rstrip("/") for x in os.getenv(
    "MASTODON_INSTANCES",
    "https://mastodon.social,https://fosstodon.org,https://mstdn.social"
).split(",") if x.strip()]

PRIMARY_SOURCE_FEEDS = [
    ("NASA", "https://www.nasa.gov/rss/dyn/breaking_news.rss"),
    ("CISA Advisories", "https://www.cisa.gov/cybersecurity-advisories/all.xml"),
    ("US Federal Register", "https://www.federalregister.gov/documents/search.rss?conditions%5Border%5D=newest"),
    ("European Commission Press", "https://ec.europa.eu/commission/presscorner/api/rss?language=en"),
]

DEFAULT_EXTERNAL_FEEDS = [
    ("BBC World", "https://feeds.bbci.co.uk/news/world/rss.xml"),
    ("BBC Technology", "https://feeds.bbci.co.uk/news/technology/rss.xml"),
    ("arXiv AI", "https://export.arxiv.org/rss/cs.AI"),
    ("arXiv ML", "https://export.arxiv.org/rss/cs.LG"),
    ("arXiv Security", "https://export.arxiv.org/rss/cs.CR"),
    ("Hacker News", "https://hnrss.org/newest?points=30"),
]

_extra_feeds = os.getenv("SECOND_BRAIN_FEEDS", "").strip()
EXTERNAL_FEEDS = list(DEFAULT_EXTERNAL_FEEDS)
if _extra_feeds:
    for raw in _extra_feeds.split(","):
        url = raw.strip()
        if url:
            EXTERNAL_FEEDS.append((url, url))
