from pathlib import Path

# Base paths
BASE_DIR    = Path(__file__).resolve().parent.parent
UPLOAD_DIR  = BASE_DIR / "uploads"
CHROMA_DIR  = BASE_DIR / "chroma_data"
DB_PATH     = BASE_DIR / "share_links.db"

# ChromaDB cosine distance = 1 - cosine_similarity  (range 0-1 for unit vectors).
# 0.3 = very confident  |  0.45 = good  |  0.6 = loose  |  >0.6 = likely mismatch
SIMILARITY_THRESHOLD: float = 0.25

# Maximum upload size per file in bytes (limit: 10 MB)
MAX_UPLOAD_BYTES: int = 10 * 1024 * 1024

# Maximum candidates returned by ChromaDB before deduplication
TOP_K: int = 50

# Shareable links
# How long a share link stays valid (seconds).  Expiry Time: 48 hours.
LINK_TTL_SECONDS: int = 48 * 60 * 60

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)
