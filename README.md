# PhotoFinder

AI-powered facial recognition system for sorting and retrieving event photos.
Photographers upload all event photos; guests upload a selfie and instantly receive every photo they appear in — including group shots — along with a shareable gallery link.

---

## How It Works

1. **Photographer uploads photos** → MTCNN detects every face in every photo → each face is converted to a 512-dimensional embedding and stored in ChromaDB
2. **Guest uploads a selfie** → the dominant face is embedded → ChromaDB finds all similar face vectors across every indexed photo → results are deduplicated by photo
3. **Shareable link** → a UUID link is generated pointing to the matched gallery, valid for 48 hours

---

## Tech Stack

| Layer | Technology |
|---|---|
| Web framework | FastAPI + Uvicorn |
| Face detection | MTCNN (facenet-pytorch) |
| Face embedding | InceptionResnetV1 — 512-dim, VGGFace2 pretrained |
| Vector search | ChromaDB (embedded, cosine similarity) |
| Link storage | SQLite (Python stdlib) |
| Frontend | Vanilla HTML / CSS / JS (no build step) |

---

## Requirements

- Python 3.10 or higher
- No external services, databases, or Docker required

---

## Setup

### 1. Clone or download the project

```bash
git clone <your-repo-url>
cd photo_finder
```

### 2. Create and activate a virtual environment

```bash
# Create
python -m venv myenv

# Activate — Windows (PowerShell)
myenv\Scripts\Activate.ps1

# Activate — Windows (Command Prompt)
myenv\Scripts\activate.bat

# Activate — macOS / Linux
source myenv/bin/activate
```

### 3. Install PyTorch

Install PyTorch first (CPU build — no GPU required):

```bash
pip install numpy torch torchvision
```

> This step downloads ~800 MB. Allow a few minutes depending on your connection.

### 4. Install remaining dependencies

```bash
pip install facenet-pytorch==2.6.0 --no-deps
pip install -r requirements.txt
```

> `facenet-pytorch` is installed with `--no-deps` to skip its outdated numpy version constraint. Everything works correctly with numpy 2.x.

---

## Running the Application

```bash
python run.py
```

> **Note:** The first time the server starts, PyTorch will download the InceptionResnetV1 model weights (~100 MB) from the internet. This happens automatically and only once — subsequent starts are immediate. **Wait up to 60 seconds** for the server to become ready on first launch.

Once ready, you will see:

```
  PhotoFinder running at  http://127.0.0.1:8000

INFO:     Uvicorn running at http://127.0.0.1:8000 (Press CTRL+C to quit)
```

Open **http://127.0.0.1:8000** in your browser.

### Optional flags

```bash
# Custom host and port
python run.py --host 0.0.0.0 --port 8080

# Auto-reload on code changes (development)
python run.py --reload
```

---

## Pages & Endpoints

| URL | Description |
|---|---|
| `http://127.0.0.1:8000/` | Main page — upload event photos or search with a selfie |
| `http://127.0.0.1:8000/admin` | Admin dashboard — view all indexed photos, delete individual photos |
| `http://127.0.0.1:8000/share/{id}` | Shared gallery link (expires 48 hours after creation) |
| `http://127.0.0.1:8000/health` | JSON health check — shows number of indexed face vectors |
| `http://127.0.0.1:8000/api/photos` | JSON list of all photos on disk |
| `http://127.0.0.1:8000/docs` | Auto-generated OpenAPI / Swagger UI |

---

## Project Structure

```
photo_finder/
├── app/
│   ├── config.py          # Paths, thresholds, TTL settings
│   ├── face_engine.py     # MTCNN face detection + InceptionResnetV1 embeddings
│   ├── vector_store.py    # ChromaDB wrapper (index + cosine search)
│   ├── share_store.py     # SQLite shareable link store with TTL
│   ├── main.py            # FastAPI routes
│   └── templates/
│       ├── index.html     # Landing page
│       ├── results.html   # Search results + share button
│       ├── share.html     # Shared gallery / expired link page
│       └── admin.html     # Admin dashboard
├── uploads/               # Stored event photos (created automatically)
├── chroma_data/           # ChromaDB vector index (created automatically)
├── requirements.txt
├── run.py                 # Entry point
└── .gitignore
```

---

## Configuration

All tuneable settings are in `app/config.py`:

| Setting | Default | Description |
|---|---|---|
| `SIMILARITY_THRESHOLD` | `0.25` | Cosine distance cutoff — lower is stricter. Raise to `0.45`–`0.5` if too few results appear |
| `TOP_K` | `50` | Max face vectors queried from ChromaDB before deduplication |
| `LINK_TTL_SECONDS` | `172800` (48 h) | How long a shareable link remains valid |
| `MAX_UPLOAD_BYTES` | `15728640` (15 MB) | Maximum file size per uploaded photo |

---

## Notes

- Uploaded photos are stored in `uploads/` and served directly by FastAPI.
- The face index (`chroma_data/`) and the share-link database (`share_links.db`) are local files — back them up if you want to preserve indexed data between sessions.
- Processing speed depends on CPU. A modern laptop indexes roughly 1–3 photos per second. Searching with a selfie takes 1–2 seconds.
- EXIF rotation is automatically applied before face detection, so portrait photos from phones work correctly.
