"""
main.py
"""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image


_VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".bmp", ".tiff"}

from app import config
from app.face_engine import detect_and_embed
from app.share_store import create_share_link, get_share_link, purge_expired
from app.vector_store import get_store


app = FastAPI(title="PhotoFinder", version="1.0.0")

# Serve uploaded photos at /uploads/<filename>
app.mount("/uploads", StaticFiles(directory=str(config.UPLOAD_DIR)), name="uploads")

templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))


def _save_upload(file: UploadFile) -> tuple[str, str]:
    """
    Save an uploaded file to UPLOAD_DIR with a UUID-prefixed filename.
    Enforces MAX_UPLOAD_BYTES file-size limit.

    Returns
    -------
    tuple[str, str]
        (photo_id, saved_filename)
    """
    contents = file.file.read()
    if len(contents) > config.MAX_UPLOAD_BYTES:
        mb = config.MAX_UPLOAD_BYTES // (1024 * 1024)
        raise ValueError(f"File exceeds maximum size of {mb} MB.")

    ext      = Path(file.filename or "photo.jpg").suffix.lower() or ".jpg"
    photo_id = str(uuid.uuid4())
    filename = f"{photo_id}{ext}"
    dest     = config.UPLOAD_DIR / filename
    dest.write_bytes(contents)
    return photo_id, filename


def _bbox_area(bbox: list) -> float:
    """Return pixel area of an [x1, y1, x2, y2] bounding box."""
    x1, y1, x2, y2 = bbox
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _list_photos() -> list[dict]:
    """Return metadata for every photo currently on disk."""
    photos = []
    for f in sorted(config.UPLOAD_DIR.iterdir()):
        if f.suffix.lower() in _VALID_EXTS:
            photos.append({"photo_id": f.stem, "filename": f.name})
    return photos


def _process_photo(photo_id: str, filename: str, original_filename: str) -> int:
    """Detect faces in a saved photo and index them.  Returns face count."""
    path  = config.UPLOAD_DIR / filename
    image = Image.open(path)
    faces = detect_and_embed(image)

    if faces:
        store = get_store()
        store.index_faces(
            photo_id=photo_id,
            original_filename=original_filename,
            embeddings=[f.embedding for f in faces],
        )
    return len(faces)


# Routes 

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Landing page — combined upload + search interface."""
    store       = get_store()
    total_faces = store.count()
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "total_faces": total_faces},
    )


@app.post("/upload")
async def upload_photos(files: List[UploadFile] = File(...)):
    """
    Photographer uploads one or more event photos.
    Each photo is saved, faces are detected, and embeddings are indexed.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    results = []
    loop    = asyncio.get_running_loop()

    for file in files:
        if not file.content_type or not file.content_type.startswith("image/"):
            results.append(
                {"filename": file.filename, "status": "skipped", "reason": "not an image"}
            )
            continue

        photo_id, saved_filename = _save_upload(file)

        # Run CPU-bound processing in a thread pool so the event loop is free
        try:
            face_count = await loop.run_in_executor(
                None, _process_photo, photo_id, saved_filename,
                file.filename or saved_filename
            )
            results.append(
                {
                    "filename":  file.filename,
                    "photo_id":  photo_id,
                    "status":    "indexed",
                    "faces":     face_count,
                }
            )
        except Exception as exc:
            results.append(
                {"filename": file.filename, "status": "error", "reason": str(exc)}
            )

    return JSONResponse({"uploaded": len(results), "results": results})


@app.post("/search", response_class=HTMLResponse)
async def search(request: Request, file: UploadFile = File(...)):
    """
    User uploads a selfie.
    Detect the (first) face, query ChromaDB, return a results page.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    loop = asyncio.get_running_loop()

    try:
        image = Image.open(file.file)
        image.load()  # Force decode now — surfaces truncated/corrupt JPEG errors early
    except Exception:
        return templates.TemplateResponse(
            "results.html",
            {
                "request":   request,
                "photo_ids": [],
                "filenames": [],
                "link_id":   None,
                "error":     "Could not read the uploaded image. "
                             "Please try a different file.",
            },
        )

    try:
        faces = await loop.run_in_executor(None, detect_and_embed, image)
    except Exception:
        return templates.TemplateResponse(
            "results.html",
            {
                "request":   request,
                "photo_ids": [],
                "filenames": [],
                "link_id":   None,
                "error":     "Face detection failed. Please try a clearer photo.",
            },
        )

    if not faces:
        return templates.TemplateResponse(
            "results.html",
            {
                "request":   request,
                "photo_ids": [],
                "filenames": [],
                "link_id":   None,
                "error":     "No face detected in the uploaded photo. "
                             "Please try a clearer selfie.",
            },
        )

    dominant = max(faces, key=lambda f: _bbox_area(f.bbox))
    query_emb = dominant.embedding
    store     = get_store()

    matched_ids: List[str] = await loop.run_in_executor(None, store.search, query_emb)

    link_id = None
    if matched_ids:
        link_id = create_share_link(matched_ids)
        await loop.run_in_executor(None, purge_expired)

    # Build display filenames (photo_id -> filename on disk)
    filenames = []
    for pid in matched_ids:
        # Find any file whose name starts with the photo_id
        candidates = list(config.UPLOAD_DIR.glob(f"{pid}.*"))
        filenames.append(candidates[0].name if candidates else "")

    return templates.TemplateResponse(
        "results.html",
        {
            "request":   request,
            "photo_ids": matched_ids,
            "filenames": filenames,
            "link_id":   link_id,
            "error":     None,
        },
    )


@app.get("/share/{link_id}", response_class=HTMLResponse)
async def share(request: Request, link_id: str):
    """Serve a shared gallery page.  Returns 410 if the link has expired."""
    photo_ids = get_share_link(link_id)

    if photo_ids is None:
        return templates.TemplateResponse(
            "share.html",
            {"request": request, "expired": True, "filenames": [], "link_id": link_id},
            status_code=410,
        )

    filenames = []
    for pid in photo_ids:
        candidates = list(config.UPLOAD_DIR.glob(f"{pid}.*"))
        filenames.append(candidates[0].name if candidates else "")

    return templates.TemplateResponse(
        "share.html",
        {
            "request":   request,
            "expired":   False,
            "photo_ids": photo_ids,
            "filenames": filenames,
            "link_id":   link_id,
        },
    )


@app.get("/health")
async def health():
    """Quick liveness check."""
    store = get_store()
    return {"status": "ok", "indexed_faces": store.count()}



@app.get("/admin", response_class=HTMLResponse)
async def admin(request: Request):
    """Photographer admin dashboard — shows all indexed photos with delete."""
    store  = get_store()
    photos = _list_photos()
    return templates.TemplateResponse(
        "admin.html",
        {
            "request":      request,
            "photos":       photos,
            "total_photos": len(photos),
            "total_faces":  store.count(),
        },
    )


@app.delete("/photos/{photo_id}")
async def delete_photo(photo_id: str):
    """
    Remove a photo from disk and its face vectors from ChromaDB.
    Safe to call if the photo_id does not exist — returns 404 in that case.
    """
    candidates = list(config.UPLOAD_DIR.glob(f"{photo_id}.*"))
    if not candidates:
        raise HTTPException(status_code=404, detail="Photo not found.")

    # Remove vectors first so the store stays consistent
    store = get_store()
    store.delete_photo(photo_id)

    # Remove file from disk
    for f in candidates:
        f.unlink(missing_ok=True)

    return {"deleted": photo_id}


@app.get("/api/photos")
async def list_photos():
    """JSON list of all photos currently on disk."""
    return {"photos": _list_photos(), "total": len(_list_photos())}
