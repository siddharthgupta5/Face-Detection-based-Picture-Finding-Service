"""
face_engine.py
--------------
Wraps MTCNN (face detection) and InceptionResnetV1 (face embedding) from
facenet-pytorch.  Both models are loaded once at module import time so they
are reused across requests.

detect_and_embed(image: PIL.Image) -> list[EmbeddingResult]
    Returns one EmbeddingResult per detected face.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
from PIL import Image, ImageOps

# Lazy-import to keep startup fast when models aren't needed in tests
try:
    import torch
    from facenet_pytorch import MTCNN, InceptionResnetV1
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


@dataclass
class EmbeddingResult:
    """One detected face and its 512-d embedding."""
    face_index: int          # 0-based index within the photo
    embedding: List[float]   # 512-dimensional L2-normalised vector
    bbox: List[float]        # [x1, y1, x2, y2] in original image pixels


class FaceEngine:

    def __init__(self) -> None:
        if not _TORCH_AVAILABLE:
            raise RuntimeError(
                "facenet-pytorch and torch are required. "
                "Run: pip install facenet-pytorch torch torchvision"
            )
        self.device = torch.device("cpu")

        # MTCNN: detect faces and return aligned 160×160 crops
        self.detector = MTCNN(
            image_size=160,
            margin=20,
            min_face_size=20,
            keep_all=True,        # return ALL faces in the image
            device=self.device,
            post_process=True,    # normalise pixel values
        )

        # InceptionResnetV1: pretrained ArcFace / VGGFace2 embeddings
        self.embedder = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)

    def detect_and_embed(self, image: Image.Image) -> List[EmbeddingResult]:
        """
        Detect all faces in *image* and return their embeddings.

        Parameters
        ----------
        image : PIL.Image.Image
            Input image in any mode; RGB conversion is handled internally.

        Returns
        -------
        list[EmbeddingResult]
            Empty list if no faces are detected.
        """
        # Apply EXIF rotation first — phone photos store orientation in metadata
        img_rgb = ImageOps.exif_transpose(image).convert("RGB")

        # boxes: (N, 4) or None  |  probs: (N,) or None
        boxes, probs = self.detector.detect(img_rgb)

        if boxes is None or len(boxes) == 0:
            return []

        # Get aligned face crops as tensor (N, 3, 160, 160)
        face_tensors = self.detector(img_rgb)  
        # Re-use the already-oriented image for the crop pass
        if face_tensors is None:
            return []

        # facenet_pytorch returns a single tensor when keep_all=True
        if face_tensors.ndim == 3:
            # Single face — add batch dimension
            face_tensors = face_tensors.unsqueeze(0)

        results: List[EmbeddingResult] = []
        with torch.no_grad():
            embeddings = self.embedder(face_tensors.to(self.device))  # (N, 512)
            # L2-normalise (InceptionResnetV1 already does this, but be explicit)
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        for i, (emb, box) in enumerate(zip(embeddings, boxes)):
            results.append(
                EmbeddingResult(
                    face_index=i,
                    embedding=emb.cpu().numpy().tolist(),
                    bbox=[float(v) for v in box],
                )
            )

        return results


# Module created on first import
_engine: FaceEngine | None = None


def get_engine() -> FaceEngine:
    """Return the module-level FaceEngine, initialising it on first call."""
    global _engine
    if _engine is None:
        _engine = FaceEngine()
    return _engine


def detect_and_embed(image: Image.Image) -> List[EmbeddingResult]:
    """Convenience wrapper around ``get_engine().detect_and_embed``."""
    return get_engine().detect_and_embed(image)
