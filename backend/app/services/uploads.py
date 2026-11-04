"""Safe handling of uploaded images: extension + magic-byte checks, size limits, temp-file cleanup."""
from __future__ import annotations

import contextlib
import hashlib
import os
import re
import tempfile
from pathlib import Path
from typing import AsyncIterator, Dict, Tuple

from fastapi import UploadFile

from ..errors import APIError
from ..settings import Settings

CHUNK = 1024 * 1024
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
JPEG_MAGIC = b"\xff\xd8\xff"


def safe_display_name(name: str | None) -> str:
    base = os.path.basename(name or "upload")
    base = re.sub(r"[^A-Za-z0-9._ -]", "_", base)[:120]
    return base or "upload"


def _detect_kind(head: bytes, ext: str) -> str:
    if head.startswith(PNG_MAGIC):
        return "png"
    if head.startswith(JPEG_MAGIC):
        return "jpeg"
    if len(head) >= 132 and head[128:132] == b"DICM":
        return "dicom"
    if ext in (".dcm", ".dicom"):
        return "dicom-no-preamble"  # verified with pydicom below
    return "unknown"


def _check_dimensions(path: Path, kind: str, settings: Settings) -> Tuple[int, int]:
    if kind in ("png", "jpeg"):
        from PIL import Image
        try:
            with Image.open(path) as im:  # lazy: reads header only
                w, h = im.size
                im.verify()  # structural check (CRC/truncation) without decoding pixels
        except Image.DecompressionBombError as exc:
            raise APIError(413, "image_too_large", "Image has too many pixels.") from exc
        except Exception as exc:  # noqa: BLE001
            raise APIError(415, "corrupt_image", "The image file is damaged or not a valid PNG/JPEG.") from exc
    else:
        import pydicom
        try:
            ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=kind == "dicom-no-preamble")
            h, w = int(ds.Rows), int(ds.Columns)
        except Exception as exc:  # noqa: BLE001
            raise APIError(415, "invalid_dicom", "The file could not be read as a DICOM image.") from exc
    if h * w > settings.max_image_pixels:
        raise APIError(413, "image_too_large", f"Image is {w}×{h} pixels; the limit is "
                       f"{settings.max_image_pixels:,} pixels.")
    if min(h, w) < settings.min_image_side:
        raise APIError(422, "image_too_small", f"Image is only {w}×{h} pixels.")
    return w, h


@contextlib.asynccontextmanager
async def saved_upload(file: UploadFile, settings: Settings) -> AsyncIterator[Dict]:
    """Validate and stream an upload to a private temp file; always deleted afterwards."""
    display = safe_display_name(file.filename)
    ext = Path(display).suffix.lower()
    if ext not in settings.allowed_extensions:
        raise APIError(415, "unsupported_file_type",
                       f"Unsupported file type '{ext or 'none'}'. Upload a PNG, JPEG or DICOM (.dcm) chest X-ray.",
                       {"allowed": list(settings.allowed_extensions)})
    fd, tmp_name = tempfile.mkstemp(prefix="rfrr_", suffix=ext)
    tmp = Path(tmp_name)
    sha = hashlib.sha256()
    size = 0
    head = b""
    try:
        with os.fdopen(fd, "wb") as out:
            while True:
                chunk = await file.read(CHUNK)
                if not chunk:
                    break
                size += len(chunk)
                if size > settings.max_upload_bytes:
                    raise APIError(413, "file_too_large",
                                   f"File exceeds the {settings.max_upload_mb:g} MB upload limit.")
                if len(head) < 132:
                    head += chunk[: 132 - len(head)]
                sha.update(chunk)
                out.write(chunk)
        if size == 0:
            raise APIError(422, "empty_file", "The uploaded file is empty.")
        kind = _detect_kind(head, ext)
        if kind == "unknown" or (kind in ("png", "jpeg") and ext in (".dcm", ".dicom")):
            raise APIError(415, "content_mismatch", "File content does not match a PNG, JPEG or DICOM image.")
        width, height = _check_dimensions(tmp, kind, settings)
        yield {"path": tmp, "display_name": display, "kind": kind, "bytes": size, "width": width, "height": height,
               "sha256": sha.hexdigest(), "reference": f"sha256:{sha.hexdigest()[:16]}"}
    finally:
        with contextlib.suppress(FileNotFoundError):
            tmp.unlink()
        await file.close()
