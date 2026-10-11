"""Image file I/O that works with non-ASCII Windows paths (cv2.imread/imwrite do not, e.g. 'æøå')."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def imread(path: Path | str, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, flags)
    if img is None:
        raise IOError(f"Could not read the image: {path}")
    return img


def imwrite(path: Path | str, img: np.ndarray, params: list[int] | None = None) -> None:
    ext = Path(path).suffix or ".jpg"
    ok, buf = cv2.imencode(ext, img, params or [])
    if not ok:
        raise IOError(f"Could not save the image: {path}")
    buf.tofile(str(path))
