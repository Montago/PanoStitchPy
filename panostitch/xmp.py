"""Write equirectangular JPEGs with Google Photo Sphere (GPano) XMP metadata, so that Facebook, Google Photos,
Kuula, VR viewers etc. recognise and display them as 360 panoramas."""
from __future__ import annotations

import struct
from pathlib import Path

import cv2
import numpy as np

_XMP_HEADER = b"http://ns.adobe.com/xap/1.0/\x00"


def full_sphere_gpano(width: int, height: int, heading_deg: float = 0.0) -> dict:
    return {"full_w": width, "full_h": height, "crop_w": width, "crop_h": height, "left": 0, "top": 0,
            "heading": heading_deg}


def gpano_xmp(g: dict) -> bytes:
    """g: full_w, full_h (the whole 360x180 sphere), crop_w, crop_h, left, top (where this image sits in it) and
    heading (compass heading of the centre of the full sphere)."""
    return (
        '<?xpacket begin="﻿" id="W5M0MpCehiHzreSzNTczkc9d"?>'
        '<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
        '<rdf:Description rdf:about="" xmlns:GPano="http://ns.google.com/photos/1.0/panorama/"'
        ' GPano:ProjectionType="equirectangular" GPano:UsePanoramaViewer="True"'
        f' GPano:FullPanoWidthPixels="{g["full_w"]}" GPano:FullPanoHeightPixels="{g["full_h"]}"'
        f' GPano:CroppedAreaImageWidthPixels="{g["crop_w"]}" GPano:CroppedAreaImageHeightPixels="{g["crop_h"]}"'
        f' GPano:CroppedAreaLeftPixels="{g["left"]}" GPano:CroppedAreaTopPixels="{g["top"]}"'
        f' GPano:PoseHeadingDegrees="{g["heading"] % 360:.1f}"'
        ' GPano:StitchingSoftware="PanoStichPy"/>'
        '</rdf:RDF></x:xmpmeta><?xpacket end="w"?>'
    ).encode("utf-8")


def insert_xmp(jpeg: bytes, xmp: bytes) -> bytes:
    """Insert an XMP APP1 segment after SOI (and after a JFIF APP0 segment, which must stay first)."""
    if jpeg[:2] != b"\xff\xd8":
        raise ValueError("not a JPEG")
    payload = _XMP_HEADER + xmp
    if len(payload) + 2 > 0xFFFF:
        raise ValueError("XMP too large for a single APP1 segment")
    segment = b"\xff\xe1" + struct.pack(">H", len(payload) + 2) + payload
    pos = 2
    if jpeg[2:4] == b"\xff\xe0":
        pos = 4 + struct.unpack(">H", jpeg[4:6])[0]
    return jpeg[:pos] + segment + jpeg[pos:]


def write_pano_jpeg(path: Path | str, img: np.ndarray, gpano: dict | None, quality: int = 93) -> None:
    """Write a JPEG, with GPano metadata when given (None: an ordinary flat image)."""
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise IOError(f"Could not encode JPEG: {path}")
    data = buf.tobytes()
    Path(path).write_bytes(insert_xmp(data, gpano_xmp(gpano)) if gpano else data)


def read_gpano(xmp: dict) -> dict | None:
    """GPano fields from read_xmp() output, or None if the image is not marked as a panorama."""
    try:
        return {"full_w": int(xmp["GPano:FullPanoWidthPixels"]), "full_h": int(xmp["GPano:FullPanoHeightPixels"]),
                "crop_w": int(xmp["GPano:CroppedAreaImageWidthPixels"]),
                "crop_h": int(xmp["GPano:CroppedAreaImageHeightPixels"]),
                "left": int(xmp.get("GPano:CroppedAreaLeftPixels", 0)),
                "top": int(xmp.get("GPano:CroppedAreaTopPixels", 0)),
                "heading": float(xmp.get("GPano:PoseHeadingDegrees", 0))}
    except (KeyError, ValueError):
        return None
