"""Side-by-side crops of the same panorama region from several outputs (coordinates as fractions of width/height).

usage: python tools/compare_crops.py out.jpg x0 y0 x1 y1 img1.jpg [img2.jpg ...]
"""
import sys

import cv2
import numpy as np

out, x0, y0, x1, y1, *files = sys.argv[1:]
x0, y0, x1, y1 = map(float, (x0, y0, x1, y1))
tiles = []
for f in files:
    im = cv2.imread(f)
    h, w = im.shape[:2]
    c = im[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]
    c = cv2.resize(c, (1400, int(1400 * c.shape[0] / c.shape[1])), interpolation=cv2.INTER_AREA)
    label = "/".join(f.replace("\\", "/").split("/")[-2:])
    cv2.putText(c, label, (10, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 6)
    cv2.putText(c, label, (10, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    tiles.append(c)
cv2.imwrite(out, np.vstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 90])
