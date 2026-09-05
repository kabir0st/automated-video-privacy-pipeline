"""Box geometry helpers. Boxes are float32 ``xyxy`` unless a name says
``cxcywh``. Pure numpy; imported by every stage, so nothing heavy here."""
from __future__ import annotations

import numpy as np


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between (N,4+) and (M,4+) xyxy boxes → (N, M) float32."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return (inter / np.maximum(union, 1e-9)).astype(np.float32)


def centres_inside(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(N, M) bool — a[i]'s centre lies inside b[j]."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), bool)
    acx = (a[:, 0] + a[:, 2]) / 2
    acy = (a[:, 1] + a[:, 3]) / 2
    return ((b[None, :, 0] <= acx[:, None]) & (acx[:, None] <= b[None, :, 2])
            & (b[None, :, 1] <= acy[:, None]) & (acy[:, None] <= b[None, :, 3]))


def nms(boxes: np.ndarray, iou_thr: float) -> np.ndarray:
    """Greedy score-ordered NMS over (K,5+) rows (score in column 4)."""
    if len(boxes) <= 1:
        return boxes
    order = np.argsort(-boxes[:, 4])
    boxes = boxes[order]
    ious = iou_matrix(boxes, boxes)
    suppressed = np.zeros(len(boxes), dtype=bool)
    keep: list[int] = []
    for i in range(len(boxes)):
        if suppressed[i]:
            continue
        keep.append(i)
        suppressed |= ious[i] > iou_thr
        suppressed[i] = True
    return boxes[keep]


def unrotate_boxes(boxes: np.ndarray, rot: int, fw: int, fh: int) -> np.ndarray:
    """Map xyxy boxes detected on a rotated copy back to the upright frame.
    ``rot`` is the cv2 rotation applied: 90 = ROTATE_90_CLOCKWISE, 270 =
    ROTATE_90_COUNTERCLOCKWISE, 180."""
    if len(boxes) == 0 or rot == 0:
        return boxes
    out = boxes.copy()
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    if rot == 90:      # (x, y) → (fh-1-y, x); inverse: (xr, yr) → (yr, fh-1-xr)
        out[:, 0], out[:, 2] = y1, y2
        out[:, 1], out[:, 3] = fh - 1 - x2, fh - 1 - x1
    elif rot == 270:   # (x, y) → (y, fw-1-x); inverse: (xr, yr) → (fw-1-yr, xr)
        out[:, 0], out[:, 2] = fw - 1 - y2, fw - 1 - y1
        out[:, 1], out[:, 3] = x1, x2
    elif rot == 180:
        out[:, 0], out[:, 2] = fw - 1 - x2, fw - 1 - x1
        out[:, 1], out[:, 3] = fh - 1 - y2, fh - 1 - y1
    return out


def grow_to_head(face: np.ndarray) -> np.ndarray:
    """Head-scale box from a face box: ×1.35 wide, ×1.55 tall, centre lifted
    0.12 face-heights for forehead and hair (~2.1× the area). A SCRFD/YOLO
    face box already spans brow to chin and cheek to cheek, so a head is only
    modestly larger; the old ×1.7/×1.9 (3.2× area) turned close-up faces into
    torso-sized claims. The renderer's mask padding adds the safety margin.
    Score column carried."""
    cx, cy = (face[0] + face[2]) / 2, (face[1] + face[3]) / 2
    w, h = face[2] - face[0], face[3] - face[1]
    pcx, pcy = cx, cy - 0.12 * h
    pw, ph = 1.35 * w, 1.55 * h
    out = np.array([pcx - pw / 2, pcy - ph / 2, pcx + pw / 2, pcy + ph / 2],
                   dtype=np.float32)
    if len(face) > 4:
        out = np.concatenate([out, np.asarray(face[4:], np.float32)])
    return out


def to_cxcywh(b: np.ndarray) -> np.ndarray:
    b = np.asarray(b, np.float32)
    return np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2,
                     b[2] - b[0], b[3] - b[1]], dtype=np.float32)


def to_xyxy(z: np.ndarray) -> np.ndarray:
    cx, cy, w, h = (float(v) for v in z[:4])
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2],
                    dtype=np.float32)


def rows_to_xyxy(z: np.ndarray) -> np.ndarray:
    """(T,4) cxcywh → (T,4) xyxy."""
    z = np.asarray(z, np.float32).reshape(-1, 4)
    return np.stack([z[:, 0] - z[:, 2] / 2, z[:, 1] - z[:, 3] / 2,
                     z[:, 0] + z[:, 2] / 2, z[:, 1] + z[:, 3] / 2], axis=1)


def clip_boxes(boxes: np.ndarray, fw: int, fh: int) -> np.ndarray:
    if len(boxes) == 0:
        return boxes
    boxes = boxes.copy()
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, fw - 1)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, fh - 1)
    return boxes
