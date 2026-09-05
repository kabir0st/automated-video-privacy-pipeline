"""Single-model detection front-end: body + head + face + part boxes in one pass.

One ONNX detector from the PINTO model zoo whose classes include whole
*heads* alongside faces, bodies and (on the default spec) facial parts and
hands/feet. The head class is trained on all 360° head orientations (back of
head, profile, top-down), which is real recall the old pose/face ensemble
kept missing on lying-down and occluded subjects — but a head/face-class
detection is *evidence*, not a blur target by itself: skin and fabric read as
a head or face often enough on this footage that a lone detection here,
however confident, is not trustworthy on its own. libs/evidence.py is what
turns these boxes (plus libs/pose.py's anatomical anchors and libs/scrfd.py's
witness faces) into gated face candidates; nothing in this module may spawn a
blur.

Two interchangeable model specs are known:

  * ``wholebody17`` (default) — 457_YOLOv9-Wholebody17, GPLv3. Body/Head/Face
    plus eye/nose/mouth/ear (part-consensus evidence) and hand/foot
    (negative-veto evidence) among its 17 classes.
  * ``yolox_bhhf`` — 434_YOLOX-Body-Head-Hand-Face, Apache-2.0. Body/Head/Hand
    /Face, 4 classes — hand is available as negative-veto evidence but there
    are no eye/nose/mouth/ear/foot classes to draw on. The escape hatch if the
    default underperforms or GPL is unwanted, at the cost of weaker
    part-consensus evidence.

Both use PINTO's post-processed exports: NMS, BGR handling and normalisation
are embedded in the graph. Input is raw BGR float32 ``1×3×H×W``; output is one
``[N, 7]`` tensor of ``[batchno, classid, score, x1, y1, x2, y2]`` with
coordinates in model-input pixel space.

DirectML rules (learned the hard way in earlier revisions): destroying an ORT
session corrupts the DirectML provider's device state and the next inference
dies with a native access violation, and DirectML validates graph shapes
strictly. Hence: the session is created lazily, exactly once, and never
destroyed; the graph input is shape-pinned to the spec's fixed resolution
before the session is built. The active spec is read from the environment once
at import — there is deliberately no runtime model switching.

Model file resolution order (first hit wins):
  1. ``AVPP_DETECTOR_ONNX`` — explicit path to a local ``.onnx``;
  2. the frozen bundle (``sys._MEIPASS/models/<member>``, PyInstaller);
  3. ``~/.cache/avpp/detector/<member>``.
If none exists, :func:`download_model` fetches the spec's archive (PINTO ships
per-size ``.tar.gz`` bundles on a public S3), extracts the single member we
need into the cache, and deletes the archive. ``AVPP_DETECTOR_URL`` may point
at either a direct ``.onnx`` or a ``.tar.gz`` mirror.
"""

from __future__ import annotations

import os
import sys
import tarfile
import tempfile
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

_CACHE_DIR = Path.home() / ".cache" / "avpp" / "detector"

# Score floors applied at parse time. Deliberately low for heads/faces: the
# BYTE association stage in libs/head_tracker.py *needs* low-score detections
# to sustain tracks through occlusion — the real gates live in the tracker
# (and, for faces, in libs/evidence.py's acceptance gate).
HEAD_FLOOR = 0.05
FACE_FLOOR = 0.05
BODY_FLOOR = 0.35
# Facial parts (eye/nose/mouth/ear) are consensus evidence only — never a box
# on their own — so the floor stays permissive; the gate needs "is there an
# eye here at all", not a confident eye detection. Negatives (hand/foot) veto
# a face candidate, so a false veto is costly and gets a stricter floor.
PART_FLOOR = 0.20
NEG_FLOOR = 0.35

# IoU above which two boxes (same class, different rotation passes) are the
# same detection.
_MERGE_IOU = 0.55
_WITNESS_MATCH_IOU = 0.20    # boxes from two detectors describe the same face

# Rotated-pass acceptance. Rotation passes exist purely as recall assist for
# sideways heads, and the model hallucinates on rotated scenes (a rotated
# upright scene can read as one giant "head" spanning the frame — or, worse,
# a moderate bed-sized one that no absolute size cap can distinguish from a
# real head). A genuinely sideways head becomes *upright* in the rotated view
# and scores high, so a stricter floor costs no real recall; the containment
# check kills the giant-hallucination case when an upright head exists, and
# the absolute area cap kills frame-spanning fakes outright.
#
# The witness rule closes the remaining hole (moderate-size fakes on frames
# the upright pass read as empty): the head class is trained on all 360°
# orientations, so a real sideways head essentially never leaves the upright
# pass *completely* blind at its location — at the 0.05 parse floor a weak
# head/face registers, or the body containing it does. A rotated-only box
# with zero upright corroboration must clear a much higher floor to survive.
_ROT_SCORE_FLOOR = 0.50    # corroborated by upright evidence at the same spot
_ROT_LONE_FLOOR = 0.75     # no upright evidence at all → presumed fake
_ROT_WITNESS_IOU = 0.10   # min IoU for a weak upright head/face to corroborate
_ROT_CONTAIN_AREA = 0.5   # rotated head is fake if it swallows an upright one
_ROT_MAX_AREA_FRAC = 0.20  # rotated-only head bigger than this × frame is fake


# Part-class enum for Detections.parts[:, 5] ("cls" column, positive/
# consensus evidence). Ear is diagnostic/overlay only — not consulted by the
# gate, since an ear alone says little about where the face is.
PART_EYE, PART_NOSE, PART_MOUTH, PART_EAR = range(4)
# Negative-class enum for Detections.negatives[:, 5] (veto evidence).
NEG_HAND, NEG_FOOT = range(2)


@dataclass(frozen=True)
class DetectorSpec:
    name: str
    url: str                        # .tar.gz archive or direct .onnx
    member: str                     # file inside the archive ("" for direct)
    input_hw: tuple[int, int]       # (H, W) — pinned into the graph for DML
    body_ids: tuple[int, ...]
    head_ids: tuple[int, ...]
    face_ids: tuple[int, ...]
    # Optional finer-grained classes. Empty tuple = the spec's export has no
    # such class; the gate degrades gracefully (fewer consensus/veto sources,
    # never a crash — see libs/evidence.py).
    eye_ids: tuple[int, ...] = ()
    nose_ids: tuple[int, ...] = ()
    mouth_ids: tuple[int, ...] = ()
    ear_ids: tuple[int, ...] = ()
    hand_ids: tuple[int, ...] = ()
    foot_ids: tuple[int, ...] = ()
    # [N,7] output columns — identical for every known PINTO post export, kept
    # explicit so a future export with a different layout is a spec edit.
    col_cls: int = 1
    col_score: int = 2
    col_box: tuple[int, int, int, int] = (3, 4, 5, 6)


_S3 = "https://s3.ap-northeast-2.wasabisys.com/pinto-model-zoo"

SPECS: dict[str, DetectorSpec] = {
    "wholebody17": DetectorSpec(
        name="wholebody17",
        url=f"{_S3}/457_YOLOv9-Wholebody17/resources_s.tar.gz",
        member="yolov9_s_wholebody17_post_0100_1x3x640x640.onnx",
        input_hw=(640, 640),
        body_ids=(0,),
        head_ids=(7,),
        face_ids=(8,),
        eye_ids=(9,),
        nose_ids=(10,),
        mouth_ids=(11,),
        ear_ids=(12,),
        hand_ids=(13, 14, 15),
        foot_ids=(16,),
    ),
    "yolox_bhhf": DetectorSpec(
        name="yolox_bhhf",
        url=f"{_S3}/434_YOLOX-Body-Head-Hand-Face/resources.tar.gz",
        member="yolox_s_body_head_hand_face_post_0299_0.4983_1x3x640x640.onnx",
        input_hw=(640, 640),
        body_ids=(0,),
        head_ids=(1,),
        face_ids=(3,),
        hand_ids=(2,),
    ),
}


def active_spec() -> DetectorSpec:
    name = os.environ.get("AVPP_DETECTOR", "wholebody17").strip().lower()
    spec = SPECS.get(name, SPECS["wholebody17"])
    url = os.environ.get("AVPP_DETECTOR_URL")
    if url:
        member = spec.member if url.endswith((".tar.gz", ".tgz")) \
            else os.path.basename(url)
        spec = DetectorSpec(**{**spec.__dict__, "url": url, "member": member})
    return spec


def _member_name(spec: DetectorSpec) -> str:
    return spec.member or os.path.basename(spec.url)


def default_cache(spec: Optional[DetectorSpec] = None) -> Path:
    spec = spec or active_spec()
    return _CACHE_DIR / _member_name(spec)


def model_path(spec: Optional[DetectorSpec] = None) -> Optional[Path]:
    """First existing model file per the resolution order in the module doc."""
    spec = spec or active_spec()
    env = os.environ.get("AVPP_DETECTOR_ONNX")
    if env and Path(env).is_file():
        return Path(env)
    bundle = getattr(sys, "_MEIPASS", None)
    if bundle:
        p = Path(bundle) / "models" / _member_name(spec)
        if p.is_file():
            return p
    cache = default_cache(spec)
    if cache.is_file():
        return cache
    return None


def download_model(
    on_status: Optional[Callable[[str], None]] = None,
    spec: Optional[DetectorSpec] = None,
) -> Path:
    """Ensure the detector ONNX exists locally, downloading it if missing.

    PINTO's default hosting is a per-size ``.tar.gz`` containing every export
    resolution; only the single ``spec.member`` file is extracted into the
    cache and the archive is deleted. A direct ``.onnx`` URL is saved as-is.
    ``.part`` temp + atomic rename so a killed download can never leave a
    truncated model behind."""
    spec = spec or active_spec()
    existing = model_path(spec)
    if existing is not None:
        return existing

    cache = default_cache(spec)
    cache.parent.mkdir(parents=True, exist_ok=True)

    def _report(block: int, block_size: int, total: int) -> None:
        if on_status and total > 0:
            done = min(block * block_size, total)
            on_status(f"Downloading {spec.name} detector… "
                      f"{done / 1e6:.0f}/{total / 1e6:.0f} MB")

    if on_status:
        on_status(f"Downloading {spec.name} detector from {spec.url}")

    if spec.url.endswith((".tar.gz", ".tgz")):
        with tempfile.NamedTemporaryFile(
                suffix=".tar.gz", dir=cache.parent, delete=False) as tf:
            archive = Path(tf.name)
        try:
            urllib.request.urlretrieve(spec.url, archive, reporthook=_report)
            if on_status:
                on_status(f"Extracting {spec.member}…")
            with tarfile.open(archive, "r:gz") as tar:
                src = tar.extractfile(spec.member)
                if src is None:
                    raise FileNotFoundError(
                        f"{spec.member} not in {spec.url}")
                tmp = cache.with_suffix(".onnx.part")
                with open(tmp, "wb") as dst:
                    while chunk := src.read(1 << 20):
                        dst.write(chunk)
                tmp.replace(cache)
        finally:
            archive.unlink(missing_ok=True)
    else:
        tmp = cache.with_suffix(".onnx.part")
        urllib.request.urlretrieve(spec.url, tmp, reporthook=_report)
        tmp.replace(cache)
    return cache


def _pinned_model(path: str, hw: tuple[int, int]) -> str:
    """Return a copy of the model with its input pinned to ``1×3×H×W``.

    PINTO's per-resolution exports still declare symbolic H/W on the graph
    input; DirectML validates shapes strictly and rejects symbolic dims inside
    Reshape nodes, so pin the input and re-infer the downstream shapes.
    Cached on disk.
    """
    import onnx
    from onnxruntime.tools.onnx_model_utils import (
        fix_output_shapes, make_input_shape_fixed)

    h, w = hw
    cache_dir = Path(tempfile.gettempdir()) / "FaceBlurInspector-models"
    cache_dir.mkdir(exist_ok=True)
    # Key on size+mtime as well as name/shape (same convention as
    # utils.fp16_model_path). Without the fingerprint, swapping the model via
    # AVPP_DETECTOR_ONNX for a different file with the same basename silently
    # reuses the previously pinned graph — you benchmark the old model and
    # never find out.
    try:
        st = Path(path).stat()
        stamp = f"_{st.st_size}_{int(st.st_mtime)}"
    except OSError:
        stamp = ""
    fixed = cache_dir / f"{Path(path).stem}_pin{w}x{h}{stamp}.onnx"
    if not fixed.exists():
        model = onnx.load(path)
        make_input_shape_fixed(model.graph, model.graph.input[0].name,
                               [1, 3, h, w])
        fix_output_shapes(model)
        onnx.save(model, str(fixed))
    return str(fixed)


@dataclass
class Detections:
    """Per-class boxes, frame coordinates.

    ``heads``/``faces``/``bodies`` are (K, 5) float32 ``[x1, y1, x2, y2,
    score]``. ``parts``/``negatives`` are (K, 6) float32 ``[x1, y1, x2, y2,
    score, cls]`` — ``cls`` indexes ``PART_EYE/PART_NOSE/PART_MOUTH/PART_EAR``
    or ``NEG_HAND/NEG_FOOT`` respectively. Both are evidence for
    libs/evidence.py's gate, never blur candidates themselves; either is
    empty when the active :class:`DetectorSpec` has no such classes.
    """
    heads: np.ndarray = field(
        default_factory=lambda: np.empty((0, 5), np.float32))
    faces: np.ndarray = field(
        default_factory=lambda: np.empty((0, 5), np.float32))
    bodies: np.ndarray = field(
        default_factory=lambda: np.empty((0, 5), np.float32))
    parts: np.ndarray = field(
        default_factory=lambda: np.empty((0, 6), np.float32))
    negatives: np.ndarray = field(
        default_factory=lambda: np.empty((0, 6), np.float32))


def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
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


def _nms(boxes: np.ndarray, iou_thr: float = _MERGE_IOU) -> np.ndarray:
    """Greedy score-ordered NMS over (K,5) boxes. Used only to merge the
    outputs of multiple rotation passes — within one pass the graph's own NMS
    already deduplicated."""
    if len(boxes) <= 1:
        return boxes
    order = np.argsort(-boxes[:, 4])
    boxes = boxes[order]
    keep: list[int] = []
    ious = _iou_matrix(boxes, boxes)
    suppressed = np.zeros(len(boxes), dtype=bool)
    for i in range(len(boxes)):
        if suppressed[i]:
            continue
        keep.append(i)
        suppressed |= ious[i] > iou_thr
        suppressed[i] = True
    return boxes[keep]


def _centres_inside(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(N, M) bool — a[i]'s centre lies inside b[j]'s box."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), bool)
    acx = (a[:, 0] + a[:, 2]) / 2
    acy = (a[:, 1] + a[:, 3]) / 2
    return ((b[None, :, 0] <= acx[:, None]) & (acx[:, None] <= b[None, :, 2])
            & (b[None, :, 1] <= acy[:, None]) & (acy[:, None] <= b[None, :, 3]))


def match_faces(boxes: np.ndarray, witnesses: np.ndarray,
                iou_thr: float = _WITNESS_MATCH_IOU) -> np.ndarray:
    """True per box when some witness face agrees with it: IoU beyond
    ``iou_thr`` or centre containment either way (two detectors box the same
    face at different scales, and a face sits well inside a head box)."""
    if len(boxes) == 0 or len(witnesses) == 0:
        return np.zeros(len(boxes), bool)
    return ((_iou_matrix(boxes, witnesses) >= iou_thr).any(axis=1)
            | _centres_inside(boxes, witnesses).any(axis=1)
            | _centres_inside(witnesses, boxes).any(axis=0))


def _unrotate_boxes(boxes: np.ndarray, rot: int, fw: int, fh: int) -> np.ndarray:
    """Map (K,5) boxes detected on a rotated frame back to original coords.

    ``rot`` is the cv2 rotation that produced the frame the boxes live in:
    90 = ROTATE_90_CLOCKWISE, 270 = ROTATE_90_COUNTERCLOCKWISE, 180.
    ``fw``/``fh`` are the *original* frame's dimensions.
    """
    if len(boxes) == 0 or rot == 0:
        return boxes
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    if rot == 90:      # (x, y) → (fh-1-y, x); inverse: (xr, yr) → (yr, fh-1-xr)
        nx1, ny1 = y1, fh - 1 - x2
        nx2, ny2 = y2, fh - 1 - x1
    elif rot == 270:   # (x, y) → (y, fw-1-x); inverse: (xr, yr) → (fw-1-yr, xr)
        nx1, ny1 = fw - 1 - y2, x1
        nx2, ny2 = fw - 1 - y1, x2
    elif rot == 180:
        nx1, ny1 = fw - 1 - x2, fh - 1 - y2
        nx2, ny2 = fw - 1 - x1, fh - 1 - y1
    else:
        return boxes
    out = boxes.copy()
    out[:, 0], out[:, 1], out[:, 2], out[:, 3] = nx1, ny1, nx2, ny2
    return out


class HeadDetector:
    """Lazy single-session wrapper → per-class boxes for the tracker.

    Session lifecycle follows the DirectML law (module docstring): built once
    on first use, never destroyed, never rebuilt. Any failure flips
    ``available`` to False and the pipeline degrades to "no blur + status
    line" rather than crashing mid-export.
    """

    def __init__(self, on_status: Optional[Callable[[str], None]] = None) -> None:
        self.spec = active_spec()
        self._on_status = on_status
        self._sess = None
        self._in_name = ""
        self.available: Optional[bool] = None
        self.last_ms = 0.0

    def _status(self, msg: str) -> None:
        if self._on_status:
            self._on_status(msg)

    def _ensure(self) -> bool:
        if self.available is not None:
            return self.available
        try:
            path = model_path(self.spec)
            if path is None:
                raise FileNotFoundError(
                    f"no detector ONNX found (set AVPP_DETECTOR_ONNX or place "
                    f"it at {default_cache(self.spec)})")
            import onnxruntime as ort

            from .utils import best_onnx_providers, make_session

            self._status(f"Loading {self.spec.name} detector ({path.name})…")
            pinned = _pinned_model(str(path), self.spec.input_hw)
            providers = best_onnx_providers()
            try:
                sess = make_session(pinned, providers)
            except Exception:  # noqa: BLE001 — fp16/DML rejected → plain fp32
                sess = ort.InferenceSession(pinned, providers=providers)
            self._in_name = sess.get_inputs()[0].name
            self._sess = sess
            self.available = True
            h, w = self.spec.input_hw
            self._status(f"{self.spec.name} detector ready on "
                         f"{sess.get_providers()[0]} @ {w}×{h}")
        except Exception as exc:  # noqa: BLE001 — degrade, never crash
            self._status(f"{self.spec.name} detector unavailable: {exc!r}")
            self.available = False
        return self.available

    def _infer(self, frame_bgr: np.ndarray) -> np.ndarray:
        """One forward pass → raw [N,7] rows in *frame* pixel coordinates."""
        import cv2

        fh, fw = frame_bgr.shape[:2]
        h, w = self.spec.input_hw
        blob = cv2.resize(frame_bgr, (w, h), interpolation=cv2.INTER_LINEAR)
        blob = np.ascontiguousarray(
            blob.transpose(2, 0, 1)[None].astype(np.float32))
        out = np.asarray(self._sess.run(None, {self._in_name: blob})[0])
        if out.ndim == 3:
            out = out[0]
        if out.ndim != 2 or out.shape[-1] != 7 or len(out) == 0:
            return np.empty((0, 7), np.float32)
        out = out.astype(np.float32, copy=True)
        bx = list(self.spec.col_box)
        out[:, bx[0]] *= fw / w
        out[:, bx[2]] *= fw / w
        out[:, bx[1]] *= fh / h
        out[:, bx[3]] *= fh / h
        return out

    def _split(self, rows: np.ndarray, fw: int, fh: int) -> Detections:
        s = self.spec
        cls = rows[:, s.col_cls].astype(int)
        score = rows[:, s.col_score]
        boxes = rows[:, list(s.col_box)]
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, fw - 1)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, fh - 1)
        packed = np.concatenate([boxes, score[:, None]], axis=1)
        ok_size = ((boxes[:, 2] - boxes[:, 0] >= 4)
                   & (boxes[:, 3] - boxes[:, 1] >= 4))

        def take(ids: tuple[int, ...], floor: float) -> np.ndarray:
            m = np.isin(cls, ids) & (score >= floor) & ok_size
            return packed[m].astype(np.float32)

        def take_tagged(groups: tuple[tuple[int, tuple[int, ...]], ...],
                        floor: float) -> np.ndarray:
            """Like ``take`` but for several id-groups, each tagged with its
            enum value in an appended 6th column (PART_*/NEG_*)."""
            out = []
            for tag, ids in groups:
                if not ids:
                    continue
                m = np.isin(cls, ids) & (score >= floor) & ok_size
                if not m.any():
                    continue
                out.append(np.concatenate(
                    [packed[m], np.full((int(m.sum()), 1), tag, np.float32)],
                    axis=1))
            return (np.concatenate(out).astype(np.float32) if out
                    else np.empty((0, 6), np.float32))

        return Detections(
            heads=take(s.head_ids, HEAD_FLOOR),
            faces=take(s.face_ids, FACE_FLOOR),
            bodies=take(s.body_ids, BODY_FLOOR),
            parts=take_tagged(
                ((PART_EYE, s.eye_ids), (PART_NOSE, s.nose_ids),
                 (PART_MOUTH, s.mouth_ids), (PART_EAR, s.ear_ids)),
                PART_FLOOR),
            negatives=take_tagged(
                ((NEG_HAND, s.hand_ids), (NEG_FOOT, s.foot_ids)), NEG_FLOOR),
        )

    @staticmethod
    def _filter_rotated(boxes: np.ndarray, upright: np.ndarray,
                        witnesses: np.ndarray, fw: int, fh: int) -> np.ndarray:
        """Acceptance gates for a rotated pass's boxes (see _ROT_* consts):
        a witness-dependent score floor, an absolute size cap (a real head
        recovered by rotation assist is never a fifth of the frame — only
        hallucinations are), and no box that contains an upright-pass box of
        half its area or less (the giant-hallucination signature).

        ``upright`` is the same class's upright boxes (containment gate);
        ``witnesses`` is *all* upright evidence — heads and faces at the
        parse floor plus bodies. A rotated box is corroborated when it
        overlaps a weak upright head/face beyond _ROT_WITNESS_IOU (the IoU
        bar keeps a big fake from being blessed by a speck of upright noise
        it happens to cover) or its centre lies inside an upright body; only
        corroborated boxes get the normal floor, the rest need
        _ROT_LONE_FLOOR."""
        if len(boxes) == 0:
            return boxes
        floor = np.full(len(boxes), _ROT_LONE_FLOOR, np.float32)
        if len(witnesses):
            iou_ok = (_iou_matrix(boxes, witnesses)
                      >= _ROT_WITNESS_IOU).any(axis=1)
            bcx = (boxes[:, 0] + boxes[:, 2]) / 2
            bcy = (boxes[:, 1] + boxes[:, 3]) / 2
            inside = ((witnesses[None, :, 0] <= bcx[:, None])
                      & (bcx[:, None] <= witnesses[None, :, 2])
                      & (witnesses[None, :, 1] <= bcy[:, None])
                      & (bcy[:, None] <= witnesses[None, :, 3])).any(axis=1)
            floor[iou_ok | inside] = _ROT_SCORE_FLOOR
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        boxes = boxes[(boxes[:, 4] >= floor)
                      & (area <= _ROT_MAX_AREA_FRAC * fw * fh)]
        if len(boxes) == 0 or len(upright) == 0:
            return boxes
        ucx = (upright[:, 0] + upright[:, 2]) / 2
        ucy = (upright[:, 1] + upright[:, 3]) / 2
        uarea = (upright[:, 2] - upright[:, 0]) * (upright[:, 3] - upright[:, 1])
        keep = []
        for b in boxes:
            barea = (b[2] - b[0]) * (b[3] - b[1])
            inside = ((b[0] <= ucx) & (ucx <= b[2])
                      & (b[1] <= ucy) & (ucy <= b[3])
                      & (uarea <= barea * _ROT_CONTAIN_AREA))
            if not inside.any():
                keep.append(b)
        return (np.stack(keep) if keep
                else np.empty((0, 5), np.float32))

    def detect(self, frame_bgr: np.ndarray,
               rotations: tuple[int, ...] = (0,)) -> Detections:
        """Detect on ``frame_bgr``; optionally also on rotated copies.

        ``rotations`` beyond ``(0,)`` re-run the same pinned session on 90°/
        180°/270° copies and merge the unrotated results — recall insurance
        for sideways heads (bed angles) that upright-trained detectors miss.
        Cost is linear in the number of rotations. Rotated heads/faces pass
        through the ``_filter_rotated`` anti-hallucination gates (bodies come
        from the upright pass only); rotated parts/negatives are simply
        unioned in at their normal floor — they are gate evidence in
        libs/evidence.py, weighed and overridable, never a box a
        hallucination can blur by itself, so they don't need the same
        machinery.
        """
        if not self._ensure():
            return Detections()
        import cv2

        rot_code = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180,
                    270: cv2.ROTATE_90_COUNTERCLOCKWISE}
        fh, fw = frame_bgr.shape[:2]
        try:
            t0 = time.perf_counter()
            base: Optional[Detections] = None
            extra_heads: list[np.ndarray] = []
            extra_faces: list[np.ndarray] = []
            extra_parts: list[np.ndarray] = []
            extra_negs: list[np.ndarray] = []
            for rot in rotations:
                img = frame_bgr if rot == 0 else cv2.rotate(
                    frame_bgr, rot_code[rot])
                rows = self._infer(img)
                d = self._split(rows, *(
                    (fw, fh) if rot in (0, 180) else (fh, fw)))
                if rot == 0:
                    base = d
                else:
                    extra_heads.append(_unrotate_boxes(d.heads, rot, fw, fh))
                    extra_faces.append(_unrotate_boxes(d.faces, rot, fw, fh))
                    extra_parts.append(_unrotate_boxes(d.parts, rot, fw, fh))
                    extra_negs.append(
                        _unrotate_boxes(d.negatives, rot, fw, fh))
            self.last_ms = (time.perf_counter() - t0) * 1e3
            if base is None:
                base = Detections()
            if not extra_heads and not extra_faces \
                    and not extra_parts and not extra_negs:
                return base
            witnesses = np.concatenate(
                [base.heads, base.faces, base.bodies])
            rot_heads = self._filter_rotated(
                np.concatenate(extra_heads) if extra_heads
                else np.empty((0, 5), np.float32),
                base.heads, witnesses, fw, fh)
            rot_faces = self._filter_rotated(
                np.concatenate(extra_faces) if extra_faces
                else np.empty((0, 5), np.float32),
                base.faces, witnesses, fw, fh)
            parts = (np.concatenate([base.parts, *extra_parts])
                     if extra_parts else base.parts)
            negatives = (np.concatenate([base.negatives, *extra_negs])
                        if extra_negs else base.negatives)
            return Detections(
                heads=_nms(np.concatenate([base.heads, rot_heads])),
                faces=_nms(np.concatenate([base.faces, rot_faces])),
                bodies=base.bodies,
                parts=parts,
                negatives=negatives)
        except Exception as exc:  # noqa: BLE001 — one bad frame ≠ crash
            self._status(f"{self.spec.name} detect failed, degrading: {exc!r}")
            self.available = False
            return Detections()
