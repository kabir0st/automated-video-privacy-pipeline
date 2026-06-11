"""Minimal FaceAnalysis replacement that loads only the models we use.

insightface.app.FaceAnalysis instantiates an ONNX Runtime session for every
model file in the buffalo_l pack just to identify it, then immediately
destroys the sessions it doesn't keep. With the DirectML execution provider
destroying a session corrupts the provider's device state, and the next
inference on a surviving session dies with a native access violation — no
Python traceback, which is what silently killed the frozen Windows build the
moment a video was opened.

FaceApp creates sessions only for det_10g (SCRFD detection) and 2d106det
(106-point landmarks), so no session is ever created and thrown away. For the
same reason it must never be re-instantiated to change det_size — call
prepare() again instead; it re-prepares the existing sessions in place.
"""
from __future__ import annotations

import os.path as osp
import tempfile
from pathlib import Path

import numpy as np
import onnx
from onnxruntime.tools.onnx_model_utils import fix_output_shapes, make_input_shape_fixed

from insightface.app.common import Face
from insightface.model_zoo import model_zoo
from insightface.utils import ensure_available


def _det_model_for_size(det_file: str, size: tuple[int, int]) -> str:
    """Return a det_10g variant whose graph is pinned to the given det size.

    The shipped det_10g.onnx declares internal/output tensor shapes traced at
    640×640 even though its input is dynamic. The CPU provider only warns
    about the mismatch, but DirectML validates strictly and fails any other
    det size inside a Reshape node ("80070057 The parameter is incorrect").
    Pinning the input and re-inferring the downstream shapes produces a
    consistent graph that DirectML accepts; results are cached on disk.
    """
    w, h = size
    cache_dir = Path(tempfile.gettempdir()) / "FaceBlurInspector-models"
    cache_dir.mkdir(exist_ok=True)
    fixed = cache_dir / f"det_10g_{w}x{h}.onnx"
    if not fixed.exists():
        model = onnx.load(det_file)
        make_input_shape_fixed(model.graph, model.graph.input[0].name, [1, 3, h, w])
        fix_output_shapes(model)
        onnx.save(model, str(fixed))
    return str(fixed)


class FaceApp:
    """Drop-in for this project's use of FaceAnalysis(name="buffalo_l",
    allowed_modules=["detection", "landmark_2d_106"])."""

    def __init__(self, providers: list[str]) -> None:
        # Downloads the buffalo_l pack on first run, same as FaceAnalysis.
        model_dir = ensure_available("models", "buffalo_l", root="~/.insightface")
        self._providers = list(providers)
        self._det_file = osp.join(model_dir, "det_10g.onnx")
        self.lmk_model = model_zoo.get_model(
            osp.join(model_dir, "2d106det.onnx"), providers=self._providers)
        # Each det size gets its own session built from a shape-pinned model
        # file (see _det_model_for_size), cached for the lifetime of the app —
        # see the module docstring for why old sessions must never be
        # destroyed. The landmark model always sees fixed 192×192 crops and
        # needs no such handling.
        self._det_by_size: dict[tuple[int, int], object] = {}
        self.det_model = None

    def prepare(self, ctx_id: int, det_size: tuple[int, int],
                det_thresh: float = 0.5) -> None:
        key = (int(det_size[0]), int(det_size[1]))
        det = self._det_by_size.get(key)
        if det is None:
            det = model_zoo.get_model(
                _det_model_for_size(self._det_file, key), providers=self._providers)
            self._det_by_size[key] = det
        det.prepare(ctx_id, input_size=key, det_thresh=det_thresh)
        self.det_model = det
        self.lmk_model.prepare(ctx_id)

    def get(self, img: np.ndarray, max_num: int = 0) -> list[Face]:
        bboxes, kpss = self.det_model.detect(img, max_num=max_num, metric="default")
        faces: list[Face] = []
        for i in range(bboxes.shape[0]):
            face = Face(bbox=bboxes[i, 0:4],
                        kps=None if kpss is None else kpss[i],
                        det_score=bboxes[i, 4])
            self.lmk_model.get(img, face)
            faces.append(face)
        return faces
