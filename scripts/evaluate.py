"""Measure head coverage on your own footage, without ground truth.

    uv run python scripts/evaluate.py CLIP.mp4 [--every 20] [--preset Balanced] [--sheet OUT.png]

Builds an independent *consensus* on every ``--every``-th frame — a head that
two detector families agree on at any rotation, or one with a landmark-
plausible SCRFD face inside — then loads the clip's project sidecar (run
``cli.py analyse`` first), rebuilds the blur table with the given preset and
the saved review decisions, and reports:

  * consensus-head coverage   — fraction of consensus heads a blur box covers
  * face coverage             — fraction of confident faces inside a blur box
  * unsupported blur boxes    — blur boxes no source at all agrees with
  * frames blurred, on/off transitions

Per-source recall against the consensus is printed too, which is how the
defaults in docs/TECH_STACK.md were chosen. ``--sheet`` writes a contact
sheet of the frames with the most disagreement (consensus heads red where
uncovered, blur boxes green, every source thin) — look at it before trusting
any number here; the consensus is a proxy, not truth.

One set of sessions for the whole run (never rebuilt — see the session-churn
rule in the README).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from libs.scrfd import WITNESS_FLOOR, kps_plausible          # noqa: E402
from pipeline import project as proj                          # noqa: E402
from pipeline.analysis import Models                          # noqa: E402
from pipeline.geom import iou_matrix, unrotate_boxes          # noqa: E402
from pipeline.presets import PRESETS                          # noqa: E402
from pipeline.render import build_table                       # noqa: E402
from pipeline.session import run_offline                      # noqa: E402

ROTS = (0, 90, 180, 270)
_ROT = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}


def sample(video: str, every: int, models: Models) -> dict[int, dict]:
    """Per sampled frame: {'headdet': [(K,5) per rot], 'wb': [...], 'faces': (F,5)}."""
    cap = cv2.VideoCapture(video)
    out: dict[int, dict] = {}
    idx = 0
    hd, wb, sc = models.headdet(), models.wb(), models.scrfd()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % every == 0:
            fh, fw = frame.shape[:2]
            rec = {"headdet": [], "wb": [], "faces": []}
            for rot in ROTS:
                img = frame if rot == 0 else cv2.rotate(frame, _ROT[rot])
                b = hd.detect(img, rotations=(0,), floor=0.2)
                rec["headdet"].append(b if rot == 0 else unrotate_boxes(b, rot, fw, fh))
                d = wb.detect(img, rotations=(0,))
                h = d.heads[d.heads[:, 4] >= 0.2] if len(d.heads) else d.heads
                rec["wb"].append(h if rot == 0 else unrotate_boxes(h, rot, fw, fh))
                if rot in (0, 180):
                    fb, k = sc.detect_full(img, floor=WITNESS_FLOOR)
                    if len(fb):
                        fb = fb[kps_plausible(k)]
                        fb = fb[fb[:, 4] >= 0.5]
                        rec["faces"].append(fb if rot == 0 else unrotate_boxes(fb, rot, fw, fh))
            rec["faces"] = (np.concatenate(rec["faces"]) if rec["faces"]
                            else np.empty((0, 5), np.float32))
            out[idx] = rec
            if len(out) % 25 == 0:
                print(f"  sampled {len(out)} frames", flush=True)
        idx += 1
    cap.release()
    return out


def consensus(rec: dict) -> tuple[list[np.ndarray], dict[str, list[bool]], np.ndarray]:
    """→ (consensus boxes, per-family hit list aligned with them, all claims)."""
    claims, fams = [], []
    for fam in ("headdet", "wb"):
        for arr in rec[fam]:
            for b in arr:
                claims.append(b)
                fams.append(fam)
    if not claims:
        return [], {"headdet": [], "wb": []}, np.empty((0, 5), np.float32)
    boxes = np.stack(claims)
    order = np.argsort(-boxes[:, 4])
    m = iou_matrix(boxes[:, :4], boxes[:, :4])
    used = np.zeros(len(boxes), bool)
    faces = rec["faces"]
    cons, hits = [], {"headdet": [], "wb": []}
    for i in order:
        if used[i]:
            continue
        mem = np.nonzero((m[i] >= 0.4) & ~used)[0]
        used[mem] = True
        fam_set = {fams[k] for k in mem}
        box = boxes[mem[0]]
        face_in = False
        if len(faces):
            fcx = (faces[:, 0] + faces[:, 2]) / 2
            fcy = (faces[:, 1] + faces[:, 3]) / 2
            face_in = bool(((box[0] <= fcx) & (fcx <= box[2]) & (box[1] <= fcy)
                            & (fcy <= box[3])).any())
        if len(fam_set) >= 2 or face_in:
            cons.append(box)
            for fam in hits:
                hits[fam].append(fam in fam_set)
    return cons, hits, boxes


def covered(box: np.ndarray, blur: list[np.ndarray]) -> bool:
    if not blur:
        return False
    bb = np.stack(blur)[:, :4]
    if iou_matrix(box[None, :4], bb)[0].max() >= 0.3:
        return True
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return bool(((bb[:, 0] <= cx) & (cx <= bb[:, 2]) & (bb[:, 1] <= cy) & (cy <= bb[:, 3])).any())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--every", type=int, default=20)
    ap.add_argument("--preset", default="Balanced", choices=list(PRESETS))
    ap.add_argument("--sheet", default="")
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args()

    p = proj.load(args.video)
    if p is None:
        raise SystemExit("no project sidecar — run `cli.py analyse` on this clip first")
    models = Models(on_status=lambda m: print(f"  {m}"))
    preset = PRESETS[args.preset]
    tracks = run_offline(p, preset, models, verify=not args.no_verify)
    table = build_table(tracks, p.manual, p.n_frames, p.enabled, preset.render)
    on = np.array([len(r) > 0 for r in table])

    print(f"sampling every {args.every}th frame with every source at every rotation…")
    frames = sample(args.video, args.every, models)

    n_cons = hit_cons = n_face = hit_face = n_blur = unsupported = oversize = 0
    area_fracs: list[float] = []
    fam_hit = {"headdet": 0, "wb": 0}
    fam_tot = 0
    worst: list[tuple[int, int]] = []
    for f, rec in frames.items():
        blur = [b for _t, b in table[f]] if f < len(table) else []
        n_blur += len(blur)
        cons, hits, claims = consensus(rec)
        miss = 0
        for i, c in enumerate(cons):
            n_cons += 1
            fam_tot += 1
            for fam in fam_hit:
                fam_hit[fam] += int(hits[fam][i])
            if covered(c, blur):
                hit_cons += 1
            else:
                miss += 1
        for fc in rec["faces"]:
            n_face += 1
            if covered(fc, blur):
                hit_face += 1
        fh, fw = p.height or 720, p.width or 1280
        mask = np.zeros((fh, fw), bool)
        for b in blur:
            x1, y1 = int(np.clip(b[0], 0, fw)), int(np.clip(b[1], 0, fh))
            x2, y2 = int(np.clip(b[2], 0, fw)), int(np.clip(b[3], 0, fh))
            mask[y1:y2, x1:x2] = True
            near = False
            if len(claims):
                near = iou_matrix(b[None, :4], claims[:, :4])[0].max() >= 0.2
                cx = (claims[:, 0] + claims[:, 2]) / 2
                cy = (claims[:, 1] + claims[:, 3]) / 2
                inside = (b[0] <= cx) & (cx <= b[2]) & (b[1] <= cy) & (cy <= b[3])
                if inside.any():
                    carea = ((claims[inside, 2] - claims[inside, 0])
                             * (claims[inside, 3] - claims[inside, 1])).max()
                    if (b[2] - b[0]) * (b[3] - b[1]) > 3.0 * carea:
                        oversize += 1          # blur box ≫ any head it covers
                        miss += 1
            if not near and len(rec["faces"]):
                near = covered(b, [x for x in rec["faces"]])
            if not near:
                unsupported += 1
                miss += 1
        area_fracs.append(float(mask.mean()))
        worst.append((miss, f))

    trans = int((on[1:] != on[:-1]).sum())
    print(f"\nconsensus heads: {n_cons} on {len(frames)} sampled frames")
    for fam in fam_hit:
        print(f"  {fam:8s} recall vs consensus: {fam_hit[fam] / max(fam_tot, 1):.1%}")
    print(f"\n{args.preset} preset, saved decisions applied:")
    print(f"  consensus-head coverage  {hit_cons}/{n_cons} = {hit_cons / max(n_cons, 1):.1%}")
    print(f"  face coverage            {hit_face}/{n_face} = {hit_face / max(n_face, 1):.1%}")
    print(f"  unsupported blur boxes   {unsupported}/{n_blur}")
    print(f"  oversized blur boxes     {oversize}/{n_blur}   (over 3× the head they cover)")
    print(f"  blurred area             {np.mean(area_fracs) if area_fracs else 0:.1%} of the frame on average")
    print(f"  frames blurred           {int(on.sum())}/{len(on)}   on/off transitions {trans}")

    if args.sheet:
        worst.sort(reverse=True)
        cap = cv2.VideoCapture(args.video)
        tiles = []
        for _miss, f in worst[:24]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, img = cap.read()
            if not ok:
                continue
            rec = frames[f]
            blur = [b for _t, b in table[f]] if f < len(table) else []
            for arr in rec["headdet"] + rec["wb"]:
                for b in arr:
                    cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (120, 120, 120), 1)
            for c in consensus(rec)[0]:
                col = (0, 200, 0) if covered(c, blur) else (0, 0, 255)
                cv2.rectangle(img, (int(c[0]), int(c[1])), (int(c[2]), int(c[3])), col, 2)
            for b in blur:
                cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 200, 0), 2)
            cv2.putText(img, f"f{f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
            tiles.append(cv2.resize(img, (426, 240), interpolation=cv2.INTER_AREA))
        cap.release()
        while len(tiles) % 4:
            tiles.append(np.zeros_like(tiles[0]))
        rows = [np.concatenate(tiles[i:i + 4], axis=1) for i in range(0, len(tiles), 4)]
        cv2.imwrite(args.sheet, np.concatenate(rows, axis=0))
        print(f"  contact sheet → {args.sheet}  (red = consensus head not blurred, "
              f"cyan = blur box, green = covered)")


if __name__ == "__main__":
    main()
