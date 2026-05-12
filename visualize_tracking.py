"""Render an mp4 visualising ConceptGraphs 3D tracking on an IsaacSim scene.

Consumes the sidecar produced by ``benchmark_tracking.py`` (per-scene
``predictions_per_frame.pkl.gz`` + a ``masks/`` folder of per-track PNGs)
and overlays each predicted track on the RGB frame:

* mask colour is determined by track id (consistent across frames; when an
  object's id changes, its colour changes too);
* masks are blended at alpha = 0.5;
* the track id is drawn at the mask's centroid.

Usage:
    python visualize_tracking.py \
        --benchmark_run benchmark_results/20260512_140000 \
        --scene_id scene_001

By default writes ``<benchmark_run>/<scene_id>/tracking.mp4``.
"""
from __future__ import annotations

import argparse
import gzip
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

# Reproducible track-id -> colour. Track ids are dense ints starting at 0, so
# we just sample from a perceptually-distinct base palette and cycle.
_BASE_PALETTE = [
    (255, 56, 56),   (56, 255, 56),   (56, 56, 255),
    (255, 200, 56),  (255, 56, 255),  (56, 255, 255),
    (255, 128, 0),   (128, 0, 255),   (0, 255, 128),
    (200, 100, 100), (100, 200, 100), (100, 100, 200),
    (255, 180, 80),  (180, 80, 255),  (80, 255, 180),
    (220, 220, 80),  (80, 220, 220),  (220, 80, 220),
]


def color_for_track(track_id: int) -> Tuple[int, int, int]:
    """Deterministic BGR colour for a given track id."""
    base = _BASE_PALETTE[track_id % len(_BASE_PALETTE)]
    # Lightly perturb so big track id ranges don't collide perceptually.
    rng = np.random.default_rng(track_id)
    jitter = rng.integers(-30, 31, size=3)
    out = np.clip(np.array(base, dtype=np.int32) + jitter, 0, 255)
    # RGB -> BGR for OpenCV.
    return int(out[2]), int(out[1]), int(out[0])


def load_sidecar(scene_dir: Path) -> dict:
    p = scene_dir / "predictions_per_frame.pkl.gz"
    if not p.exists():
        raise FileNotFoundError(
            f"Cannot find {p}. Run benchmark_tracking.py first (saves this sidecar)."
        )
    with gzip.open(p, "rb") as f:
        return pickle.load(f)


def overlay_mask(
    image: np.ndarray,
    mask: np.ndarray,
    color_bgr: Tuple[int, int, int],
    alpha: float = 0.5,
) -> np.ndarray:
    """Blend a single-channel boolean/uint8 mask with the image at alpha."""
    if mask.dtype != bool:
        mask = mask > 0
    if not mask.any():
        return image
    color_layer = np.zeros_like(image)
    color_layer[mask] = color_bgr
    blended = image.copy()
    blended[mask] = cv2.addWeighted(
        image[mask], 1.0 - alpha, color_layer[mask], alpha, 0.0
    )
    return blended


def draw_label(
    image: np.ndarray,
    mask: np.ndarray,
    text: str,
    color_bgr: Tuple[int, int, int],
) -> None:
    """Write ``text`` centred at the centroid of ``mask``, in place."""
    if mask.dtype != bool:
        mask = mask > 0
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return
    cx, cy = int(xs.mean()), int(ys.mean())
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.7
    thickness = 2
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    pt1 = (cx - tw // 2 - 4, cy - th // 2 - 4)
    pt2 = (cx + tw // 2 + 4, cy + th // 2 + baseline)
    # Slight dark bg so text is readable against any mask colour.
    cv2.rectangle(image, pt1, pt2, (0, 0, 0), thickness=-1)
    cv2.putText(
        image, text, (cx - tw // 2, cy + th // 2),
        font, scale, color_bgr, thickness, cv2.LINE_AA,
    )


def render_frame(
    rgb_path: Path,
    preds: List[dict],
    scene_dir: Path,
    alpha: float,
) -> Optional[np.ndarray]:
    """Compose one annotated frame. Returns the BGR image or None on failure."""
    image = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if image is None:
        return None

    # Sort by mask area descending so small masks paint on top of big ones.
    sorted_preds = []
    for p in preds:
        mp = p.get("mask_path")
        if not mp:
            continue
        mfp = scene_dir / mp
        if not mfp.exists():
            continue
        m = cv2.imread(str(mfp), cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        sorted_preds.append((int((m > 0).sum()), p, m))
    sorted_preds.sort(key=lambda t: -t[0])

    for _area, p, mask in sorted_preds:
        if mask.shape[:2] != image.shape[:2]:
            mask = cv2.resize(
                mask, (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        color = color_for_track(int(p["track_id"]))
        image = overlay_mask(image, mask > 0, color, alpha=alpha)
        draw_label(image, mask > 0, str(int(p["track_id"])), (255, 255, 255))

    return image


def main(args: argparse.Namespace) -> None:
    scene_dir = Path(args.benchmark_run) / args.scene_id
    if not scene_dir.is_dir():
        raise RuntimeError(f"No such scene output dir: {scene_dir}")

    sidecar = load_sidecar(scene_dir)
    color_paths: List[str] = sidecar["color_paths"]
    frame_predictions: Dict[int, List[dict]] = sidecar["frame_predictions"]

    if not color_paths:
        raise RuntimeError(f"Sidecar at {scene_dir} has empty color_paths.")

    first_image = cv2.imread(color_paths[0], cv2.IMREAD_COLOR)
    if first_image is None:
        raise RuntimeError(f"Cannot open first RGB frame {color_paths[0]}.")
    h, w = first_image.shape[:2]

    output_path = Path(args.output) if args.output else scene_dir / "tracking.mp4"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, args.fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"cv2.VideoWriter could not open {output_path}.")

    n = len(color_paths)
    if args.limit is not None:
        n = min(n, args.limit)

    n_frames_with_preds = 0
    try:
        for step_idx in tqdm(range(n), desc=f"render {args.scene_id}"):
            preds = frame_predictions.get(step_idx, [])
            if preds:
                n_frames_with_preds += 1
            frame_img = render_frame(
                Path(color_paths[step_idx]), preds, scene_dir, args.alpha
            )
            if frame_img is None:
                # Fall back to original RGB if we cannot read it.
                continue
            if frame_img.shape[:2] != (h, w):
                frame_img = cv2.resize(frame_img, (w, h))
            writer.write(frame_img)
    finally:
        writer.release()

    print(
        f"Wrote {output_path} | {n} frames, {n_frames_with_preds} with predictions, "
        f"alpha={args.alpha}, fps={args.fps}."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Render an mp4 of ConceptGraphs 3D tracking on one IsaacSim scene."
    )
    parser.add_argument(
        "--benchmark_run", required=True,
        help="Timestamped output dir produced by benchmark_tracking.py.",
    )
    parser.add_argument(
        "--scene_id", required=True,
        help="Scene subfolder name inside --benchmark_run.",
    )
    parser.add_argument(
        "--output", default=None,
        help="MP4 output path (default: <benchmark_run>/<scene_id>/tracking.mp4).",
    )
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Mask blend factor (0=invisible, 1=opaque). Default 0.5.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Render at most this many frames (debugging).")
    args = parser.parse_args()
    main(args)
