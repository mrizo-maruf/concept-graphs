"""Ground-truth loader for IsaacSim sequences used by the tracking benchmark.

Each scene folder contains, per frame:

    bbox/bboxes######_info.json   — both 2D (tight) and 3D AABB GT, with track ids
    seg/semantic######.png        — colourised instance segmentation
    seg/semantic######_info.json  — int instance id -> {label, color_bgr, ...}

This module turns one frame's JSON+PNG into a list of
``GTInstance``s the metric module can consume.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

from conceptgraph.benchmark.tracking_metrics import GTInstance

DEFAULT_SKIP_LABELS: Set[str] = {
    "wall", "floor", "ground", "ceiling", "background", "unlabelled", "unlabeled",
}


def _read_json(path: Path) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        return None


def _parse_bbox2d_tight(bbox_data: dict) -> Dict[int, dict]:
    """Return ``bbox_2d_id -> raw 2D box dict``."""
    boxes = (
        bbox_data.get("bboxes", {})
        .get("bbox_2d_tight", {})
        .get("boxes", [])
    )
    out: Dict[int, dict] = {}
    for b in boxes:
        bid = b.get("bbox_2d_id", b.get("bbox_id"))
        if bid is not None:
            out[int(bid)] = b
    return out


def _parse_bbox3d(bbox_data: dict) -> List[dict]:
    return [
        b for b in bbox_data.get("bboxes", {}).get("bbox_3d", {}).get("boxes", [])
        if "track_id" in b
    ]


def _parse_instance_color_map(seg_info: dict) -> Dict[int, Tuple[int, int, int]]:
    """``instance_seg_id -> (B, G, R)`` for direct comparison against a BGR PNG."""
    out: Dict[int, Tuple[int, int, int]] = {}
    for k, v in seg_info.items():
        if not str(k).isdigit():
            continue
        c = v.get("color_bgr")
        if c and len(c) == 3:
            out[int(k)] = (int(c[0]), int(c[1]), int(c[2]))
    return out


def _infer_class_name(b3d: dict, b2d: Optional[dict]) -> str:
    if isinstance(b3d.get("label"), str) and b3d["label"]:
        return b3d["label"]
    if b2d is not None:
        lab = b2d.get("label")
        if isinstance(lab, dict) and lab:
            return str(next(iter(lab.values())))
        if isinstance(lab, str) and lab:
            return lab
    return "unknown"


def _extract_bbox_xyzxyz(b3d: dict) -> Optional[Tuple[float, ...]]:
    aabb = b3d.get("aabb_xyzmin_xyzmax")
    if isinstance(aabb, (list, tuple)) and len(aabb) == 6:
        return tuple(float(v) for v in aabb)
    aabb = b3d.get("aabb")
    if isinstance(aabb, dict):
        mn, mx = aabb.get("min"), aabb.get("max")
        if (
            isinstance(mn, (list, tuple)) and isinstance(mx, (list, tuple))
            and len(mn) == 3 and len(mx) == 3
        ):
            return tuple(float(v) for v in list(mn) + list(mx))
    return None


def load_gt_for_frame(
    scene_dir: Path,
    frame_num: int,
    skip_labels: Optional[Set[str]] = None,
    load_masks: bool = False,
) -> Optional[List[GTInstance]]:
    """Load all GT instances for one frame.

    ``frame_num`` is the integer in the filename (``frame000005`` → 5).
    Returns ``None`` if the GT bbox JSON for that frame is missing.
    """
    if skip_labels is None:
        skip_labels = DEFAULT_SKIP_LABELS

    fs = f"{frame_num:06d}"
    bbox_path = scene_dir / "bbox" / f"bboxes{fs}_info.json"
    seg_png_path = scene_dir / "seg" / f"semantic{fs}.png"
    seg_info_path = scene_dir / "seg" / f"semantic{fs}_info.json"

    bbox_data = _read_json(bbox_path)
    if bbox_data is None:
        return None

    seg_info = _read_json(seg_info_path) or {}
    seg_img = None
    if load_masks and seg_png_path.exists():
        seg_img = cv2.imread(str(seg_png_path), cv2.IMREAD_COLOR)

    bbox2d_by_id = _parse_bbox2d_tight(bbox_data)
    bbox3d_list = _parse_bbox3d(bbox_data)
    color_by_inst = _parse_instance_color_map(seg_info)

    instances: List[GTInstance] = []
    for b3d in bbox3d_list:
        track_id = int(b3d["track_id"])
        inst_seg_id = int(b3d.get("instance_seg_id", -1))
        bbox_2d_id = int(b3d.get("bbox_2d_id", -1))
        bbox_3d_id = int(b3d.get("bbox_3d_id", -1))
        if bbox_3d_id < 0 or bbox_2d_id < 0 or inst_seg_id < 0:
            continue

        b2d = bbox2d_by_id.get(bbox_2d_id)
        class_name = _infer_class_name(b3d, b2d)
        if any(skip in class_name.lower() for skip in skip_labels):
            continue

        bbox_xyxy = None
        if b2d is not None:
            xyxy = b2d.get("xyxy")
            if xyxy and len(xyxy) == 4:
                bbox_xyxy = tuple(float(v) for v in xyxy)

        mask = None
        if seg_img is not None:
            color = color_by_inst.get(inst_seg_id)
            if color is not None:
                mask = np.all(seg_img == np.array(color, dtype=np.uint8), axis=2)

        bbox_xyzxyz = _extract_bbox_xyzxyz(b3d)
        if bbox_xyzxyz is None:
            continue

        instances.append(GTInstance(
            track_id=track_id,
            class_name=class_name,
            mask=mask,
            bbox_xyxy=bbox_xyxy,
            bbox_xyzxyz=bbox_xyzxyz,
        ))
    return instances
