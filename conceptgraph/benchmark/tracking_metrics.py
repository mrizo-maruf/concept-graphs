"""3D multi-object tracking metrics for the ConceptGraphs benchmark.

Defines the data containers (``GTInstance`` / ``PredInstance`` / ``FrameRecord``),
the 3D AABB IoU primitive, two matchers (greedy + Hungarian), and the per-scene
``MetricsAccumulator`` that produces MOTA, MOTP, T-mIoU, T-SR and ID-consistency.

All metrics are computed in **3D bbox space** (axis-aligned). Each GT/Pred
instance carries an ``(xmin, ymin, zmin, xmax, ymax, zmax)`` tuple in world
coordinates.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment as _hungarian
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "scipy is required for Hungarian matching. Install with `pip install scipy`."
    ) from e


Aabb = Tuple[float, float, float, float, float, float]


@dataclass
class GTInstance:
    track_id: int
    class_name: str
    mask: Optional[np.ndarray]                # 2D bool mask, optional
    bbox_xyxy: Optional[Tuple[float, float, float, float]]
    bbox_xyzxyz: Aabb                         # world-space AABB (required)


@dataclass
class PredInstance:
    pred_id: int
    class_name: Optional[str]
    bbox_xyzxyz: Aabb
    mask: Optional[np.ndarray] = None


@dataclass
class FrameRecord:
    frame_idx: int
    gt_objects: List[GTInstance]
    pred_objects: List[PredInstance]
    # gt_track_id -> pred_id (only matched gts appear)
    mapping: Dict[int, int] = field(default_factory=dict)
    # (gt_track_id, pred_id) -> IoU
    ious: Dict[Tuple[int, int], float] = field(default_factory=dict)


# ----------------------------------------------------------------- IoU 3D
def iou_aabb(a: Aabb, b: Aabb) -> float:
    """Volumetric IoU between two axis-aligned 3D boxes (xyzxyz)."""
    ax1, ay1, az1, ax2, ay2, az2 = a
    bx1, by1, bz1, bx2, by2, bz2 = b
    ix1, iy1, iz1 = max(ax1, bx1), max(ay1, by1), max(az1, bz1)
    ix2, iy2, iz2 = min(ax2, bx2), min(ay2, by2), min(az2, bz2)
    iw, ih, idp = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1), max(0.0, iz2 - iz1)
    inter = iw * ih * idp
    va = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1) * max(0.0, az2 - az1)
    vb = max(0.0, bx2 - bx1) * max(0.0, by2 - by1) * max(0.0, bz2 - bz1)
    union = va + vb - inter
    return float(inter / union) if union > 0 else 0.0


def build_iou_matrix(
    gt: Sequence[GTInstance],
    pred: Sequence[PredInstance],
) -> np.ndarray:
    """Return an ``(|gt|, |pred|)`` matrix of pairwise 3D IoUs."""
    M = np.zeros((len(gt), len(pred)), dtype=np.float64)
    for i, g in enumerate(gt):
        for j, p in enumerate(pred):
            M[i, j] = iou_aabb(g.bbox_xyzxyz, p.bbox_xyzxyz)
    return M


# ----------------------------------------------------------------- matchers
def _match_result(
    gt: Sequence[GTInstance],
    pred: Sequence[PredInstance],
    pairs: Sequence[Tuple[int, int]],
    iou_mat: np.ndarray,
    iou_threshold: float,
) -> Tuple[Dict[int, int], Dict[Tuple[int, int], float]]:
    mapping: Dict[int, int] = {}
    ious: Dict[Tuple[int, int], float] = {}
    for i, j in pairs:
        v = float(iou_mat[i, j])
        if v < iou_threshold:
            continue
        mapping[gt[i].track_id] = pred[j].pred_id
        ious[(gt[i].track_id, pred[j].pred_id)] = v
    return mapping, ious


def greedy_match(
    gt: Sequence[GTInstance],
    pred: Sequence[PredInstance],
    iou_threshold: float = 0.25,
    match_mode: str = "bbox3d",
) -> Tuple[Dict[int, int], Dict[Tuple[int, int], float]]:
    """Greedy 1-1 matching: pick the highest-IoU pair, lock it, repeat."""
    if not gt or not pred:
        return {}, {}
    M = build_iou_matrix(gt, pred)
    used_g, used_p = set(), set()
    pairs: List[Tuple[int, int]] = []
    flat = [(M[i, j], i, j) for i in range(M.shape[0]) for j in range(M.shape[1])]
    flat.sort(reverse=True)
    for v, i, j in flat:
        if v < iou_threshold:
            break
        if i in used_g or j in used_p:
            continue
        used_g.add(i); used_p.add(j)
        pairs.append((i, j))
    return _match_result(gt, pred, pairs, M, iou_threshold)


def hungarian_match(
    gt: Sequence[GTInstance],
    pred: Sequence[PredInstance],
    iou_threshold: float = 0.25,
    match_mode: str = "bbox3d",
) -> Tuple[Dict[int, int], Dict[Tuple[int, int], float]]:
    """Optimal 1-1 assignment via Hungarian algorithm on cost = 1 - IoU.

    Pairs with IoU < ``iou_threshold`` are dropped after the assignment.
    """
    if not gt or not pred:
        return {}, {}
    M = build_iou_matrix(gt, pred)
    # Block infeasible pairs with a large cost so they never get picked unless forced.
    cost = 1.0 - M
    cost[M < iou_threshold] = 1e6
    row_ind, col_ind = _hungarian(cost)
    pairs = [(int(i), int(j)) for i, j in zip(row_ind, col_ind) if cost[i, j] < 1e6 - 1]
    return _match_result(gt, pred, pairs, M, iou_threshold)


def match_mode_factory(name: str) -> Callable:
    name = (name or "hungarian").lower()
    if name == "hungarian":
        return hungarian_match
    if name == "greedy":
        return greedy_match
    raise ValueError(f"Unknown matcher '{name}'. Use 'hungarian' or 'greedy'.")


# ----------------------------------------------------------------- accumulator
class MetricsAccumulator:
    """Streams per-frame ``FrameRecord``s and computes summary metrics.

    Metric formulas (all in 3D AABB-IoU space, thresholded by `iou_threshold`):

    * **MOTA**  = 1 - (FN + FP + IDSW) / GT_total  (CLEAR MOT, Bernardin 2008).
    * **MOTP**  = mean IoU over all true-positive matches (IoU-based variant).
    * **T-mIoU** = same as MOTP but reported separately + std-dev.
    * **T-SR**   = per-frame recall @ IoU>=thr (matches / |GT|), averaged across frames.
    * **ID consistency** = for each GT track, fraction of its matched frames that
      went to its *dominant* predicted id (the most common pred_id assigned to
      it). Averaged across GT tracks. 1.0 means every GT object always got the
      same predicted id; 0 means every frame got a different id.
    * **ID switches** = #frames where matched(pred_id) != last_matched(pred_id)
      for the same GT track.
    """

    def __init__(self) -> None:
        self.records: List[FrameRecord] = []

    def add_frame(self, record: FrameRecord) -> None:
        self.records.append(record)

    # ---- summary ------------------------------------------------------
    def compute(self) -> dict:
        total_gt = 0
        total_pred = 0
        total_tp = 0
        total_fn = 0
        total_fp = 0
        idsw_total = 0

        sr_per_frame: List[float] = []
        all_match_ious: List[float] = []

        gt_to_pred_history: Dict[int, List[int]] = defaultdict(list)
        last_pred_for_gt: Dict[int, int] = {}
        unique_gt: set = set()

        for rec in self.records:
            n_gt = len(rec.gt_objects)
            n_pred = len(rec.pred_objects)
            n_match = len(rec.mapping)
            total_gt += n_gt
            total_pred += n_pred
            total_tp += n_match
            total_fn += n_gt - n_match
            total_fp += n_pred - n_match
            for g in rec.gt_objects:
                unique_gt.add(g.track_id)

            if n_gt > 0:
                sr_per_frame.append(n_match / float(n_gt))

            for gt_id, pred_id in rec.mapping.items():
                iou = rec.ious.get((gt_id, pred_id), 0.0)
                all_match_ious.append(iou)
                gt_to_pred_history[gt_id].append(pred_id)
                prev = last_pred_for_gt.get(gt_id)
                if prev is not None and prev != pred_id:
                    idsw_total += 1
                last_pred_for_gt[gt_id] = pred_id

        # ID consistency (per-GT-track dominance ratio, then macro-mean).
        id_consistency_per_track: List[float] = []
        for hist in gt_to_pred_history.values():
            if not hist:
                continue
            dominant = Counter(hist).most_common(1)[0][1]
            id_consistency_per_track.append(dominant / len(hist))
        id_consistency = (
            float(np.mean(id_consistency_per_track)) if id_consistency_per_track else 0.0
        )

        mota_num = total_fn + total_fp + idsw_total
        mota = 1.0 - mota_num / total_gt if total_gt > 0 else 0.0
        motp = float(np.mean(all_match_ious)) if all_match_ious else 0.0
        t_miou = motp
        t_miou_std = float(np.std(all_match_ious)) if all_match_ious else 0.0
        t_sr = float(np.mean(sr_per_frame)) if sr_per_frame else 0.0

        return {
            "frames_processed": len(self.records),
            "unique_gt_objects": len(unique_gt),
            "total_gt_instances": total_gt,
            "total_pred_instances": total_pred,
            "total_matches": total_tp,
            "total_false_positives": total_fp,
            "total_false_negatives": total_fn,
            "ID_switches_total": idsw_total,
            "MOTA": float(mota),
            "MOTA_FN_ratio": total_fn / total_gt if total_gt else 0.0,
            "MOTA_FP_ratio": total_fp / total_gt if total_gt else 0.0,
            "MOTA_IDSW_ratio": idsw_total / total_gt if total_gt else 0.0,
            "MOTP": motp,
            "T_mIoU": t_miou,
            "T_mIoU_std": t_miou_std,
            "T_SR": t_sr,
            "ID_consistency": id_consistency,
        }

    # ---- per-frame printout (used for live progress) ------------------
    def frame_summary(self, record: FrameRecord) -> dict:
        n_gt = len(record.gt_objects)
        n_pred = len(record.pred_objects)
        n_match = len(record.mapping)
        mean_iou = (
            float(np.mean(list(record.ious.values()))) if record.ious else 0.0
        )
        return {
            "frame": record.frame_idx,
            "GT": n_gt,
            "Pred": n_pred,
            "Matches": n_match,
            "FN": n_gt - n_match,
            "FP": n_pred - n_match,
            "mean_IoU": mean_iou,
        }


# ----------------------------------------------------------------- IO helpers
def _fmt_row(label: str, value, width: int = 28) -> str:
    if isinstance(value, float):
        return f"  {label:<{width}}{value:.4f}"
    return f"  {label:<{width}}{value}"


def print_summary(results: dict, title: str = "TRACKING METRICS") -> None:
    bar = "=" * 64
    print(f"\n{bar}\n{title}\n{bar}")
    headline_order = [
        "MOTA", "MOTP", "T_mIoU", "T_mIoU_std", "T_SR", "ID_consistency",
        "ID_switches_total",
        "total_gt_instances", "total_pred_instances", "total_matches",
        "total_false_positives", "total_false_negatives",
        "unique_gt_objects", "frames_processed",
    ]
    for k in headline_order:
        if k in results:
            print(_fmt_row(k, results[k]))
    # Print anything else (e.g. scene name).
    for k, v in results.items():
        if k in headline_order or k == "per_scene_summary":
            continue
        if isinstance(v, (int, float, str)):
            print(_fmt_row(k, v))
    print(bar)


def save_metrics(results: dict, out_dir: Path, scene_name: str) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{scene_name}.json"
    # Replace numpy scalars with python natives for clean JSON.
    def _native(x):
        if isinstance(x, dict):
            return {k: _native(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [_native(v) for v in x]
        if isinstance(x, np.generic):
            return x.item()
        return x

    with open(path, "w") as f:
        json.dump(_native(results), f, indent=2)
    return path
