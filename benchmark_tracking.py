"""3D tracking benchmark for the ConceptGraphs pipeline on IsaacSim sequences.

Single-scene mode:
    python benchmark_tracking.py \
        --pipeline_config conceptgraph/configs/slam_pipeline/base.yaml \
        --dataset_config conceptgraph/dataset/dataconfigs/isaacsim/isaacsim.yaml \
        --dataset_root /path/to/isaacsim_dataset \
        --scene_id scene_001 \
        --gsa_variant ram_withbg_allclasses

Multi-scene mode (one ConceptGraphs run per subfolder under --dataset_root):
    python benchmark_tracking.py \
        --pipeline_config conceptgraph/configs/slam_pipeline/base.yaml \
        --dataset_config conceptgraph/dataset/dataconfigs/isaacsim/isaacsim.yaml \
        --dataset_root /path/to/isaacsim_dataset \
        --gsa_variant ram_withbg_allclasses \
        --all_scenes

Each scene folder must contain ``rgb/``, ``depth/``, ``bbox/``, ``seg/`` and
``traj.txt`` (the IsaacSim layout). Per-frame and per-scene metrics are
printed and saved, plus a macro average across scenes at the end.

Precondition: run ``conceptgraph/scripts/generate_gsa_results.py`` on each
scene first so each scene has a ``gsa_detections_<variant>/`` folder and a
``gsa_classes_<variant>.json`` next to it (same as for any other dataset).
"""
from __future__ import annotations

import argparse
import copy
import gzip
import json
import os
import pickle
import random
import re
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
from loguru import logger
from omegaconf import OmegaConf
from tqdm import tqdm

from conceptgraph.benchmark.cg_tracker import ConceptGraphTracker, bbox_to_xyzxyz
from conceptgraph.benchmark.isaacsim_gt import DEFAULT_SKIP_LABELS, load_gt_for_frame
from conceptgraph.benchmark.tracking_metrics import (
    FrameRecord,
    MetricsAccumulator,
    PredInstance,
    match_mode_factory,
    print_summary,
    save_metrics,
)
from conceptgraph.dataset.datasets_common import get_dataset
from conceptgraph.slam.slam_classes import MapObjectList
from conceptgraph.slam.utils import create_or_load_colors

warnings.filterwarnings("ignore")

_FRAME_RE = re.compile(r"frame(\d+)\.(?:jpg|png)$")


def set_seed(seed: int = 18) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def frame_num_from_path(path: str) -> Optional[int]:
    m = _FRAME_RE.search(path)
    return int(m.group(1)) if m else None


def discover_scenes(root: Path) -> List[Path]:
    """Subfolders of *root* that look like IsaacSim scenes (have rgb/ and bbox/)."""
    if not root.exists() or not root.is_dir():
        raise RuntimeError(f"Dataset root not found or not a directory: {root}")
    scenes = sorted(
        d for d in root.iterdir()
        if d.is_dir() and (d / "rgb").is_dir() and (d / "bbox").is_dir()
    )
    if not scenes:
        raise RuntimeError(
            f"No scenes under {root}. Expected subfolders containing rgb/ and bbox/."
        )
    return scenes


class StableTrackIDs:
    """Maps each map-object dict to a monotonic integer track id.

    Keyed on ``id(obj)`` of the dict; holds a strong ref to every dict it has
    seen so a dropped track's address cannot be recycled by another object.
    """

    def __init__(self) -> None:
        self._tid: Dict[int, int] = {}
        self._refs: List[dict] = []
        self._next = 0

    def get(self, obj: dict) -> int:
        key = id(obj)
        tid = self._tid.get(key)
        if tid is None:
            tid = self._next
            self._next += 1
            self._tid[key] = tid
            self._refs.append(obj)
        return tid


def build_cfg(args: argparse.Namespace, scene_dir: Path) -> OmegaConf:
    """Build the OmegaConf used by the ConceptGraphs slam internals."""
    base = OmegaConf.load(args.pipeline_config)
    base.dataset_root = str(scene_dir.parent)
    base.dataset_config = str(args.dataset_config)
    base.scene_id = scene_dir.name
    base.start = args.start
    base.end = args.end
    base.stride = args.stride
    base.gsa_variant = args.gsa_variant
    base.detection_folder_name = f"gsa_detections_{args.gsa_variant}"
    base.color_file_name = f"gsa_classes_{args.gsa_variant}"
    base.save_pcd = bool(args.save_pcd)
    base.save_suffix = f"benchmark_{args.gsa_variant}"
    base.save_objects_all_frames = False
    base.vis_render = False
    # Internal slam code expects Path-like dataset_root + scene_id.
    base.dataset_root = str(scene_dir.parent)
    # Mirror cfslam_pipeline_batch.process_cfg behaviour: pull image size from dataset config.
    dataset_cfg = OmegaConf.load(args.dataset_config)
    base.image_height = dataset_cfg.camera_params.image_height
    base.image_width = dataset_cfg.camera_params.image_width
    return base


def benchmark_scene(
    scene_dir: Path,
    args: argparse.Namespace,
    run_dir: Path,
) -> dict:
    """Run ConceptGraphs over one scene; return per-scene metrics dict."""
    cfg = build_cfg(args, scene_dir)
    # The slam utilities expect cfg.dataset_root to be a path-like that supports `/`.
    cfg.dataset_root = str(cfg.dataset_root)
    cfg_internal = copy.deepcopy(cfg)
    cfg_internal.dataset_root = Path(cfg.dataset_root)

    dataset = get_dataset(
        dataconfig=cfg.dataset_config,
        start=cfg.start,
        end=cfg.end,
        stride=cfg.stride,
        basedir=cfg.dataset_root,
        sequence=cfg.scene_id,
        desired_height=cfg.image_height,
        desired_width=cfg.image_width,
        device="cpu",
        dtype=torch.float,
        relative_pose=False,
    )

    classes, _ = create_or_load_colors(cfg_internal, cfg.color_file_name)
    tracker = ConceptGraphTracker(cfg_internal, classes)
    tracker_ids = StableTrackIDs()
    accumulator = MetricsAccumulator()
    matcher = match_mode_factory(args.matcher)

    n_frames = len(dataset)
    if args.limit is not None:
        n_frames = min(n_frames, args.limit)

    pred_sidecar: Dict[int, List[dict]] = {}
    n_with_gt = 0

    scene_out_dir = run_dir / scene_dir.name
    mask_dir = scene_out_dir / "masks"
    if args.save_masks:
        mask_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        f"[{scene_dir.name}] {n_frames} frames | matcher={args.matcher} | "
        f"IoU>={args.iou_threshold:.2f}"
    )
    pbar = tqdm(range(n_frames), desc=scene_dir.name, leave=False)
    for step_idx in pbar:
        frame = dataset[step_idx]
        color_path = dataset.color_paths[step_idx]
        try:
            tracker.integrate(step_idx, frame, color_path)
        except FileNotFoundError as e:
            raise RuntimeError(f"[{scene_dir.name}] {e}") from e
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        frame_num = frame_num_from_path(color_path)
        if frame_num is None:
            logger.warning(
                f"[{scene_dir.name}] step {step_idx}: cannot parse frame number from "
                f"{Path(color_path).name}; skipping GT comparison."
            )
            continue

        gt_instances = load_gt_for_frame(
            scene_dir, frame_num,
            skip_labels=DEFAULT_SKIP_LABELS,
            load_masks=False,
        )
        if not gt_instances:
            continue
        n_with_gt += 1

        # ---- gather predictions for this frame ------------------------
        pred_instances: List[PredInstance] = []
        per_frame_pred_records: List[dict] = []
        for _, obj in tracker.predictions_for_frame(step_idx):
            bbox = obj.get("bbox")
            if bbox is None:
                continue
            try:
                bb = bbox_to_xyzxyz(bbox)
            except Exception as e:
                logger.debug(f"[{scene_dir.name}] frame {step_idx}: bbox parse failed: {e}")
                continue
            track_id = tracker_ids.get(obj)
            class_name = None
            if obj.get("class_name"):
                class_name = obj["class_name"][-1]
            pred_instances.append(PredInstance(
                pred_id=track_id,
                class_name=class_name,
                bbox_xyzxyz=bb,
            ))
            # Find the index of this frame in the object's detection history so the
            # visualiser can recover the original 2D mask without re-running the pipeline.
            try:
                local_i = obj["image_idx"].index(step_idx)
            except ValueError:
                local_i = None
            mask_rel_path = None
            if args.save_masks and local_i is not None and "mask" in obj:
                m = obj["mask"][local_i]
                m_arr = np.asarray(m)
                if m_arr.dtype != np.uint8:
                    m_arr = (m_arr > 0).astype(np.uint8) * 255
                mask_filename = f"frame{step_idx:06d}_t{track_id}.png"
                cv2.imwrite(str(mask_dir / mask_filename), m_arr)
                mask_rel_path = f"masks/{mask_filename}"
            per_frame_pred_records.append({
                "track_id": track_id,
                "class_name": class_name,
                "bbox_xyzxyz": list(bb),
                "xyxy": (
                    [float(v) for v in obj["xyxy"][local_i]]
                    if local_i is not None and "xyxy" in obj else None
                ),
                "local_det_idx": local_i,
                "mask_idx": (
                    int(obj["mask_idx"][local_i])
                    if local_i is not None and "mask_idx" in obj else None
                ),
                "mask_path": mask_rel_path,
            })

        mapping, ious = matcher(
            gt_instances, pred_instances,
            iou_threshold=args.iou_threshold,
            match_mode="bbox3d",
        )
        rec = FrameRecord(
            frame_idx=step_idx,
            gt_objects=gt_instances,
            pred_objects=pred_instances,
            mapping=mapping,
            ious=ious,
        )
        accumulator.add_frame(rec)

        # Per-frame log line.
        fs = accumulator.frame_summary(rec)
        pbar.set_postfix(GT=fs["GT"], P=fs["Pred"], M=fs["Matches"],
                          mIoU=f"{fs['mean_IoU']:.2f}")
        if args.print_every_frame:
            logger.info(
                f"[{scene_dir.name}] frame {step_idx:04d} | "
                f"GT={fs['GT']} Pred={fs['Pred']} Matches={fs['Matches']} "
                f"FN={fs['FN']} FP={fs['FP']} mean_IoU={fs['mean_IoU']:.3f}"
            )

        pred_sidecar[step_idx] = per_frame_pred_records

    if n_with_gt == 0:
        raise RuntimeError(
            f"[{scene_dir.name}] no GT loaded. Check that "
            f"{scene_dir}/bbox/bboxes######_info.json files exist."
        )

    tracker.finalize()

    results = accumulator.compute()
    results["scene"] = scene_dir.name
    results["frames_with_gt"] = n_with_gt

    # Save sidecar predictions (used by visualize_tracking.py).
    scene_out_dir.mkdir(parents=True, exist_ok=True)
    with gzip.open(scene_out_dir / "predictions_per_frame.pkl.gz", "wb") as f:
        pickle.dump({
            "scene_dir": str(scene_dir),
            "color_paths": [str(p) for p in dataset.color_paths[:n_frames]],
            "frame_predictions": pred_sidecar,
            "iou_threshold": args.iou_threshold,
            "gsa_variant": args.gsa_variant,
            "detection_folder_name": f"gsa_detections_{args.gsa_variant}",
        }, f)

    if args.save_final_objects:
        final_objs = MapObjectList(list(tracker.objects)).to_serializable()
        with gzip.open(scene_out_dir / "final_objects.pkl.gz", "wb") as f:
            pickle.dump(final_objs, f)

    return results


# ----------------------------------------------------------------- aggregation
_AGG_MEAN_KEYS = [
    "T_mIoU", "T_mIoU_std", "T_SR", "ID_consistency",
    "MOTA", "MOTA_FN_ratio", "MOTA_FP_ratio", "MOTA_IDSW_ratio", "MOTP",
]
_AGG_SUM_KEYS = [
    "frames_processed", "unique_gt_objects",
    "total_gt_instances", "total_pred_instances", "total_matches",
    "total_false_positives", "total_false_negatives", "ID_switches_total",
]


def aggregate_macro(per_scene: Dict[str, dict]) -> dict:
    out: Dict = {}
    for k in _AGG_MEAN_KEYS:
        vals = [r[k] for r in per_scene.values() if k in r]
        out[k] = float(np.mean(vals)) if vals else 0.0
    for k in _AGG_SUM_KEYS:
        out[k] = int(sum(int(r.get(k, 0)) for r in per_scene.values()))
    out["scenes_evaluated"] = len(per_scene)
    out["per_scene_summary"] = {
        name: {k: r.get(k) for k in _AGG_MEAN_KEYS}
        for name, r in per_scene.items()
    }
    return out


def main(args: argparse.Namespace) -> None:
    if args.all_scenes:
        if not args.dataset_root:
            raise RuntimeError("--all_scenes requires --dataset_root.")
        scenes = discover_scenes(Path(args.dataset_root))
        logger.info(f"Discovered {len(scenes)} scenes under {args.dataset_root}.")
    else:
        if not (args.dataset_root and args.scene_id):
            raise RuntimeError(
                "Single-scene mode needs --dataset_root and --scene_id "
                "(or pass --all_scenes for multi-scene)."
            )
        scenes = [Path(args.dataset_root) / args.scene_id]

    run_dir = Path(args.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Writing outputs to {run_dir}.")

    per_scene: Dict[str, dict] = {}
    failures: Dict[str, str] = {}

    for scene_dir in scenes:
        try:
            results = benchmark_scene(scene_dir, args, run_dir)
        except Exception as e:
            logger.exception(f"[{scene_dir.name}] failed: {e}")
            failures[scene_dir.name] = str(e)
            continue

        print_summary(results, title=f"3D TRACKING - {scene_dir.name}")
        save_metrics(results, run_dir, scene_name=scene_dir.name)
        per_scene[scene_dir.name] = results

    if len(per_scene) > 1:
        agg = aggregate_macro(per_scene)
        print_summary(agg, title=f"MACRO AVG ACROSS {agg['scenes_evaluated']} SCENES")
        save_metrics(agg, run_dir, scene_name="_macro_avg_all_scenes")

    if failures:
        logger.warning(f"{len(failures)} scene(s) failed: {sorted(failures)}")
        with open(run_dir / "_failures.json", "w") as f:
            json.dump(failures, f, indent=2)

    logger.info(f"All outputs saved under {run_dir}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="3D tracking benchmark for ConceptGraphs on IsaacSim."
    )
    parser.add_argument(
        "--pipeline_config",
        default="conceptgraph/configs/slam_pipeline/base.yaml",
        help="ConceptGraphs slam_pipeline YAML (overridden per-scene).",
    )
    parser.add_argument(
        "--dataset_config",
        default="conceptgraph/dataset/dataconfigs/isaacsim/isaacsim.yaml",
        help="Dataset YAML with camera intrinsics + png_depth_scale.",
    )
    parser.add_argument(
        "--dataset_root", required=True,
        help="Path to the IsaacSim root (folder containing scene subfolders).",
    )
    parser.add_argument(
        "--scene_id", default=None,
        help="Name of one scene subfolder under --dataset_root (single-scene mode).",
    )
    parser.add_argument(
        "--all_scenes", action="store_true",
        help="Benchmark every scene subfolder under --dataset_root.",
    )
    parser.add_argument(
        "--gsa_variant", default="ram_withbg_allclasses",
        help="Same gsa_variant used in generate_gsa_results.py for this scene.",
    )
    parser.add_argument(
        "--output_dir", default="benchmark_results",
        help="Where per-run metric JSONs are written.",
    )
    parser.add_argument("--iou_threshold", type=float, default=0.25,
                        help="3D bbox IoU threshold for a successful match.")
    parser.add_argument("--matcher", choices=["greedy", "hungarian"], default="hungarian")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=-1)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None,
                        help="Per scene: process at most this many frames (smoke test).")
    parser.add_argument("--print_every_frame", action="store_true",
                        help="Log a one-line per-frame summary in addition to the progress bar.")
    parser.add_argument("--save_pcd", action="store_true",
                        help="Forward to ConceptGraphs: save full point cloud per scene.")
    parser.add_argument("--save_final_objects", action="store_true",
                        help="Save the finalised MapObjectList serialisation per scene.")
    parser.add_argument(
        "--save_masks", action=argparse.BooleanOptionalAction, default=True,
        help="Save per-frame 2D masks for the visualiser (default on; --no-save_masks to disable).",
    )
    parser.add_argument("--logger_level", default="INFO")
    args = parser.parse_args()

    logger.remove()
    logger.add(lambda m: tqdm.write(m, end=""), level=args.logger_level, colorize=True)

    set_seed()
    main(args)
