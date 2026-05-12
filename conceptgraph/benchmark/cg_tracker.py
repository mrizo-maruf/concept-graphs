"""Per-frame ConceptGraphs mapping wrapper used by the tracking benchmark.

This re-implements the inner loop of ``conceptgraph/slam/cfslam_pipeline_batch.py``
as a stateful object so the benchmark can drive it one frame at a time, then
query which final 3D objects were observed in this frame and what their
2D masks were. The pipeline behaviour (similarity, merging, post-processing)
is unchanged — only the iteration structure is refactored.
"""
from __future__ import annotations

import gzip
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from omegaconf import DictConfig

from conceptgraph.slam.mapping import (
    aggregate_similarities,
    compute_spatial_similarities,
    compute_visual_similarities,
    merge_detections_to_objects,
)
from conceptgraph.slam.slam_classes import DetectionList, MapObjectList
from conceptgraph.slam.utils import (
    denoise_objects,
    filter_objects,
    gobs_to_detection_list,
    merge_obj2_into_obj1,
    merge_objects,
)
from conceptgraph.utils.ious import compute_2d_box_contained_batch


BG_CLASSES = ["wall", "floor", "ceiling"]


class ConceptGraphTracker:
    """Drives one ConceptGraphs mapping pass frame-by-frame.

    Usage:
        tracker = ConceptGraphTracker(cfg, classes)
        for idx in range(len(dataset)):
            frame = dataset[idx]
            tracker.integrate(idx, frame, dataset.color_paths[idx])
            preds = tracker.predictions_for_frame(idx)
        tracker.finalize()
    """

    def __init__(self, cfg: DictConfig, classes: List[str]) -> None:
        self.cfg = cfg
        self.classes = classes
        self.objects: MapObjectList = MapObjectList(device=cfg.device)
        if not cfg.skip_bg:
            self.bg_objects: Optional[Dict[str, dict]] = {c: None for c in BG_CLASSES}
        else:
            self.bg_objects = None

    # ---- frame integration -------------------------------------------------
    def integrate(self, idx: int, frame, color_path: str) -> None:
        """Process one dataset frame. ``frame`` is a dataset ``__getitem__`` tuple."""
        color_tensor, depth_tensor, intrinsics, *_ = frame
        color_np = color_tensor.cpu().numpy()
        image_rgb = color_np.astype(np.uint8)
        depth_array = depth_tensor[..., 0].cpu().numpy()
        cam_K = intrinsics.cpu().numpy()[:3, :3]

        detections_path = (
            Path(self.cfg.dataset_root)
            / self.cfg.scene_id
            / self.cfg.detection_folder_name
            / Path(color_path).name
        ).with_suffix(".pkl.gz")
        if not detections_path.exists():
            raise FileNotFoundError(
                f"Detection file {detections_path} missing. Run "
                f"`scripts/generate_gsa_results.py` for scene "
                f"`{self.cfg.scene_id}` first."
            )
        with gzip.open(detections_path, "rb") as f:
            gobs = pickle.load(f)

        # Camera pose is whatever the dataset stored.
        # Use the un-transformed (absolute) pose so 3D bboxes are in world space.
        # Note: the dataset's relative_pose flag must be False at construction.
        # We rely on `dataset.poses` (loaded c2w) being absolute.
        adjusted_pose = frame[3].cpu().numpy() if len(frame) > 3 else None
        if adjusted_pose is None:
            raise RuntimeError("Frame does not include a pose; check dataset loader.")

        fg_dets, bg_dets = gobs_to_detection_list(
            cfg=self.cfg,
            image=image_rgb,
            depth_array=depth_array,
            cam_K=cam_K,
            idx=idx,
            gobs=gobs,
            trans_pose=adjusted_pose,
            class_names=self.classes,
            BG_CLASSES=BG_CLASSES,
            color_path=color_path,
        )

        if self.bg_objects is not None and len(bg_dets) > 0:
            for det in bg_dets:
                cname = det["class_name"][0]
                if self.bg_objects[cname] is None:
                    self.bg_objects[cname] = det
                else:
                    self.bg_objects[cname] = merge_obj2_into_obj1(
                        self.cfg, self.bg_objects[cname], det, run_dbscan=False
                    )

        if len(fg_dets) == 0:
            return

        if self.cfg.use_contain_number:
            xyxy = fg_dets.get_stacked_values_torch("xyxy", 0)
            contain_numbers = compute_2d_box_contained_batch(
                xyxy, self.cfg.contain_area_thresh
            )
            for i in range(len(fg_dets)):
                fg_dets[i]["contain_number"] = [contain_numbers[i]]

        if len(self.objects) == 0:
            for det in fg_dets:
                self.objects.append(det)
            return

        spatial_sim = compute_spatial_similarities(self.cfg, fg_dets, self.objects)
        visual_sim = compute_visual_similarities(self.cfg, fg_dets, self.objects)
        agg_sim = aggregate_similarities(self.cfg, spatial_sim, visual_sim)

        if self.cfg.use_contain_number:
            cn_objs = torch.Tensor([o["contain_number"][0] for o in self.objects])
            d_contained = contain_numbers > 0
            o_contained = cn_objs > 0
            xor = d_contained.unsqueeze(1) ^ o_contained.unsqueeze(0)
            agg_sim[xor] = agg_sim[xor] - self.cfg.contain_mismatch_penalty

        agg_sim[agg_sim < self.cfg.sim_threshold] = float("-inf")
        self.objects = merge_detections_to_objects(
            self.cfg, fg_dets, self.objects, agg_sim
        )

        if self.cfg.denoise_interval > 0 and (idx + 1) % self.cfg.denoise_interval == 0:
            self.objects = denoise_objects(self.cfg, self.objects)
        if self.cfg.filter_interval > 0 and (idx + 1) % self.cfg.filter_interval == 0:
            self.objects = filter_objects(self.cfg, self.objects)
        if self.cfg.merge_interval > 0 and (idx + 1) % self.cfg.merge_interval == 0:
            self.objects = merge_objects(self.cfg, self.objects)

    # ---- prediction extraction --------------------------------------------
    def predictions_for_frame(self, idx: int) -> List[Tuple[int, dict]]:
        """Return ``[(stable_track_id, object_dict), ...]`` for objects seen in this frame.

        The "stable track id" is ``id(obj)`` of the python dict in the current
        ``self.objects`` list. Because ``merge_obj2_into_obj1`` mutates obj1
        in place, identity is preserved across merges of new detections into
        existing tracks. After ``filter_objects`` / ``merge_objects`` runs at
        the end of a scene, surviving track ids remain stable; dropped tracks
        simply never appear again.
        """
        out: List[Tuple[int, dict]] = []
        for obj in self.objects:
            image_idx_list = obj.get("image_idx")
            if image_idx_list is None:
                continue
            if idx in image_idx_list:
                out.append((id(obj), obj))
        return out

    def finalize(self) -> None:
        """Run the same end-of-pipeline post-processing as the original script."""
        if self.bg_objects is not None:
            self.bg_objects = MapObjectList(
                [o for o in self.bg_objects.values() if o is not None]
            )
            self.bg_objects = denoise_objects(self.cfg, self.bg_objects)
        self.objects = denoise_objects(self.cfg, self.objects)
        self.objects = filter_objects(self.cfg, self.objects)
        self.objects = merge_objects(self.cfg, self.objects)


def bbox_to_xyzxyz(bbox) -> Tuple[float, float, float, float, float, float]:
    """Convert an Open3D AxisAlignedBoundingBox / OrientedBoundingBox to an AABB tuple."""
    pts = np.asarray(bbox.get_box_points())
    mn = pts.min(axis=0)
    mx = pts.max(axis=0)
    return (
        float(mn[0]), float(mn[1]), float(mn[2]),
        float(mx[0]), float(mx[1]), float(mx[2]),
    )
