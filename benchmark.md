# 3D Tracking Benchmark for ConceptGraphs on IsaacSim

This document describes how `benchmark_tracking.py` measures the 3D tracking
quality of the ConceptGraphs pipeline on IsaacSim sequences and how the
metrics are computed.

## TL;DR

```bash
# 1. (precondition) Generate 2D detections for every scene.
python conceptgraph/scripts/generate_gsa_results.py \
    --dataset_root /path/to/isaacsim_root \
    --dataset_config conceptgraph/dataset/dataconfigs/isaacsim/isaacsim.yaml \
    --scene_id <scene_name> \
    --class_set ram \
    --box_threshold 0.2 --text_threshold 0.2 \
    --stride 1 --add_bg_classes --accumu_classes \
    --exp_suffix withbg_allclasses

# 2. Benchmark every scene under a dataset root.
python benchmark_tracking.py \
    --dataset_root /path/to/isaacsim_root \
    --all_scenes \
    --gsa_variant ram_withbg_allclasses

# 3. Visualise tracking for one scene from a benchmark run.
python visualize_tracking.py \
    --benchmark_run benchmark_results/20260512_140000 \
    --scene_id <scene_name>
```

---

## Dataset layout

Each scene is a folder under `--dataset_root` matching the IsaacSim layout:

```
<scene_name>/
├── rgb/        frame000001.jpg, ...        # 1280×720 RGB
├── depth/      depth000001.png, ...        # 16-bit, depth_mm = pixel value
├── bbox/       bboxes000001_info.json, ... # GT 3D + 2D boxes with track_id
├── seg/        semantic000001.png, ...     # colourised instance segmentation
│               semantic000001_info.json    # instance_id -> {label, color_bgr}
└── traj.txt                                # 4×4 c2w pose per line (16 floats)
```

The dataset YAML at
[`conceptgraph/dataset/dataconfigs/isaacsim/isaacsim.yaml`](conceptgraph/dataset/dataconfigs/isaacsim/isaacsim.yaml)
holds intrinsics and `png_depth_scale`. The IsaacSim loader is registered in
[`datasets_common.py`](conceptgraph/dataset/datasets_common.py) and uses
`relative_pose=False` so predicted 3D boxes are in the **same world frame** as
the GT 3D AABBs.

---

## End-to-end flow

```
                ┌──────────────────────┐
                │ generate_gsa_results │  (once per scene, offline)
                └──────────┬───────────┘
                           │  gsa_detections_<variant>/*.pkl.gz
                           ▼
┌────────────────────────────────────────────────────────────────┐
│ benchmark_tracking.py                                          │
│                                                                │
│  for each scene:                                               │
│    ConceptGraphTracker.integrate(idx, frame, color_path)       │
│        ├── gobs_to_detection_list (project mask → 3D pcd)      │
│        ├── compute_spatial_similarities (3D IoU/overlap)       │
│        ├── compute_visual_similarities (CLIP cosine)           │
│        └── merge_detections_to_objects (in-place)              │
│                                                                │
│    tracker.predictions_for_frame(idx)                          │
│        → [(stable_track_id, object_dict), ...]                 │
│                                                                │
│    load_gt_for_frame(scene_dir, frame_num)                     │
│        → [GTInstance(track_id, class, 3D-AABB), ...]           │
│                                                                │
│    hungarian_match(GT, Pred, iou_thr=0.25)                     │
│        → mapping[gt_id] = pred_id                              │
│                                                                │
│    accumulator.add_frame(FrameRecord(...))                     │
│                                                                │
│  accumulator.compute() → MOTA, MOTP, T_mIoU, T_SR, ID_cons     │
└────────────────────────────────────────────────────────────────┘
```

`generate_gsa_results.py` is a hard precondition — the benchmark loads the
detections it dropped in `gsa_detections_<variant>/`. Skipping that step
raises `FileNotFoundError`.

---

## How predictions are extracted from ConceptGraphs

ConceptGraphs is a **static-scene mapper**, not a tracker in the per-frame
Kalman/IoU sense. But every merged map object stores its detection history,
so a tracker view drops out for free:

| field in `object` dict | meaning |
|---|---|
| `image_idx`  | list of frame indices the object was observed in |
| `mask_idx`   | per-observation index into that frame's detection pkl.gz |
| `xyxy`       | 2D bboxes per observation |
| `mask`       | 2D SAM masks per observation |
| `bbox`       | Open3D 3D bbox of the merged point cloud |
| `pcd`        | accumulated 3D point cloud |

For each frame `f`, the benchmark scans `tracker.objects` and emits every
object whose `image_idx` contains `f`. The track id is **`id(obj_dict)`**
mapped to a monotonic integer by `StableTrackIDs`. This works because:

* `merge_obj2_into_obj1` mutates `obj1` in place — when a new detection
  joins an existing track, the dict identity is preserved.
* `filter_objects` / `merge_objects` may drop or absorb tracks. Absorbed
  tracks simply disappear from later frames; the surviving dict keeps its id.
* `StableTrackIDs` holds strong refs to every dict it has assigned an id to,
  so the Python GC cannot recycle an address after a track is dropped.

The 3D AABB used for matching is `bbox.get_box_points().min/max` (axis-aligned),
which works for both `AxisAlignedBoundingBox` and `OrientedBoundingBox`.

---

## How GT is loaded

[`isaacsim_gt.load_gt_for_frame()`](conceptgraph/benchmark/isaacsim_gt.py)
reads `bbox/bboxes######_info.json` and emits one `GTInstance` per entry in
`bboxes.bbox_3d.boxes` with a `track_id`. We pull:

* `track_id` → integer GT identity (persists across frames).
* `aabb_xyzmin_xyzmax` → 3D AABB in world coords.
* `bbox_2d_id` + `bbox_2d_tight.boxes[]` → 2D bbox.
* `instance_seg_id` + `seg/semantic######_info.json` → BGR colour, used to
  pull the instance's 2D mask from `semantic######.png` (only loaded when
  `load_masks=True`, which the benchmark itself disables for speed).

Background categories — `wall`, `floor`, `ground`, `ceiling`, `background`,
`unlabel(l)ed` — are dropped from the GT set so the tracker isn't penalised
for not tracking them. (ConceptGraphs treats wall/floor/ceiling as a separate
`bg_objects` map by default.)

---

## How matching works

For every frame the benchmark builds the pairwise 3D IoU matrix
`M[i, j] = iou_aabb(gt[i].bbox_xyzxyz, pred[j].bbox_xyzxyz)`.

### Hungarian (default)
Cost matrix is `1 - M`, with pairs below `--iou_threshold` (default 0.25)
masked at cost `1e6` so the optimiser only picks them as a last resort —
which we then discard. `scipy.optimize.linear_sum_assignment` returns the
optimal 1-1 assignment minimising total cost.

### Greedy (`--matcher greedy`)
Sort all (i, j) pairs by IoU descending. Take the highest, lock those rows
and columns, repeat. Stops when the next pair's IoU drops below the
threshold.

Both produce `mapping: {gt_id → pred_id}` plus per-pair IoUs. Pass
`--matcher hungarian` (the default) for optimal assignment.

---

## Metrics

All defined in [`tracking_metrics.py`](conceptgraph/benchmark/tracking_metrics.py).
Let TP/FP/FN be summed across all frames at the chosen IoU threshold.

### MOTA — Multi-Object Tracking Accuracy
```
MOTA = 1 - (FN + FP + IDSW) / Σ |GT_per_frame|
```
Standard CLEAR-MOT definition (Bernardin & Stiefelhagen 2008). Penalises
misses, hallucinations, and identity switches together. Range
`(-∞, 1]`; 1 = perfect, can be negative if FP > GT.

### MOTP — Multi-Object Tracking Precision
```
MOTP = mean IoU over all matched pairs (TPs only)
```
Higher is better (IoU-style, range `[0, 1]`). Reports how well the matched
boxes align spatially, ignoring identity/recall.

### T-mIoU — Tracked mean IoU
Same value as MOTP above, plus its standard deviation. Reported separately
because some papers define MOTP as `1 - IoU` (distance) instead.

### T-SR — Tracking Success Rate
Per frame `f`: `SR_f = matches_f / |GT_f|` (recall@IoU≥thr).
`T-SR = mean over frames`. Equivalent to a "tracked-on-this-frame" rate.

### ID switches (IDSW)
For each GT track `g`, walk the frames where it was matched. Whenever the
matched `pred_id` differs from the previous matched `pred_id` for the same
GT, IDSW += 1.

### ID consistency
For each GT track `g`, find the **dominant** predicted id assigned to it
(the most frequent one in its match history).
`consistency(g) = (#frames matched to dominant) / (#total matched frames)`.
Final metric is the mean across GT tracks. 1.0 = every GT object got a
single persistent predicted id throughout; ~0 = predicted id flipped every
frame. This is a softer companion to IDSW that doesn't double-count
churn.

---

## Output layout

After a run, `--output_dir` contains a timestamped subdirectory with:

```
20260512_140000/
├── <scene_name>.json               # per-scene metrics
├── <scene_name>/
│   ├── predictions_per_frame.pkl.gz   # sidecar for visualize_tracking.py
│   └── final_objects.pkl.gz           # only with --save_final_objects
├── _macro_avg_all_scenes.json      # only with --all_scenes
└── _failures.json                  # any scenes that crashed
```

Each per-scene metric JSON contains the full dict produced by
`MetricsAccumulator.compute()` plus `scene`, `frames_with_gt`. The macro
average JSON contains the mean of headline metrics and the sum of count
metrics across all scenes, plus a per-scene breakdown under
`per_scene_summary`.

---

## Caveats & gotchas

1. **Static-scene assumption.** ConceptGraphs accumulates points across
   frames. If GT objects move within a sequence, the predicted 3D bbox is
   essentially time-averaged. MOTP / T-mIoU will be penalised for reasons
   that are not tracking errors. Recommend evaluating on sequences with
   stationary objects, or interpreting MOTP as a noisy upper bound on
   per-frame spatial precision.
2. **Open-vocab vs. taxonomy.** This benchmark is **class-agnostic** by
   design: matching uses only 3D IoU. The class predicted by the tagging
   model is recorded in the sidecar but not used by Hungarian. Add
   class-aware matching by filtering `M[i, j] = 0` when classes mismatch.
3. **Pose frame.** The dataset is loaded with `relative_pose=False` so the
   first pose isn't snapped to identity. The GT 3D AABBs in
   `bbox_3d.aabb_xyzmin_xyzmax` are in IsaacSim world coords; if you change
   the loader to use a different pose convention you must transform GT
   accordingly.
4. **Detections must already exist.** The benchmark does not re-run SAM /
   Grounding-DINO; run `generate_gsa_results.py` first.
5. **IDSW counting.** We count a switch only between two consecutive
   *matched* frames for the same GT. If a GT goes unmatched in the middle
   and then comes back with a different id, that counts as a single switch
   when it reappears.
