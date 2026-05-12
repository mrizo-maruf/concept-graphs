# Running the ConceptGraphs 3D-tracking benchmark on IsaacSim

End-to-end runbook for Ubuntu 20.04 / 16 GB VRAM.

## 1. Hardware check

| Resource | Required | Notes |
|---|---|---|
| GPU VRAM | ≥ 10 GB | 16 GB is plenty. Peak comes from `generate_gsa_results.py` (RAM + Grounding-DINO + SAM-H + CLIP co-resident ≈ 9–11 GB). The mapping/benchmark stage itself uses ~1–2 GB. |
| CUDA driver | ≥ 11.8 | CUDA 11.8 toolkit recommended (matches Pytorch 2.0.1 wheel). |
| RAM | 16 GB | comfortable. |
| Disk | ~5–10 GB per scene | Detections (`gsa_detections_*`) dominate; one pkl.gz per frame at ~100–500 KB. |

If 16 GB ever turns tight, swap SAM-H for `mobile_sam.pt` (EfficientSAM path, ~1 GB) or `sam_hq_vit_tiny.pth`.

## 2. Conda env

```bash
conda create -n conceptgraph anaconda python=3.10 -y
conda activate conceptgraph

pip install tyro open_clip_torch wandb h5py openai hydra-core distinctipy ultralytics
conda install -c pytorch faiss-cpu=1.7.4 mkl=2021 blas=1.0=mkl -y

# Pytorch 2.0.1 + CUDA 11.8 (what the repo is tested against).
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
    pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Pytorch3D 0.7.4 prebuilt wheel matching the torch above.
conda install \
    https://anaconda.org/pytorch3d/pytorch3d/0.7.4/download/linux-64/pytorch3d-0.7.4-py310_cu118_pyt201.tar.bz2 -y

# gradslam + chamferdist (clone *outside* the concept-graphs folder).
cd ..
git clone https://github.com/krrish94/chamferdist.git && pip install ./chamferdist
git clone https://github.com/gradslam/gradslam.git && cd gradslam && git checkout conceptfusion && pip install . && cd ..

# Extras the benchmark adds on top of the original repo:
pip install loguru scipy imageio
```

`scipy` is the only new hard dep introduced by `benchmark_tracking.py` (Hungarian matcher). `loguru` is used for per-frame logs. `cv2` + `tqdm` + `omegaconf` are already pulled in by ConceptGraphs.

## 3. Grounded-SAM + checkpoints

The detection stage is unchanged from the upstream repo. Follow the README's
[Grounded-SAM section](README.md#install-grounded-sam-package). The short
version:

```bash
git clone git@github.com:IDEA-Research/Grounded-Segment-Anything.git
cd Grounded-Segment-Anything
# install per their README; pin to commit a4d76a2b55e3 for parity.

# Download the three checkpoints into Grounded-Segment-Anything/:
#   ram_swin_large_14m.pth
#   groundingdino_swint_ogc.pth
#   sam_vit_h_4b8939.pth
```

Export the path before running anything:

```bash
export GSA_PATH=/abs/path/to/Grounded-Segment-Anything
```

You do **not** need LLaVA for benchmarking (scene-graph captions are unused
by the tracker).

## 4. Install this repo

```bash
cd /path/to/concept-graphs
pip install -e .
```

## 5. Data layout

```
ISAAC_ROOT/
├── scene_001/
│   ├── rgb/        frame000001.jpg  ...
│   ├── depth/      depth000001.png  ...   # 16-bit, depth_mm = pixel value
│   ├── bbox/       bboxes000001_info.json ...
│   ├── seg/        semantic000001.png + semantic000001_info.json ...
│   └── traj.txt
├── scene_002/...
└── ...
```

Verify the dataset YAML matches your camera intrinsics before launching:

```bash
$EDITOR conceptgraph/dataset/dataconfigs/isaacsim/isaacsim.yaml
# image_height/width, fx, fy, cx, cy, png_depth_scale (usually 1000 for IsaacSim mm depth)
```

## 6. Stage A — detections per scene (offline, GPU-heavy)

The benchmark assumes detection has already been done. Loop over scenes:

```bash
export ISAAC_ROOT=/path/to/IsaacSimData
export CG_FOLDER=/path/to/concept-graphs
cd $CG_FOLDER

for SCENE in $(ls $ISAAC_ROOT); do
    [ -d "$ISAAC_ROOT/$SCENE/rgb" ] || continue
    python conceptgraph/scripts/generate_gsa_results.py \
        --dataset_root $ISAAC_ROOT \
        --dataset_config conceptgraph/dataset/dataconfigs/isaacsim/isaacsim.yaml \
        --scene_id $SCENE \
        --class_set ram \
        --box_threshold 0.2 --text_threshold 0.2 \
        --stride 1 --add_bg_classes --accumu_classes \
        --exp_suffix withbg_allclasses
done
```

This produces, per scene:

* `gsa_detections_ram_withbg_allclasses/frame######.pkl.gz`
* `gsa_classes_ram_withbg_allclasses.json`
* `gsa_vis_ram_withbg_allclasses/` (debug overlays you can ignore)

VRAM during this stage: ~9–11 GB peak with SAM-H. Lower with `--sam_variant mobile`.

## 7. Stage B — benchmark all scenes

```bash
python benchmark_tracking.py \
    --dataset_root $ISAAC_ROOT \
    --all_scenes \
    --gsa_variant ram_withbg_allclasses \
    --iou_threshold 0.25 \
    --matcher hungarian
```

Useful flags:

| Flag | Effect |
|---|---|
| `--print_every_frame` | Log a one-line summary every frame (GT/Pred/Matches/FN/FP/mIoU). |
| `--limit N` | Cap frames per scene (smoke test). |
| `--stride S` | Skip frames (S=5 ≈ 5× faster, comparable to the README defaults). |
| `--no-save_masks` | Skip writing per-track PNG masks (smaller output, no visualisation). |
| `--save_pcd` / `--save_final_objects` | Dump finalised 3D map per scene. |
| `--scene_id <name>` (no `--all_scenes`) | Run a single scene. |

Output (timestamped under `benchmark_results/`):

```
benchmark_results/20260512_140000/
├── scene_001.json                       # per-scene metrics
├── scene_001/
│   ├── predictions_per_frame.pkl.gz     # consumed by the visualiser
│   └── masks/frame######_t<tid>.png     # one PNG per (frame, track)
├── scene_002.json
├── scene_002/...
├── _macro_avg_all_scenes.json           # macro mean across scenes
└── _failures.json                       # any scene that crashed
```

VRAM during this stage: ~1–2 GB. Wall time: roughly the same as
`cfslam_pipeline_batch.py` would take on the same data (detection is
already cached on disk).

## 8. Stage C — visualise tracking

```bash
python visualize_tracking.py \
    --benchmark_run benchmark_results/20260512_140000 \
    --scene_id scene_001
# → benchmark_results/20260512_140000/scene_001/tracking.mp4
```

Options:

| Flag | Default | Effect |
|---|---|---|
| `--alpha` | 0.5 | Mask blend factor (0=invisible, 1=opaque). |
| `--fps` | 15 | Output frame rate. |
| `--limit N` | — | Cap rendered frames. |
| `--output PATH` | `<scene_dir>/tracking.mp4` | Custom output path. |

Each predicted track gets a stable BGR colour derived from its id; if the
pipeline reassigns an object to a new id, the colour changes. The track id
is printed at the mask centroid.

## 9. Troubleshooting

* **`FileNotFoundError: Detection file ... missing.`** → Stage A wasn't run
  for that scene, or `--gsa_variant` doesn't match the `--exp_suffix` used
  in Stage A (`ram_withbg_allclasses` ↔ `ram_withbg_allclasses`).
* **`traj.txt has N poses but M RGB frames`** → trajectory length must
  equal the number of `frame*.jpg`s. Check stride/sub-sampling.
* **All `MOTA` come out very negative** → IoU threshold may be too strict
  for your scenes; try `--iou_threshold 0.1` first to sanity-check matching
  is finding *any* GT-Pred overlap, then tighten.
* **OOM during Stage A** → switch to `--sam_variant mobile` (or
  `light_hqsam`) in `generate_gsa_results.py`.
* **Wrong intrinsics** → check `dataconfigs/isaacsim/isaacsim.yaml`. The
  defaults are for 1280×720 at ~70° HFOV; your IsaacSim camera prim may
  use different fx/fy.
