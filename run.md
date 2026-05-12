generate_gsa_results.py per scene → gsa_detections_<variant>/.
benchmark_tracking.py --dataset_root <root> --all_scenes --gsa_variant ram_withbg_allclasses → metrics + sidecar.
visualize_tracking.py --benchmark_run <ts_dir> --scene_id <scene> → tracking.mp4.