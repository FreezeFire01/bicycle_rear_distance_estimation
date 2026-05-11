import pandas as pd
import numpy as np
import time
import os
from pathlib import Path

BASE_DIR = Path('/BP1')
YOLO_DIR = BASE_DIR / 'yolo_experiments'
GARMIN_YAML = BASE_DIR / 'garmin_dataset' / 'dataset.yaml'
NUSCENES_YAML = BASE_DIR / 'bicycle_safety_dataset_final' / 'data.yaml'
RESULTS_DIR = BASE_DIR / 'yolo_comparison'
RESULTS_DIR.mkdir(exist_ok=True)

STRATEGIES = ['nuscenes_only', 'garmin_only', 'combined', 'nuscenes_to_garmin']
MODELS = ['yolo11n', 'yolo11s', 'yolov8n', 'yolo26n']
IMGSZ = 832


def evaluate_on_dataset(best_pt_path, dataset_yaml, split='val'):
    from ultralytics import YOLO

    model = YOLO(str(best_pt_path))
    try:
        metrics = model.val(
            data=str(dataset_yaml),
            split=split,
            imgsz=IMGSZ,
            rect=True,
            verbose=False,
            plots=False,
        )
        return {
            'mAP50': float(metrics.box.map50),
            'mAP50-95': float(metrics.box.map),
            'precision': float(metrics.box.mp),
            'recall': float(metrics.box.mr),
            'car_mAP50': float(metrics.box.maps[0]) if len(metrics.box.maps) > 0 else None,
            'truck_mAP50': float(metrics.box.maps[1]) if len(metrics.box.maps) > 1 else None,
        }
    except Exception as e:
        print(f"    ❌ Chyba: {e}")
        return None


def main():
    rows = []

    for strategy in STRATEGIES:
        for model_name in MODELS:
            run_name = f"{strategy}_{model_name}"
            best_pt = YOLO_DIR / run_name / 'weights' / 'best.pt'

            if not best_pt.exists():
                print(f"  ⏭  {run_name} nemá best.pt, preskakujem")
                continue

            garmin_results = evaluate_on_dataset(best_pt, GARMIN_YAML, split='test')
            nu_results = evaluate_on_dataset(best_pt, NUSCENES_YAML, split='val')

            row = {
                'strategy': strategy,
                'model': model_name,
            }
            if garmin_results:
                for k, v in garmin_results.items():
                    row[f'garmin_{k}'] = v
                print(
                    f"    Garmin test:   mAP50={garmin_results['mAP50']:.3f}, mAP50-95={garmin_results['mAP50-95']:.3f}")
            if nu_results:
                for k, v in nu_results.items():
                    row[f'nuscenes_{k}'] = v
                print(f"    nuScenes val:  mAP50={nu_results['mAP50']:.3f}, mAP50-95={nu_results['mAP50-95']:.3f}")

            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / 'yolo_crossval.csv', index=False)

if __name__ == '__main__':
    main()