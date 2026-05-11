#!/usr/bin/env python3
"""
YOLO Training Pipeline — Bicycle Safety Dataset

Stratégie:
  1. nuScenes only — trénovanie na automotive datasete
  2. Garmin only — trénovanie len na bicyklovom datasete
  3. Combined — oba datasety dohromady
  4. nuScenes → Garmin finetune — pretrain na nuScenes, finetune na Garmine

Modely: yolo11n, yolo11s, yolov8n
Imgsz: 832×480 (16:9 rectangular)
Epochs: 100

Použitie:
  python train_yolo.py --strategy nuscenes_only --model yolo11n
  python train_yolo.py --strategy combined --model yolo11s
  python train_yolo.py --strategy all   # spustí všetkých 12 experimentov

Výstup:
  runs/detect/{strategy}_{model}/
    weights/best.pt, weights/last.pt
    results.csv, confusion_matrix.png
    per-class metrics
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

# === CESTY ===
BASE_DIR = Path('/BP1')
GARMIN_DIR = BASE_DIR / 'garmin_dataset'
NUSCENES_DIR = BASE_DIR / 'bicycle_safety_dataset_final'
OUTPUT_DIR = BASE_DIR / 'yolo_experiments'

# === TRÉNINGOVÉ PARAMETRE ===
EPOCHS = 100
IMGSZ = 832  # YOLO auto-deteguje rectangular pre 16:9 obrázky
BATCH_SIZE = 16  # auto-adjust podľa VRAM
PATIENCE = 30  # early stopping

MODELS = {
    'yolo11n': 'yolo11n.pt',
    'yolo11s': 'yolo11s.pt',
    'yolov8n': 'yolov8n.pt',
    'yolo26n': 'yolo26n.pt',
}

STRATEGIES = ['nuscenes_only', 'garmin_only', 'combined', 'nuscenes_to_garmin']


def check_datasets():

    issues = []

    # Garmin
    for split in ['train', 'val']:
        img_dir = GARMIN_DIR / split / 'images'
        lbl_dir = GARMIN_DIR / split / 'labels'
        if not img_dir.exists():
            issues.append(f"Chýba {img_dir}")
        else:
            n_img = len(list(img_dir.glob('*.jpg')))
            n_lbl = len(list(lbl_dir.glob('*.txt')))
            print(f"Garmin/{split}: {n_img} obrázkov, {n_lbl} labelov")

    garmin_yaml = GARMIN_DIR / 'dataset.yaml'
    if not garmin_yaml.exists():
        issues.append(f"Chýba Garmin dataset.yaml — spusti split_dataset.py")
    else:
        print(f"Garmin yaml:")

    # nuScenes
    for split in ['train', 'val']:
        img_dir = NUSCENES_DIR / split / 'images'
        lbl_dir = NUSCENES_DIR / split / 'labels'
        if not img_dir.exists():
            issues.append(f"Chýba {img_dir}")
        else:
            n_img = len(list(img_dir.glob('*.jpg')))
            n_lbl = len(list(lbl_dir.glob('*.txt')))
            print(f"  nuScenes/{split}: {n_img} obrázkov, {n_lbl} labelov")

    nu_yaml = NUSCENES_DIR / 'data.yaml'
    if not nu_yaml.exists():
        issues.append(f"Chýba nuScenes data.yaml")
    else:
        print(f"  nuScenes yaml:")

    if issues:
        for i in issues:
            print(f"  ❌ {i}")
        sys.exit(1)
    print()
    return True


def build_combined_dataset():
    combined_dir = BASE_DIR / 'combined_dataset'

    if combined_dir.exists():
        print(f"Combined dataset už existuje: {combined_dir}")
        return combined_dir / 'data.yaml'

    print(f"Vytváram combined dataset v {combined_dir}...")

    for split in ['train', 'val']:
        (combined_dir / split / 'images').mkdir(parents=True, exist_ok=True)
        (combined_dir / split / 'labels').mkdir(parents=True, exist_ok=True)

        # Symlink Garmin
        for img in (GARMIN_DIR / split / 'images').glob('*.jpg'):
            dst = combined_dir / split / 'images' / f'garmin_{img.name}'
            if not dst.exists():
                os.symlink(img, dst)
            lbl_src = GARMIN_DIR / split / 'labels' / img.name.replace('.jpg', '.txt')
            lbl_dst = combined_dir / split / 'labels' / f'garmin_{img.name.replace(".jpg", ".txt")}'
            if lbl_src.exists() and not lbl_dst.exists():
                os.symlink(lbl_src, lbl_dst)

        # Symlink nuScenes
        for img in (NUSCENES_DIR / split / 'images').glob('*.jpg'):
            dst = combined_dir / split / 'images' / f'nuscenes_{img.name}'
            if not dst.exists():
                os.symlink(img, dst)
            lbl_src = NUSCENES_DIR / split / 'labels' / img.name.replace('.jpg', '.txt')
            lbl_dst = combined_dir / split / 'labels' / f'nuscenes_{img.name.replace(".jpg", ".txt")}'
            if lbl_src.exists() and not lbl_dst.exists():
                os.symlink(lbl_src, lbl_dst)

    # yaml
    yaml_path = combined_dir / 'data.yaml'
    yaml_path.write_text(f"""# Combined dataset: Garmin + nuScenes
path: {combined_dir}
train: train/images
val: val/images

names:
  0: car
  1: truck

nc: 2
""")

    # Stats
    for split in ['train', 'val']:
        n = len(list((combined_dir / split / 'images').glob('*.jpg')))
        print(f"  combined/{split}: {n} obrázkov")

    return yaml_path


def train(strategy, model_name):
    """Spusti YOLO tréning pre danú stratégiu a model."""
    from ultralytics import YOLO

    run_name = f"{strategy}_{model_name}"
    output_path = OUTPUT_DIR / run_name

    # Skip ak už je hotový
    best_pt = output_path / 'weights' / 'best.pt'
    if best_pt.exists():
        print(f"\n⏭  {run_name} už existuje, preskakujem ({best_pt})")
        return output_path

    print(f"\n{'=' * 60}")
    print(f"TRÉNING: {run_name}")
    print(f"{'=' * 60}")

    # Vyber dataset yaml
    if strategy == 'nuscenes_only':
        data_yaml = str(NUSCENES_DIR / 'data.yaml')
        pretrained = MODELS[model_name]
    elif strategy == 'garmin_only':
        data_yaml = str(GARMIN_DIR / 'dataset.yaml')
        pretrained = MODELS[model_name]
    elif strategy == 'combined':
        data_yaml = str(build_combined_dataset())
        pretrained = MODELS[model_name]
    elif strategy == 'nuscenes_to_garmin':
        # Použi best.pt z nuscenes_only ako pretrained
        nu_best = OUTPUT_DIR / f'nuscenes_only_{model_name}' / 'weights' / 'best.pt'
        if not nu_best.exists():
            print(f"❌ Najprv natrénuj nuscenes_only_{model_name}")
            return None
        data_yaml = str(GARMIN_DIR / 'dataset.yaml')
        pretrained = str(nu_best)
        print(f"  Pretrained: {pretrained}")
    else:
        raise ValueError(f"Neznáma stratégia: {strategy}")

    print(f"  Dataset: {data_yaml}")
    print(f"  Model: {pretrained}")
    print(f"  Epochs: {EPOCHS}, imgsz: {IMGSZ}, batch: {BATCH_SIZE}")

    # Tréning
    model = YOLO(pretrained)
    results = model.train(
        data=data_yaml,
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH_SIZE,
        patience=PATIENCE,
        project=str(OUTPUT_DIR),
        name=run_name,
        exist_ok=True,
        rect=True,  # rectangular training pre 16:9
        cache='ram',  # rýchlejšie ak sa zmestí
        device=0,  # RTX 5070 Ti
        workers=8,
        verbose=True,
        save=True,
        plots=True,
    )

    # Validácia na test sete (ak existuje)
    print(f"\n  Validácia na test sete...")
    try:
        metrics = model.val(data=data_yaml, split='test', imgsz=IMGSZ, rect=True)
        print(f"  Test mAP@0.5: {metrics.box.map50:.4f}")
        print(f"  Test mAP@0.5:0.95: {metrics.box.map:.4f}")
    except Exception as e:
        print(f"  (Test set nie je k dispozícii alebo zlyhal: {e})")

    print(f"\n✓ Tréning hotový: {output_path}")
    print(f"  best.pt: {output_path}/weights/best.pt")
    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--strategy', choices=STRATEGIES + ['all'], required=True)
    parser.add_argument('--model', choices=list(MODELS.keys()) + ['all'], default='all')
    parser.add_argument('--skip-check', action='store_true')
    args = parser.parse_args()

    if not args.skip_check:
        check_datasets()

    OUTPUT_DIR.mkdir(exist_ok=True)

    strategies = STRATEGIES if args.strategy == 'all' else [args.strategy]
    models = list(MODELS.keys()) if args.model == 'all' else [args.model]

    # Poradie dôležité: nuscenes_only musí ísť pred nuscenes_to_garmin
    if 'all' in [args.strategy]:
        order = ['nuscenes_only', 'garmin_only', 'combined', 'nuscenes_to_garmin']
        strategies = [s for s in order if s in strategies]

    print(f"\nSpúšťam {len(strategies)} × {len(models)} = {len(strategies) * len(models)} experimentov")

    for strategy in strategies:
        for model_name in models:
            try:
                train(strategy, model_name)
            except Exception as e:
                print(f"❌ Chyba pri {strategy}_{model_name}: {e}")


if __name__ == '__main__':
    main()