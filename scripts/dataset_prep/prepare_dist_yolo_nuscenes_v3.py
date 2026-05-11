#!/usr/bin/env python3
"""
prepare_dist_yolo_nuscenes_v5.py
=================================
Príprava nuScenes pre Dist-YOLO s konzistentnými triedami ako Garmin.

Class mapping (nuScenes → Garmin):
  0 (car)        → 0 (car)
  1 (truck)      → 1 (truck)
  2 (bus)        → 1 (truck)         # zlúčené - podobný profil bboxu
  3 (motorcycle) → vyhodené          # úplne iný profil

Splity: train/val/test (75/15/10 ekvivalent — train ako pôvodný, val/test 50/50 split)
"""

import os
import shutil
import hashlib
import pandas as pd
from pathlib import Path

BASE = Path('/BP1')
NUSCENES_SRC = BASE / 'bicycle_safety_dataset_final'
NUSCENES_DST = BASE / 'nuscenes_dist_dataset'
META_CSV = NUSCENES_SRC / 'distance_regression_meta.csv'

MAX_DISTANCE = 90.0

# Mapping: pôvodná trieda → nová trieda. None = vyhodiť.
CLASS_REMAP = {
    0: 0,      # car → car
    1: 1,      # truck → truck
    2: 1,      # bus → truck (zlúčené — podobný profil)
    3: None,   # motorcycle → vyhodené
}
NEW_CLASS_NAMES = {0: 'car', 1: 'truck'}
ORIG_NAMES = {0: 'car', 1: 'truck', 2: 'bus', 3: 'motorcycle'}


def stable_split(image_id: str) -> str:
    h = int(hashlib.md5(image_id.encode()).hexdigest(), 16)
    return 'val' if h % 2 == 0 else 'test'


def link_or_copy(src: Path, dst: Path):
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main():
    print("=" * 60)
    print(f"NUSCENES DIST-YOLO PREP v5")
    print(f"  MAX_DISTANCE: {MAX_DISTANCE} m")
    print(f"  Class remap:")
    for old, new in CLASS_REMAP.items():
        old_name = ORIG_NAMES.get(old, f'cls{old}')
        if new is None:
            print(f"    {old} ({old_name}) → DROP")
        else:
            new_name = NEW_CLASS_NAMES[new]
            note = ' (zlúčené)' if old != new else ''
            print(f"    {old} ({old_name}) → {new} ({new_name}){note}")
    print("=" * 60)

    if NUSCENES_DST.exists():
        print(f"\n⚠️  Mažem {NUSCENES_DST}")
        shutil.rmtree(NUSCENES_DST)
    NUSCENES_DST.mkdir(parents=True)

    # CSV
    print(f"\n[1/3] Načítavam {META_CSV.name}")
    df = pd.read_csv(META_CSV)
    print(f"  CSV rows: {len(df)}")
    dist_map = dict(zip(df['image_id'], df['distance_m']))

    for split in ('train', 'val', 'test'):
        (NUSCENES_DST / split / 'images').mkdir(parents=True, exist_ok=True)
        (NUSCENES_DST / split / 'labels').mkdir(parents=True, exist_ok=True)

    # Class distribution check
    print(f"\n[2/3] Distribúcia tried v zdrojovom datasete")
    cls_count = {}
    for src_split in ('train', 'val'):
        for lbl in (NUSCENES_SRC / src_split / 'labels').glob('*.txt'):
            with open(lbl) as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        c = int(parts[0])
                        cls_count[c] = cls_count.get(c, 0) + 1
    total = sum(cls_count.values())
    for c in sorted(cls_count):
        name = ORIG_NAMES.get(c, f'cls{c}')
        new = CLASS_REMAP.get(c)
        if new is None:
            kept = '❌ DROP'
        elif new == c:
            kept = f'✅ → {new}'
        else:
            kept = f'✅ → {new} (zlúčené)'
        print(f"    cls {c} ({name}): {cls_count[c]:>5} ({100*cls_count[c]/total:5.1f}%)  {kept}")

    n_kept = sum(cls_count.get(c, 0) for c in CLASS_REMAP if CLASS_REMAP[c] is not None)
    print(f"  → Po mappingu zachovaných: {n_kept} bboxov ({100*n_kept/total:.1f}%)")

    # Konvertuj
    print(f"\n[3/3] Konverzia s class remap + dist filter...")
    stats = {'train': 0, 'val': 0, 'test': 0,
             'no_csv_match': 0, 'over_max_distance': 0,
             'class_dropped': 0, 'class_remapped': 0,
             'bbox_total': 0, 'bbox_per_new_class': {0: 0, 1: 0}}

    for src_split in ('train', 'val'):
        src_img_dir = NUSCENES_SRC / src_split / 'images'
        src_lbl_dir = NUSCENES_SRC / src_split / 'labels'

        if not src_lbl_dir.exists():
            continue

        label_files = sorted(src_lbl_dir.glob('*.txt'))
        print(f"\n  Source split '{src_split}': {len(label_files)} labelov")

        for lbl_file in label_files:
            img_id = lbl_file.stem

            if img_id not in dist_map:
                stats['no_csv_match'] += 1
                continue

            dist_m = dist_map[img_id]
            if dist_m > MAX_DISTANCE or dist_m < 3.0:
                stats['over_max_distance'] += 1
                continue

            dist_norm = dist_m / MAX_DISTANCE

            with open(lbl_file) as f:
                lines = [l.strip() for l in f if l.strip()]

            new_lines = []
            for line in lines:
                parts = line.split()
                if len(parts) < 5:
                    continue
                old_cls = int(parts[0])
                new_cls = CLASS_REMAP.get(old_cls)
                if new_cls is None:
                    stats['class_dropped'] += 1
                    continue
                if new_cls != old_cls:
                    stats['class_remapped'] += 1
                cx, cy, w, h = parts[1:5]
                new_lines.append(f"{new_cls} {cx} {cy} {w} {h} {dist_norm:.6f}")
                stats['bbox_total'] += 1
                stats['bbox_per_new_class'][new_cls] += 1

            if not new_lines:
                continue

            target = 'train' if src_split == 'train' else stable_split(img_id)

            out_lbl = NUSCENES_DST / target / 'labels' / lbl_file.name
            out_lbl.write_text('\n'.join(new_lines) + '\n')

            for ext in ('.jpg', '.png', '.jpeg'):
                src_img = src_img_dir / (img_id + ext)
                if src_img.exists():
                    out_img = NUSCENES_DST / target / 'images' / src_img.name
                    link_or_copy(src_img, out_img)
                    break

            stats[target] += 1

    # Súhrn
    print("\n" + "=" * 60)
    print("VÝSLEDKY")
    print("=" * 60)
    total_imgs = stats['train'] + stats['val'] + stats['test']
    print(f"  Train: {stats['train']:>6} obrázkov")
    print(f"  Val:   {stats['val']:>6} obrázkov")
    print(f"  Test:  {stats['test']:>6} obrázkov")
    print(f"  Total: {total_imgs:>6} obrázkov, {stats['bbox_total']} bboxov")
    print(f"\n  Po triedach (po remappingu):")
    bp = stats['bbox_per_new_class']
    if stats['bbox_total'] > 0:
        for c, n in bp.items():
            print(f"    {c} ({NEW_CLASS_NAMES[c]}): {n} ({100*n/stats['bbox_total']:.1f}%)")
    print(f"\n  Skipped:")
    print(f"    Bez CSV matchu:           {stats['no_csv_match']}")
    print(f"    Vzdialenosť > {MAX_DISTANCE}m:    {stats['over_max_distance']}")
    print(f"    Trieda dropped (motorbike): {stats['class_dropped']}")
    print(f"    Trieda remap (bus → truck): {stats['class_remapped']}")

    # YAML
    yaml_text = f"""# nuScenes Dist-YOLO dataset
# Class remap: 0=car, 1=truck (bus zlúčený do truck, motorcycle vyhodený)
path: {NUSCENES_DST}
train: train/images
val: val/images
test: test/images

nc: 2
names:
  0: car
  1: truck

max_distance_m: {MAX_DISTANCE}
"""
    yaml_path = NUSCENES_DST / 'dataset.yaml'
    yaml_path.write_text(yaml_text)
    print(f"\n  YAML: {yaml_path}")

    sample = next((NUSCENES_DST / 'train' / 'labels').glob('*.txt'), None)
    if sample:
        print(f"\n[OVERENIE] Sample {sample.name}:")
        print(f"  {sample.read_text().strip()[:200]}")

    print("\n" + "=" * 60)
    print("✅ HOTOVO. Ďalší krok:")
    print(f"   python train_dist_yolo_nuscenes.py --data {yaml_path}")
    print("=" * 60)


if __name__ == '__main__':
    main()