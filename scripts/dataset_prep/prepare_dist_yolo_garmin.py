"""
Dist-YOLO Data Preparation
===========================
Konvertuje Garmin YOLO dataset na formát pre Dist-YOLO:
  Pôvodne: class_id cx cy w h
  Nový:    class_id cx cy w h distance_normalized
"""

import os
import pandas as pd
import shutil
from pathlib import Path

BASE = Path('/BP1')

GARMIN_SRC = BASE / 'garmin_dataset'
GARMIN_DST = BASE / 'garmin_dist_dataset'
MAX_DISTANCE = 90.0


def build_bbox_distance_map(rf_csv_path):
    df = pd.read_csv(rf_csv_path)
    print(f"  rf_dataset.csv: {len(df)} riadkov")

    bbox_map = {}
    for _, row in df.iterrows():
        key = (
            row['image'],
            round(float(row['cx']), 4),
            round(float(row['cy']), 4),
            round(float(row['w']), 4),
            round(float(row['h']), 4),
        )
        bbox_map[key] = float(row['distance_m'])

    print(f"  Unique bbox-distance pairs: {len(bbox_map)}")
    return bbox_map


def convert_labels(src_split_dir, dst_split_dir, bbox_map):
    src_lbl = src_split_dir / 'labels'
    src_img = src_split_dir / 'images'
    dst_lbl = dst_split_dir / 'labels'
    dst_img = dst_split_dir / 'images'

    dst_lbl.mkdir(parents=True, exist_ok=True)
    dst_img.mkdir(parents=True, exist_ok=True)

    n_images = 0
    for img_file in src_img.glob('*.jpg'):
        dst_path = dst_img / img_file.name
        if dst_path.exists():
            dst_path.unlink()
        dst_path.symlink_to(img_file)
        n_images += 1

    n_total = 0
    n_with_dist = 0
    n_clamped = 0
    n_skipped = 0

    for lbl_file in src_lbl.glob('*.txt'):
        img_name = lbl_file.stem + '.jpg'
        out_file = dst_lbl / lbl_file.name

        with open(lbl_file) as f:
            lines = [l.strip() for l in f if l.strip()]

        new_lines = []
        for line in lines:
            parts = line.split()
            if len(parts) < 5:
                continue
            cls = parts[0]
            cx, cy, w, h = [float(x) for x in parts[1:5]]

            n_total += 1

            key = (img_name, round(cx, 4), round(cy, 4), round(w, 4), round(h, 4))
            if key in bbox_map:
                dist_m = bbox_map[key]

                if dist_m > MAX_DISTANCE:
                    n_clamped += 1
                    continue

                dist_norm = dist_m / MAX_DISTANCE
                new_lines.append(
                    f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {dist_norm:.6f}"
                )
                n_with_dist += 1
            else:
                n_skipped += 1

        if new_lines:
            with open(out_file, 'w') as f:
                f.write('\n'.join(new_lines) + '\n')
        elif out_file.exists():
            out_file.unlink()

    return {
        'images': n_images,
        'labels_total': n_total,
        'labels_with_distance': n_with_dist,
        'labels_clamped': n_clamped,
        'labels_skipped': n_skipped,
    }


def main():
    print("=" * 60)
    print(f"Dist-YOLO Data Preparation (MAX_DISTANCE = {MAX_DISTANCE}m)")
    print("=" * 60)

    if GARMIN_DST.exists():
        print(f"\n⚠️  Maže sa existujúci {GARMIN_DST}")
        shutil.rmtree(GARMIN_DST)
    GARMIN_DST.mkdir(parents=True)

    print(f"\n[1/2] Načítavam rf_dataset.csv...")
    bbox_map = build_bbox_distance_map(GARMIN_SRC / 'rf_dataset.csv')

    print(f"\n[2/2] Konvertujem splity...")
    total_stats = {}
    for split in ['train', 'val', 'test']:
        print(f"\n  Split: {split}")
        src = GARMIN_SRC / split
        dst = GARMIN_DST / split

        if not src.exists():
            print(f"    ⚠️ {src} neexistuje, preskakujem")
            continue

        stats = convert_labels(src, dst, bbox_map)
        print(f"    Images: {stats['images']}")
        print(f"    Labels s distance: {stats['labels_with_distance']}/{stats['labels_total']}")
        print(f"    Clamped (>{MAX_DISTANCE}m): {stats['labels_clamped']}")
        print(f"    Bez match: {stats['labels_skipped']}")
        total_stats[split] = stats

    yaml_path = GARMIN_DST / 'dataset.yaml'
    with open(yaml_path, 'w') as f:
        f.write(f"""# Dist-YOLO Garmin dataset
path: {GARMIN_DST}
train: train/images
val: val/images
test: test/images

nc: 2
names:
  0: car
  1: truck

max_distance_m: {MAX_DISTANCE}
""")

    print(f"\n{'=' * 60}")
    print("SÚHRN")
    print(f"{'=' * 60}")
    print(f"Output: {GARMIN_DST}")
    print(f"YAML:   {yaml_path}")
    for split, stats in total_stats.items():
        total = stats['labels_total']
        with_d = stats['labels_with_distance']
        if total > 0:
            pct = 100 * with_d / total
            print(f"  {split}: {with_d}/{total} labelov s distance ({pct:.1f}%)")

    print(f"\n[OVERENIE] Príklad labelu:")
    sample_label = next((GARMIN_DST / 'train' / 'labels').glob('*.txt'), None)
    if sample_label:
        print(f"  Súbor: {sample_label.name}")
        print(f"  Obsah:")
        with open(sample_label) as f:
            for line in f.read().strip().split('\n')[:3]:
                print(f"    {line}")



if __name__ == '__main__':
    main()