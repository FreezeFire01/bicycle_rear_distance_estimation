#!/usr/bin/env python3
"""
Prepare Wave 2 for Label Studio
================================
1. Skopíruje len wave2 obrázky do samostatného priečinka
2. Skonvertuje YOLO labely na Label Studio JSON pre pre-anotácie
3. Vypíše návod na import

Použitie:
  python prepare_wave.py \
    --metadata metadata.csv \
    --labels-dir labels/ \
    --images-dir images/ \
    --output-dir ./wave2_for_labelstudio

Potom v Label Studio:
  1. Import obrázkov z wave2_for_labelstudio/images/
  2. Import pre-anotácií z wave2_for_labelstudio/preannotations.json
"""

import argparse
import csv
import json
import os
import shutil
from pathlib import Path


# Tvoje triedy — uprav ak máš iné
CLASS_NAMES = {0: 'car', 1: 'truck'}


def count_labels(txt_path):
    if not os.path.exists(txt_path):
        return 0
    with open(txt_path) as f:
        return len([l for l in f if l.strip()])


def parse_yolo_labels(txt_path, img_width=1920, img_height=1080):
    """Parse YOLO txt → list of (class_id, x_center, y_center, width, height) normalized."""
    labels = []
    if not os.path.exists(txt_path):
        return labels
    with open(txt_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls = int(parts[0])
                x, y, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                labels.append((cls, x, y, w, h))
    return labels


def yolo_to_labelstudio(image_name, yolo_labels, class_names):
    """Convert YOLO labels to Label Studio JSON format."""
    results = []
    for cls_id, xc, yc, w, h in yolo_labels:
        # YOLO: normalized center x,y,w,h → Label Studio: x,y,w,h as % from top-left
        x_pct = (xc - w/2) * 100
        y_pct = (yc - h/2) * 100
        w_pct = w * 100
        h_pct = h * 100

        cls_name = class_names.get(cls_id, f'class_{cls_id}')

        results.append({
            "from_name": "label",
            "to_name": "image",
            "type": "rectanglelabels",
            "value": {
                "x": x_pct,
                "y": y_pct,
                "width": w_pct,
                "height": h_pct,
                "rotation": 0,
                "rectanglelabels": [cls_name]
            }
        })
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata', required=True)
    parser.add_argument('--labels-dir', required=True)
    parser.add_argument('--images-dir', required=True)
    parser.add_argument('--output-dir', default='./wave2_for_labelstudio')
    args = parser.parse_args()

    # Parse metadata
    with open(args.metadata, 'r') as f:
        reader = csv.DictReader(f)
        meta = {row['image']: row for row in reader}
        meta = {k: v for k, v in meta.items() if k.startswith('20260328')}

    labels_dir = Path(args.labels_dir)
    images_dir = Path(args.images_dir)

    # Find wave2 frames: radar>0, label_count != radar_count
    wave2 = []
    for image_name, row in meta.items():
        label_file = labels_dir / image_name.replace('.jpg', '.txt')
        num_labels = count_labels(label_file)
        num_radar = int(row['num_cars'])

        if num_radar > 0 and num_labels != num_radar:
            wave2.append((image_name, num_labels, num_radar, row))

    print(f"Wave 2: {len(wave2)} framov na kontrolu")

    # Create output dirs
    out_images = os.path.join(args.output_dir, 'images')
    os.makedirs(out_images, exist_ok=True)

    # Copy images + build Label Studio JSON
    ls_tasks = []

    for image_name, num_labels, num_radar, row in wave2:
        # Copy image
        src = images_dir / image_name
        if not src.exists():
            continue
        shutil.copy2(src, os.path.join(out_images, image_name))

        # Parse YOLO labels
        label_file = labels_dir / image_name.replace('.jpg', '.txt')
        yolo_labels = parse_yolo_labels(label_file)

        # Convert to Label Studio format
        results = yolo_to_labelstudio(image_name, yolo_labels, CLASS_NAMES)

        # Build task with metadata in description for reference
        closest = row['closest_m']
        ranges = row['all_ranges']

        task = {
            "data": {
                "image": f"/data/local-files/?d=wave2_images/{image_name}"
            },
            "annotations": [{
                "result": results
            }] if results else [],
            "meta": {
                "radar_num_cars": num_radar,
                "label_count": num_labels,
                "closest_m": closest,
                "all_ranges": ranges,
                "note": f"Radar: {num_radar} áut ({ranges}), YOLO: {num_labels} labelov"
            }
        }
        ls_tasks.append(task)

    # Write Label Studio JSON
    json_path = os.path.join(args.output_dir, 'preannotations.json')
    with open(json_path, 'w') as f:
        json.dump(ls_tasks, f, indent=2, ensure_ascii=False)

    # Write simple list
    list_path = os.path.join(args.output_dir, 'wave2_list.txt')
    with open(list_path, 'w') as f:
        for img, nl, nr, row in wave2:
            f.write(f"{img}  labels={nl}  radar={nr}  ranges={row['all_ranges']}\n")

    print(f"\nVýstup:")
    print(f"  Obrázky:        {out_images}/ ({len(wave2)} súborov)")
    print(f"  Pre-anotácie:   {json_path}")
    print(f"  Zoznam:         {list_path}")

if __name__ == '__main__':
    main()