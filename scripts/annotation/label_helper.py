#!/usr/bin/env python3
"""
Label Cleaning Helper
=====================
Porovná YOLO labely s metadata.csv a rozdelí framy do kategórií
na efektívne manuálne čistenie.

Použitie:
  python label_helper.py --metadata metadata.csv --labels-dir labels/ --images-dir images/

Výstup:
  cleaning_report.txt   ← prehľad čo treba skontrolovať
  wave1_delete_all/     ← framy kde radar=0 ale YOLO niečo detegoval (zmazať všetky labely)
  wave2_check/          ← framy kde počet labelov != počet radar áut (skontrolovať)
  wave3_ok/             ← framy kde všetko sedí (len rýchly scan)
"""

import argparse
import csv
import os
import shutil
from pathlib import Path
from collections import defaultdict


def count_labels_in_file(txt_path):
    """Počet bounding boxov v YOLO txt súbore."""
    if not os.path.exists(txt_path):
        return 0
    with open(txt_path) as f:
        lines = [l.strip() for l in f if l.strip()]
    return len(lines)


def main():
    parser = argparse.ArgumentParser(description='Label Cleaning Helper')
    parser.add_argument('--metadata', required=True, help='metadata.csv')
    parser.add_argument('--labels-dir', required=True, help='Priečinok s YOLO .txt labelmi')
    parser.add_argument('--images-dir', required=True, help='Priečinok s obrázkami')
    parser.add_argument('--output-dir', default='./cleaning_report', help='Výstup')
    parser.add_argument('--auto-clean-wave1', action='store_true',
                        help='Automaticky vymaž labely kde radar=0 (Wave 1)')
    args = parser.parse_args()

    # Parse metadata
    with open(args.metadata, 'r') as f:
        reader = csv.DictReader(f)
        meta = {row['image']: row for row in reader}

    print(f"Metadata: {len(meta)} framov")

    # Analyze each frame
    wave1 = []  # radar=0, labels>0 → DELETE ALL labels
    wave2 = []  # radar>0, label_count != radar_count → CHECK
    wave3 = []  # radar>0, label_count == radar_count → QUICK SCAN
    wave4 = []  # radar>0, labels=0 → MISSING labels (maybe OK)
    no_labels = []  # radar=0, labels=0 → SKIP (nothing to do)

    labels_dir = Path(args.labels_dir)
    stats = defaultdict(int)

    for image_name, row in sorted(meta.items()):
        label_file = labels_dir / image_name.replace('.jpg', '.txt')
        num_labels = count_labels_in_file(label_file)
        num_radar = int(row['num_cars'])

        if num_radar == 0 and num_labels == 0:
            no_labels.append((image_name, num_labels, num_radar))
            stats['skip_empty'] += 1
        elif num_radar == 0 and num_labels > 0:
            wave1.append((image_name, num_labels, num_radar, row))
            stats['wave1_delete'] += 1
        elif num_radar > 0 and num_labels == 0:
            wave4.append((image_name, num_labels, num_radar, row))
            stats['wave4_missing'] += 1
        elif num_labels != num_radar:
            wave2.append((image_name, num_labels, num_radar, row))
            stats['wave2_check'] += 1
        else:
            wave3.append((image_name, num_labels, num_radar, row))
            stats['wave3_ok'] += 1

    # Write detailed report
    os.makedirs(args.output_dir, exist_ok=True)

    report_path = os.path.join(args.output_dir, 'cleaning_report.txt')
    with open(report_path, 'w') as f:
        f.write("WAVE 1 — ZMAZAŤ VŠETKY LABELY (radar=0, labels>0)\n")
        f.write(f"Počet: {len(wave1)}\n")
        for img, nl, nr, row in wave1:
            f.write(f"  {img}  labels={nl}  radar={nr}\n")

        f.write(f"\n\nWAVE 2 — SKONTROLOVAŤ (labels ≠ radar)\n")
        f.write(f"Počet: {len(wave2)}\n")
        for img, nl, nr, row in wave2:
            closest = row['closest_m']
            ranges = row['all_ranges']
            f.write(f"  {img}  labels={nl}  radar={nr}  closest={closest}m  ranges={ranges}\n")

        f.write(f"\n\nWAVE 3 — RÝCHLY SCAN (labels = radar, pravdepodobne OK)\n")
        f.write(f"Počet: {len(wave3)}\n")
        for img, nl, nr, row in wave3:
            closest = row['closest_m']
            f.write(f"  {img}  labels={nl}  radar={nr}  closest={closest}m\n")

        f.write(f"\n\nWAVE 4 — CHÝBAJÚCE LABELY (radar>0, labels=0)\n")
        f.write(f"Počet: {len(wave4)}\n")
        for img, nl, nr, row in wave4:
            closest = row['closest_m']
            ranges = row['all_ranges']
            f.write(f"  {img}  radar={nr}  closest={closest}m  ranges={ranges}\n")

    print(f"Report: {report_path}")

    # Write wave lists (image names per file for Label Studio filtering)
    for wave_name, wave_data in [('wave1', wave1), ('wave2', wave2), ('wave3', wave3), ('wave4', wave4)]:
        list_path = os.path.join(args.output_dir, f'{wave_name}_images.txt')
        with open(list_path, 'w') as f:
            for img, *_ in wave_data:
                f.write(f"{img}\n")
        print(f"  {wave_name}: {list_path} ({len(wave_data)} images)")

    # Auto-clean Wave 1 if requested
    if args.auto_clean_wave1 and wave1:
        print(f"AUTO-CLEAN WAVE 1: Mažem labely pre {len(wave1)} framov...")
        deleted = 0
        for img, nl, nr, row in wave1:
            label_file = labels_dir / img.replace('.jpg', '.txt')
            if label_file.exists():
                # Don't delete — just empty the file (YOLO treats empty = no objects)
                with open(label_file, 'w') as f:
                    pass  # empty file
                deleted += 1
        print(f"  Vyprázdnených: {deleted} label súborov")


if __name__ == '__main__':
    main()