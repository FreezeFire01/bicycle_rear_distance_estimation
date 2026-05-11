import pandas as pd
from pathlib import Path
import re

BASE = Path('/BP1')
DATASET = BASE / 'garmin_dataset_v2'
OUT_CSV = BASE / 'garmin_dataset_v2' / 'rf_dataset.csv'

MAX_DISTANCE = 145


def main():
    rows = []

    for split in ['train', 'val', 'test']:
        labels_dir = DATASET / split / 'labels'
        if not labels_dir.exists():
            continue

        for lbl in labels_dir.glob('*.txt'):
            with open(lbl) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 6:
                        continue
                    try:
                        cls = int(parts[0])
                        cx = float(parts[1])
                        cy = float(parts[2])
                        w = float(parts[3])
                        h = float(parts[4])
                        dist_norm = float(parts[5])
                        dist_m = dist_norm * MAX_DISTANCE
                    except ValueError:
                        continue

                    rows.append({
                        'image': lbl.stem + '.jpg',
                        'cls': cls,
                        'cx': cx,
                        'cy': cy,
                        'w': w,
                        'h': h,
                        'area': w * h,
                        'aspect_ratio': w / (h + 1e-6),
                        'y_bottom': cy + h / 2,
                        'distance_m': dist_m,
                        'split': split,
                    })

    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    print(f'✅ Saved: {OUT_CSV}')
    print(f'   Total rows: {len(df)}')
    print(f'   Train: {(df.split == "train").sum()}')
    print(f'   Val:   {(df.split == "val").sum()}')
    print(f'   Test:  {(df.split == "test").sum()}')
    print(f'   Distance range: {df.distance_m.min():.1f}m - {df.distance_m.max():.1f}m')


if __name__ == '__main__':
    main()