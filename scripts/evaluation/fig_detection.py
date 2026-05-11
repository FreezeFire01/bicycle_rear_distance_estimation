"""
Vyrobí fig:detection — 2x2 mriežku 4 ukážok z Garmin datasetu
s ohraničujúcimi rámikmi a referenčnými vzdialenosťami z radaru.
"""
import argparse
import random
from pathlib import Path
import cv2
import numpy as np

BASE = Path('/BP1')
GARMIN_DIST_DATASET = BASE / 'garmin_dist_dataset'
MAX_DISTANCE = 90.0
TARGET_BANDS = [(5, 15), (20, 35), (45, 60), (70, 90)]

def color_for_distance(d_m):
    if d_m < 15:
        return (0, 0, 255)         # červená
    elif d_m < 30:
        return (0, 165, 255)       # oranžová
    elif d_m < 50:
        return (0, 255, 255)       # žltá
    else:
        return (0, 200, 0)         # zelená


def find_samples_in_bands(bands):
    candidates = {i: [] for i in range(len(bands))}

    for split in ('train', 'val', 'test'):
        labels_dir = GARMIN_DIST_DATASET / split / 'labels'
        images_dir = GARMIN_DIST_DATASET / split / 'images'

        if not labels_dir.exists():
            continue

        for lbl_file in labels_dir.glob('*.txt'):
            img_file = None
            for ext in ('.jpg', '.png', '.jpeg'):
                cand = images_dir / (lbl_file.stem + ext)
                if cand.exists():
                    img_file = cand
                    break
            if img_file is None:
                continue

            with open(lbl_file) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 6:
                        continue
                    try:
                        cls = int(parts[0])
                        cx, cy, w, h = map(float, parts[1:5])
                        dist = float(parts[5]) * MAX_DISTANCE
                    except ValueError:
                        continue

                    for i, (lo, hi) in enumerate(bands):
                        if lo <= dist < hi:
                            candidates[i].append({
                                'image': img_file,
                                'cls': cls,
                                'cx': cx, 'cy': cy,
                                'w': w, 'h': h,
                                'distance_m': dist,
                            })

    return candidates


def draw_bbox_with_label(img, sample):
    H, W = img.shape[:2]
    cx, cy = sample['cx'], sample['cy']
    w, h = sample['w'], sample['h']
    d_m = sample['distance_m']
    cls = sample['cls']

    x1 = int((cx - w / 2) * W)
    y1 = int((cy - h / 2) * H)
    x2 = int((cx + w / 2) * W)
    y2 = int((cy + h / 2) * H)

    color = color_for_distance(d_m)

    # bbox
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 4)

    cls_name = ['car', 'truck'][cls] if cls < 2 else f'cls{cls}'
    label = f"{cls_name}: {d_m:.1f} m"

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.2
    thickness = 3

    (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)

    bg_y1 = max(y1 - th - 15, 0)
    bg_y2 = y1
    bg_x2 = min(x1 + tw + 12, W)
    cv2.rectangle(img, (x1, bg_y1), (bg_x2, bg_y2), color, -1)

    cv2.putText(img, label, (x1 + 6, y1 - 8), font, font_scale, (0, 0, 0), thickness)

    return img


def add_panel_label(img, text, height=70):
    canvas = np.ones((height, img.shape[1], 3), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.5
    thickness = 3
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    x = (img.shape[1] - tw) // 2
    y = (height + th) // 2
    cv2.putText(canvas, text, (x, y), font, font_scale, (0, 0, 0), thickness)
    return np.vstack([img, canvas])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='/home/jozef/Documents/FIIT/5thSemester/BP1/figures_for_bp/fig_detection.png')
    parser.add_argument('--target-width', type=int, default=900,
                        help='cielova sirka jedneho panelu')
    parser.add_argument('--seed', type=int, default=67)
    args = parser.parse_args()

    random.seed(args.seed)
    candidates = find_samples_in_bands(TARGET_BANDS)

    for i, (lo, hi) in enumerate(TARGET_BANDS):
        print(f"  Pásmo {lo}-{hi}m: {len(candidates[i])} kandidátov")
        if len(candidates[i]) == 0:
            print(f"Žiadne vzorky v pásme {lo}-{hi}m!")
            return

    selected = []
    for i in range(4):
        cands = candidates[i]
        cands_sorted = sorted(cands, key=lambda s: abs(s['cx'] - 0.5))
        top20 = cands_sorted[:min(20, len(cands_sorted))]
        sample = random.choice(top20)
        selected.append(sample)
        print(f"  Vybraté pre pásmo {TARGET_BANDS[i]}: {sample['image'].name} "
              f"(d={sample['distance_m']:.1f}m)")

    annotated = []
    for sample in selected:
        img = cv2.imread(str(sample['image']))
        if img is None:
            print(f"Nepodarilo sa načítať {sample['image']}")
            return
        img = draw_bbox_with_label(img, sample)

        # Resize na target width
        h, w = img.shape[:2]
        new_h = int(h * args.target_width / w)
        img_resized = cv2.resize(img, (args.target_width, new_h),
                                  interpolation=cv2.INTER_AREA)
        annotated.append(img_resized)

    max_h = max(img.shape[0] for img in annotated)
    for i in range(len(annotated)):
        if annotated[i].shape[0] < max_h:
            pad = max_h - annotated[i].shape[0]
            annotated[i] = cv2.copyMakeBorder(annotated[i], 0, pad, 0, 0,
                                                cv2.BORDER_CONSTANT, value=(255, 255, 255))

    # Pridaj label pod každý panel
    labeled = annotated

    # Vytvor 2x2 grid
    gap = 5
    gap_color = 255

    h_panel = labeled[0].shape[0]
    w_panel = labeled[0].shape[1]

    top_row = np.hstack([
        labeled[0],
        np.full((h_panel, gap, 3), gap_color, dtype=np.uint8),
        labeled[1]
    ])
    bot_row = np.hstack([
        labeled[2],
        np.full((h_panel, gap, 3), gap_color, dtype=np.uint8),
        labeled[3]
    ])

    middle_gap = np.full((gap, top_row.shape[1], 3), gap_color, dtype=np.uint8)
    final = np.vstack([top_row, middle_gap, bot_row])

    cv2.imwrite(args.output, final)
    print(f"\nSaved: {args.output}")
    print(f"Rozmery: {final.shape[1]} × {final.shape[0]}")


if __name__ == '__main__':
    main()