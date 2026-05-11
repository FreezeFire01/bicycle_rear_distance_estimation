#!/usr/bin/env python3
"""
make_block_split_v3.py
=======================
Spravne block-split pre OBA datasety:

1. garmin_dataset_v2/        - 5-stlpcove labely, PLNY rozsah (3-143m)
   - Zdroj: garmin_dataset/  (povodny YOLO dataset)

2. garmin_dist_dataset_v2/   - 6-stlpcove labely, OREZANE na 90m
   - Zdroj: garmin_dist_dataset/  (povodny Dist-YOLO dataset)

Block-split aplikujeme NA OBA datasety s rovnakou logikou,
ale rozne snimky podla toho co je v zdroji.
"""

import re
import shutil
from pathlib import Path
from collections import defaultdict

BASE = Path('/BP1')

# Zdrojove datasety
SOURCE_YOLO = BASE / 'garmin_dataset'  # 5-stlpec, vsetky auta
SOURCE_DIST = BASE / 'garmin_dist_dataset'  # 6-stlpec, <=90m

# Cielove datasety (block-split)
DEST_YOLO = BASE / 'garmin_dataset_v2'
DEST_DIST = BASE / 'garmin_dist_dataset_v2'

TRAIN_RATIO = 0.75
VAL_RATIO = 0.15
TEST_RATIO = 0.10
BUFFER_SIZE = 1


def parse(stem):
    m = re.match(r'(\d{8})_GRMN(\d+)_s(\d+)', stem)
    if m:
        return {
            'date': m.group(1),
            'video_num': int(m.group(2)),
            'second': int(m.group(3)),
            'video_id': f'{m.group(1)}_GRMN{m.group(2)}',
        }
    return None


def collect_images_with_videos(source_dir):
    """Zhromazdi vsetky obrazky a vytvor map video_id -> images."""
    by_video = defaultdict(list)

    for split in ['train', 'val', 'test']:
        img_dir = source_dir / split / 'images'
        if not img_dir.exists():
            continue
        for img in img_dir.glob('*.jpg'):
            p = parse(img.stem)
            if p:
                p['img_path'] = img
                p['lbl_path'] = source_dir / split / 'labels' / (img.stem + '.txt')
                by_video[p['video_id']].append(p)

    return by_video


def determine_block_split(by_video):
    """Aplikuj block-split logiku na zoradene videa."""
    sorted_videos = sorted(by_video.keys(), key=lambda v: (
        re.match(r'(\d{8})', v).group(1),
        int(re.match(r'\d{8}_GRMN(\d+)', v).group(1)),
    ))

    by_date = defaultdict(list)
    for vid in sorted_videos:
        date = vid.split('_')[0]
        by_date[date].append(vid)

    train_videos = []
    val_videos = []
    test_videos = []
    buffer_videos = []

    for date in sorted(by_date.keys()):
        videos = by_date[date]
        n = len(videos)
        n_train = int(n * TRAIN_RATIO)
        n_val = int(n * VAL_RATIO)
        n_test = int(n * TEST_RATIO)

        idx = 0
        train_block = videos[idx:idx + n_train]
        idx += n_train
        buffer1 = videos[idx:idx + BUFFER_SIZE]
        idx += BUFFER_SIZE
        val_block = videos[idx:idx + n_val]
        idx += n_val
        buffer2 = videos[idx:idx + BUFFER_SIZE]
        idx += BUFFER_SIZE
        test_block = videos[idx:idx + n_test]
        idx += n_test
        rest = videos[idx:]

        if rest:
            buffer3 = rest[:BUFFER_SIZE]
            rest_train = rest[BUFFER_SIZE:]
            buffer_videos.extend(buffer3)
            train_videos.extend(rest_train)

        train_videos.extend(train_block)
        val_videos.extend(val_block)
        test_videos.extend(test_block)
        buffer_videos.extend(buffer1 + buffer2)

    return train_videos, val_videos, test_videos, buffer_videos


def copy_dataset(by_video, train_videos, val_videos, test_videos, dest_dir, name):
    """Skopiruj snimky a labely do dest_dir podla split-u."""
    print(f'\n{"=" * 60}')
    print(f'Vytvaram dataset: {name}')
    print(f'{"=" * 60}')

    if dest_dir.exists():
        print(f'Mazem existujuci {dest_dir}')
        shutil.rmtree(dest_dir)

    splits_data = [('train', train_videos), ('val', val_videos), ('test', test_videos)]
    total_imgs = {'train': 0, 'val': 0, 'test': 0}
    total_bbox = {'train': 0, 'val': 0, 'test': 0}

    for split, videos in splits_data:
        (dest_dir / split / 'images').mkdir(parents=True)
        (dest_dir / split / 'labels').mkdir(parents=True)

        for vid in videos:
            if vid not in by_video:
                continue
            for p in by_video[vid]:
                shutil.copy2(p['img_path'], dest_dir / split / 'images' / p['img_path'].name)
                if p['lbl_path'].exists():
                    shutil.copy2(p['lbl_path'], dest_dir / split / 'labels' / p['lbl_path'].name)
                    # Spocitaj bbox
                    with open(p['lbl_path']) as f:
                        total_bbox[split] += len([l for l in f if l.strip()])
                total_imgs[split] += 1

    # YAML
    yaml_path = dest_dir / 'dataset.yaml'
    with open(yaml_path, 'w') as f:
        f.write(f"""# Block-split dataset (leakage-free, buffer=1)
path: {dest_dir}
train: train/images
val: val/images
test: test/images

nc: 2
names:
  0: car
  1: truck
""")

    print(f'\n{"Split":<10} {"Snimky":<10} {"Bbox":<10}')
    for split in ['train', 'val', 'test']:
        print(f'{split:<10} {total_imgs[split]:<10} {total_bbox[split]:<10}')
    print(f'YAML: {yaml_path}')


def main():
    print('=' * 60)
    print('BLOCK SPLIT v3 (PLNY YOLO + OREZANY Dist-YOLO)')
    print('=' * 60)

    if not SOURCE_YOLO.exists():
        print(f'❌ Zdroj YOLO neexistuje: {SOURCE_YOLO}')
        return

    if not SOURCE_DIST.exists():
        print(f'❌ Zdroj Dist neexistuje: {SOURCE_DIST}')
        return

    # === 1. SPRACOVANIE garmin_dataset (PLNY) ===
    print(f'\n[1] Citam povodny garmin_dataset/ (PLNY rozsah, 5-stlpec)')
    by_video_yolo = collect_images_with_videos(SOURCE_YOLO)
    n_total_yolo = sum(len(v) for v in by_video_yolo.values())
    print(f'   {len(by_video_yolo)} unikatnych videi, {n_total_yolo} snimok')

    train_y, val_y, test_y, buffer_y = determine_block_split(by_video_yolo)

    # === 2. SPRACOVANIE garmin_dist_dataset (OREZANY) ===
    print(f'\n[2] Citam povodny garmin_dist_dataset/ (<=90m, 6-stlpec)')
    by_video_dist = collect_images_with_videos(SOURCE_DIST)
    n_total_dist = sum(len(v) for v in by_video_dist.values())
    print(f'   {len(by_video_dist)} unikatnych videi, {n_total_dist} snimok')

    train_d, val_d, test_d, buffer_d = determine_block_split(by_video_dist)

    # === 3. KONTROLA SPLITOV ===
    print(f'\n[3] Kontrola block-splitov')

    def check_leakage(train, val, test, name):
        new_split = {}
        for v in train: new_split[v] = 'train'
        for v in val: new_split[v] = 'val'
        for v in test: new_split[v] = 'test'

        leakage = 0
        all_videos = sorted(set(list(new_split.keys())))
        for vid in all_videos:
            m = re.match(r'(\d{8})_GRMN(\d+)', vid)
            date, num = m.group(1), int(m.group(2))
            sp = new_split[vid]
            for delta in [-1, 1]:
                other_id = f'{date}_GRMN{num + delta:04d}'
                if other_id in new_split and new_split[other_id] != sp:
                    leakage += 1
        return leakage

    leak_yolo = check_leakage(train_y, val_y, test_y, 'YOLO')
    leak_dist = check_leakage(train_d, val_d, test_d, 'Dist')

    print(f'   YOLO dataset:  {leak_yolo} boundary leakage' + ('  ✅' if leak_yolo == 0 else ''))
    print(f'   Dist dataset:  {leak_dist} boundary leakage' + ('  ✅' if leak_dist == 0 else ''))

    copy_dataset(by_video_yolo, train_y, val_y, test_y, DEST_YOLO, 'garmin_dataset_v2')
    copy_dataset(by_video_dist, train_d, val_d, test_d, DEST_DIST, 'garmin_dist_dataset_v2')

    print('\n' + '=' * 60)
    print('HOTOVO!')
    print('=' * 60)
    print(f'\ngarmin_dataset_v2 (pre YOLO baseline + RF) - PLNY rozsah 3-143m')
    print(f'garmin_dist_dataset_v2 (pre Dist-YOLO) - rozsah 3-90m')


if __name__ == '__main__':
    main()