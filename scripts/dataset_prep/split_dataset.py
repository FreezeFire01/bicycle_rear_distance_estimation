import os
import shutil
import random
import pandas as pd
from pathlib import Path
from collections import Counter


DATASET_DIR = '/BP1/garmin_dataset'
IMAGES_DIR = os.path.join(DATASET_DIR, 'images_clean')
LABELS_DIR = os.path.join(DATASET_DIR, 'labels')

TRAIN_RATIO = 0.75
VAL_RATIO = 0.15
TEST_RATIO = 0.10
SEED = 69


def get_clip_name(image_name):
    parts = image_name.rsplit('_', 1)
    return parts[0] if len(parts) == 2 else image_name


def main():
    random.seed(SEED)

    images = sorted([f for f in os.listdir(IMAGES_DIR) if f.endswith('.jpg')])
    labels = {f.replace('.txt', '.jpg') for f in os.listdir(LABELS_DIR) if f.endswith('.txt')}

    # Len obrázky čo majú label
    paired = [img for img in images if img in labels]
    print(f"Obrázkov s labelmi: {len(paired)}")

    # Group by clip
    clips = {}
    for img in paired:
        clip = get_clip_name(img)
        if clip not in clips:
            clips[clip] = []
        clips[clip].append(img)

    print(f"Klipov: {len(clips)}")

    # Shuffle clips
    clip_list = list(clips.keys())
    random.shuffle(clip_list)

    # Split clips
    n = len(clip_list)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    train_clips = clip_list[:n_train]
    val_clips = clip_list[n_train:n_train + n_val]
    test_clips = clip_list[n_train + n_val:]

    # Map clips → images
    splits = {
        'train': [img for c in train_clips for img in clips[c]],
        'val': [img for c in val_clips for img in clips[c]],
        'test': [img for c in test_clips for img in clips[c]],
    }

    print(f"\nSplit:")
    print(
        f"  Train: {len(train_clips)} klipov, {len(splits['train'])} obrázkov ({100 * len(splits['train']) / len(paired):.0f}%)")
    print(
        f"  Val:   {len(val_clips)} klipov, {len(splits['val'])} obrázkov ({100 * len(splits['val']) / len(paired):.0f}%)")
    print(
        f"  Test:  {len(test_clips)} klipov, {len(splits['test'])} obrázkov ({100 * len(splits['test']) / len(paired):.0f}%)")

    for split_name, split_images in splits.items():
        rides = Counter(img[:8] for img in split_images)
        print(f"  {split_name} rides: {dict(rides)}")

    for split_name, split_images in splits.items():
        img_dir = os.path.join(DATASET_DIR, split_name, 'images')
        lbl_dir = os.path.join(DATASET_DIR, split_name, 'labels')
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)

        for img_name in split_images:
            src_img = os.path.join(IMAGES_DIR, img_name)
            if os.path.exists(src_img):
                shutil.copy2(src_img, os.path.join(img_dir, img_name))

            txt_name = img_name.replace('.jpg', '.txt')
            src_lbl = os.path.join(LABELS_DIR, txt_name)
            if os.path.exists(src_lbl):
                shutil.copy2(src_lbl, os.path.join(lbl_dir, txt_name))

    print(f"\nPozitívne/negatívne:")
    for split_name in ['train', 'val', 'test']:
        lbl_dir = os.path.join(DATASET_DIR, split_name, 'labels')
        total = len(os.listdir(lbl_dir))
        empty = sum(1 for f in os.listdir(lbl_dir)
                    if f.endswith('.txt') and os.path.getsize(os.path.join(lbl_dir, f)) == 0)
        print(f"  {split_name}: {total} celkom, {total - empty} s autami, {empty} negatívnych")

    # Write dataset.yaml
    yaml_path = os.path.join(DATASET_DIR, 'dataset.yaml')
    with open(yaml_path, 'w') as f:
        f.write(f"# Garmin Bicycle Safety Dataset\n")
        f.write(f"path: {DATASET_DIR}\n")
        f.write(f"train: train/images\n")
        f.write(f"val: val/images\n")
        f.write(f"test: test/images\n")
        f.write(f"\n")
        f.write(f"names:\n")
        f.write(f"  0: car\n")
        f.write(f"  1: truck\n")
        f.write(f"\n")
        f.write(f"nc: 2\n")

    print(f"\nYAML: {yaml_path}")

    # Write split lists (for reproducibility)
    for split_name, split_images in splits.items():
        list_path = os.path.join(DATASET_DIR, f'{split_name}_images.txt')
        with open(list_path, 'w') as f:
            for img in sorted(split_images):
                f.write(f"{img}\n")
        print(f"  {split_name} list: {list_path}")



if __name__ == '__main__':
    main()