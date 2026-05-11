import os
import shutil
import cv2
import numpy as np
import pandas as pd
import json
from collections import defaultdict
from sklearn.model_selection import train_test_split
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import box_in_image, view_points
from pyquaternion import Quaternion
import time
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================

# UPDATE THIS PATH TO YOUR NUSCENES LOCATION
DATAROOT = r"/home/jozef/Documents/FIIT/5thSemester/data/nuScenes"

# Output directory
OUTPUT_DIR = "../../bicycle_safety_dataset_final"

# Class mapping (YOLO format)
# NOTE: Only motorized vehicles that pose a threat to cyclists
# Excluded: bicycles (not a threat) and trailers (already behind detected vehicle)
CLASS_MAP = {
    "vehicle.car": 0,
    "vehicle.truck": 1,
    "vehicle.bus": 1,
    #"vehicle.motorcycle": 3,
    # "vehicle.bicycle": 4,    # EXCLUDED - not a threat to cyclists
    # "vehicle.trailer": 5,    # EXCLUDED - always behind a vehicle
}

# === CRITICAL FILTERS (FIXES) ===
MIN_DISTANCE = 3.0  # Minimum distance in meters
MAX_DISTANCE = 140  # Maximum distance (objects beyond are too small)
MIN_BOX_SIZE_PX = 15  # Minimum bounding box dimension in pixels
MIN_VISIBILITY = 2  # nuScenes visibility level (1=worst, 4=best)
MIN_LIDAR_POINTS = 5  # Minimum lidar points (occlusion indicator)

# Train/val split
VAL_SPLIT_RATIO = 0.15
RANDOM_SEED = 69


def calculate_box_size_pixels(corners):
    """Calculate bounding box width and height in pixels"""
    x_coords = corners[0, :]
    y_coords = corners[1, :]
    width_px = np.max(x_coords) - np.min(x_coords)
    height_px = np.max(y_coords) - np.min(y_coords)
    return width_px, height_px


def get_corners_2d(box, intrinsic, img_w, img_h):
    """
    Project 3D box to 2D image plane
    Returns: (corners, bbox) or (None, None) if invalid
    """
    try:
        corners = view_points(box.corners(), intrinsic, normalize=True)[:2, :]
        x1, y1 = np.min(corners[0]), np.min(corners[1])
        x2, y2 = np.max(corners[0]), np.max(corners[1])

        # Bounds check
        if x1 < 0 or y1 < 0 or x2 > img_w or y2 > img_h or x2 <= x1 or y2 <= y1:
            return None, None

        return corners, (x1, y1, x2, y2)
    except Exception:
        return None, None


def to_yolo_format(x1, y1, x2, y2, img_w, img_h):
    """Convert bbox to YOLO format: x_center, y_center, width, height (normalized)"""
    bbox_w = x2 - x1
    bbox_h = y2 - y1
    x_center = x1 + bbox_w / 2.0
    y_center = y1 + bbox_h / 2.0

    return (
        x_center / img_w,  # x_center normalized
        y_center / img_h,  # y_center normalized
        bbox_w / img_w,  # width normalized
        bbox_h / img_h,  # height normalized
        bbox_w,  # width pixels (for regression)
        bbox_h,  # height pixels (for regression)
        x_center,  # x_center pixels
        y_center  # y_center pixels
    )


def is_vehicle_relevant(nusc, ann):
    """
    CRITICAL FILTER: Determine if vehicle should be included
    Returns: (is_relevant, reason)
    """
    # 1. Check parked/stopped status
    is_parked_or_stopped = False
    is_moving = False

    for attribute_token in ann['attribute_tokens']:
        attribute = nusc.get('attribute', attribute_token)
        attr_name = attribute['name']

        if 'vehicle.parked' in attr_name or 'vehicle.stopped' in attr_name:
            is_parked_or_stopped = True
        if 'vehicle.moving' in attr_name:
            is_moving = True

        # SKIP parked or stopped vehicles
        if is_parked_or_stopped:
            return False, "parked_or_stopped"

    # SKIP if not moving (except bicycles - they can be slow)
    if not is_moving and 'bicycle' not in ann['category_name']:
        return False, "not_moving"

    # 2. Check visibility (nuScenes format: 'v0-40', 'v40-60', 'v60-80', 'v80-100')
    if ann.get('visibility_token'):
        visibility = nusc.get('visibility', ann['visibility_token'])
        vis_name = visibility['level']

        # Map visibility string to numeric level
        # 'v0-40' = 1 (worst), 'v40-60' = 2, 'v60-80' = 3, 'v80-100' = 4 (best)
        vis_mapping = {
            'v0-40': 1,
            'v40-60': 2,
            'v60-80': 3,
            'v80-100': 4,
            '': 1  # Unknown/empty -> worst case
        }

        vis_level = vis_mapping.get(vis_name, 1)

        if vis_level < MIN_VISIBILITY:
            return False, f"low_visibility_{vis_name}"

    # 3. Check occlusion via lidar points
    num_lidar_pts = ann.get('num_lidar_pts', 0)
    if num_lidar_pts < MIN_LIDAR_POINTS:
        return False, "heavily_occluded"

    return True, "relevant"


# =============================================================================
# MAIN PROCESSING
# =============================================================================

def main():

    print("nuScenes Bicycle Safety Dataset Builder - FIXED VERSION")

    print(f"\nConfiguration:")
    print(f"  nuScenes path:     {DATAROOT}")
    print(f"  Output directory:  {OUTPUT_DIR}")
    print(f"  Distance range:    {MIN_DISTANCE}m - {MAX_DISTANCE}m")
    print(f"  Min box size:      {MIN_BOX_SIZE_PX}px")
    print(f"  Min visibility:    {MIN_VISIBILITY}/4")
    print(f"  Min lidar points:  {MIN_LIDAR_POINTS}")
    print(f"  Val split:         {VAL_SPLIT_RATIO * 100:.0f}%")
    print(f"  Random seed:       {RANDOM_SEED}")


    # Check if nuScenes path exists
    if not os.path.exists(DATAROOT):
        print(f"\n❌ ERROR: nuScenes path not found: {DATAROOT}")
        print("\nPlease update DATAROOT in the script to your nuScenes location.")
        return

    # Check required files
    metadata_path = os.path.join(DATAROOT, "v1.0-trainval")
    cam_back_path = os.path.join(DATAROOT, "samples", "CAM_BACK")

    if not os.path.exists(metadata_path):
        print(f"\n❌ ERROR: Metadata not found: {metadata_path}")
        print("\nYou need nuScenes v1.0-trainval metadata (~200 MB)")
        return

    if not os.path.exists(cam_back_path):
        print(f"\n❌ ERROR: CAM_BACK images not found: {cam_back_path}")
        print("\nYou need samples/CAM_BACK/ directory (~15 GB)")
        return

    print("\n✓ Found required nuScenes files")

    # Clean output directory
    if os.path.exists(OUTPUT_DIR):
        print(f"\n⚠  Removing existing output directory: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)

    # Create output structure
    for split in ['train', 'val']:
        os.makedirs(f"{OUTPUT_DIR}/{split}/images", exist_ok=True)
        os.makedirs(f"{OUTPUT_DIR}/{split}/labels", exist_ok=True)

    # Load nuScenes

    print("Loading nuScenes dataset...")


    start_time = time.time()
    nusc = NuScenes(version='v1.0-trainval', dataroot=DATAROOT, verbose=True)
    print(f"✓ Loaded in {time.time() - start_time:.1f}s")

    # Create train/val split by scenes
    all_scene_names = [s['name'] for s in nusc.scene]

    if len(all_scene_names) < 2:
        train_scenes, val_scenes = all_scene_names, []
    else:
        train_scenes, val_scenes = train_test_split(
            all_scene_names,
            test_size=VAL_SPLIT_RATIO,
            random_state=RANDOM_SEED
        )

    scene_to_split = {name: 'train' for name in train_scenes}
    for name in val_scenes:
        scene_to_split[name] = 'val'

    print(f"\nScene split:")
    print(f"  Train scenes: {len(train_scenes)}")
    print(f"  Val scenes:   {len(val_scenes)}")

    # Statistics
    stats = {
        'total_samples': 0,
        'total_vehicles': 0,
        'used_objects': 0,
        'skipped_parked': 0,
        'skipped_not_moving': 0,
        'skipped_occluded': 0,
        'skipped_low_visibility': 0,
        'skipped_too_small': 0,
        'skipped_out_of_range': 0,
        'skipped_out_of_image': 0,
        'images_train': 0,
        'images_val': 0,
    }

    split_records = {'train': [], 'val': []}

    # Process samples

    print("Processing samples...")


    progress_interval = 50

    for idx, sample in enumerate(nusc.sample):
        stats['total_samples'] += 1

        # Get scene and split
        scene = nusc.get('scene', sample['scene_token'])
        scene_name = scene['name']
        split_name = scene_to_split[scene_name]

        # Get rear camera
        cam_token = sample['data'].get('CAM_BACK')
        if cam_token is None:
            continue

        cam_data = nusc.get('sample_data', cam_token)
        cam_path = nusc.get_sample_data_path(cam_token)

        # Load image
        img_bgr = cv2.imread(cam_path)
        if img_bgr is None:
            continue

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_h, img_w = img_rgb.shape[:2]

        # Get calibration
        calib = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
        ego_pose = nusc.get('ego_pose', cam_data['ego_pose_token'])
        intrinsic = np.array(calib['camera_intrinsic'])

        # Process annotations for this image
        image_annotations = []

        for ann_token in sample['anns']:
            ann = nusc.get('sample_annotation', ann_token)

            # Filter: only vehicle classes
            if ann['category_name'] not in CLASS_MAP:
                continue

            stats['total_vehicles'] += 1

            # === CRITICAL FILTER 1: Relevance (moving, visible) ===
            is_relevant, reason = is_vehicle_relevant(nusc, ann)
            if not is_relevant:
                if reason == "parked_or_stopped":
                    stats['skipped_parked'] += 1
                elif reason == "not_moving":
                    stats['skipped_not_moving'] += 1
                elif "visibility" in reason:
                    stats['skipped_low_visibility'] += 1
                elif "occluded" in reason:
                    stats['skipped_occluded'] += 1
                continue

            # Get 3D box and transform to camera coordinates
            box = nusc.get_box(ann_token)

            # Transform: global -> ego -> camera
            box.translate(-np.array(ego_pose['translation']))
            box.rotate(Quaternion(ego_pose['rotation']).inverse)
            box.translate(-np.array(calib['translation']))
            box.rotate(Quaternion(calib['rotation']).inverse)

            # Check if box is in image
            if not box_in_image(box, intrinsic, (img_w, img_h), vis_level=0):
                stats['skipped_out_of_image'] += 1
                continue

            # Get box center (camera coordinates)
            x, y, z = box.center

            # === CRITICAL FILTER 2: Distance ===
            # Z is longitudinal distance (depth) in camera frame
            if z <= 0 or z < MIN_DISTANCE or z > MAX_DISTANCE:
                stats['skipped_out_of_range'] += 1
                continue

            distance = float(z)

            # Project to 2D
            corners, bbox = get_corners_2d(box, intrinsic, img_w, img_h)
            if corners is None:
                stats['skipped_out_of_image'] += 1
                continue

            x1, y1, x2, y2 = bbox

            # === CRITICAL FILTER 3: Minimum size ===
            width_px, height_px = calculate_box_size_pixels(corners)

            if width_px < MIN_BOX_SIZE_PX or height_px < MIN_BOX_SIZE_PX:
                stats['skipped_too_small'] += 1
                continue

            # Create record
            record = {
                'image_id': f"sample_{idx:05d}",
                'vehicle_id': f"vehicle_{idx:05d}_{len(image_annotations):02d}",
                'category': ann['category_name'],
                'distance_m': distance,
                'lateral_offset_m': float(abs(x)),
                'width_pixels': float(width_px),
                'height_pixels': float(height_px),
                'bbox_x1': float(x1),
                'bbox_y1': float(y1),
                'bbox_x2': float(x2),
                'bbox_y2': float(y2),
                'split': split_name,
            }

            image_annotations.append(record)

        # Save image and annotations if we have any valid objects
        if len(image_annotations) > 0:
            # Save image
            image_id = image_annotations[0]['image_id']
            out_img_path = f"{OUTPUT_DIR}/{split_name}/images/{image_id}.jpg"
            cv2.imwrite(out_img_path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))

            # Update stats
            if split_name == 'train':
                stats['images_train'] += 1
            else:
                stats['images_val'] += 1

            # Add to records
            for rec in image_annotations:
                split_records[split_name].append(rec)
                stats['used_objects'] += 1

        # Progress update
        if (idx + 1) % progress_interval == 0:
            print(f"[{idx + 1:4d}/{len(nusc.sample)}] "
                  f"Train: {stats['images_train']:4d} | "
                  f"Val: {stats['images_val']:3d} | "
                  f"Objects: {stats['used_objects']:5d} | "
                  f"Skipped: {stats['skipped_parked'] + stats['skipped_not_moving'] + stats['skipped_too_small']:5d}")

    # Final statistics

    print("DATASET STATISTICS")

    print(f"\nProcessed:")
    print(f"  Total samples:         {stats['total_samples']}")
    print(f"  Total vehicles:        {stats['total_vehicles']}")
    print(f"\n✓ Used:")
    print(f"  Objects included:      {stats['used_objects']}")
    print(f"  Train images:          {stats['images_train']}")
    print(f"  Val images:            {stats['images_val']}")
    print(f"\n✗ Skipped:")
    print(f"  Parked/stopped:        {stats['skipped_parked']}")
    print(f"  Not moving:            {stats['skipped_not_moving']}")
    print(f"  Too small (<{MIN_BOX_SIZE_PX}px):  {stats['skipped_too_small']}")
    print(f"  Out of range:          {stats['skipped_out_of_range']}")
    print(f"  Low visibility:        {stats['skipped_low_visibility']}")
    print(f"  Heavily occluded:      {stats['skipped_occluded']}")
    print(f"  Out of image:          {stats['skipped_out_of_image']}")

    total_skipped = sum([
        stats['skipped_parked'],
        stats['skipped_not_moving'],
        stats['skipped_too_small'],
        stats['skipped_out_of_range'],
        stats['skipped_low_visibility'],
        stats['skipped_occluded'],
        stats['skipped_out_of_image']
    ])
    print(f"\n  Total skipped:         {total_skipped}")
    print(f"  Retention rate:        {stats['used_objects'] / stats['total_vehicles'] * 100:.1f}%")

    # Generate YOLO labels and regression metadata

    print("Generating YOLO labels...")


    regression_rows = []

    for split_name, records in split_records.items():
        # Group by image_id
        per_image = defaultdict(list)
        for rec in records:
            per_image[rec['image_id']].append(rec)

        for image_id, anns in per_image.items():
            # Load image to get dimensions
            img_path = f"{OUTPUT_DIR}/{split_name}/images/{image_id}.jpg"
            img = cv2.imread(img_path)
            if img is None:
                continue

            ih, iw = img.shape[:2]

            yolo_lines = []

            for rec in anns:
                x1, y1, x2, y2 = rec['bbox_x1'], rec['bbox_y1'], rec['bbox_x2'], rec['bbox_y2']

                # Convert to YOLO format
                x_c_norm, y_c_norm, w_norm, h_norm, w_px, h_px, x_c_px, y_c_px = \
                    to_yolo_format(x1, y1, x2, y2, iw, ih)

                cls_id = CLASS_MAP[rec['category']]

                # YOLO label line
                yolo_lines.append(f"{cls_id} {x_c_norm:.6f} {y_c_norm:.6f} {w_norm:.6f} {h_norm:.6f}")

                # Regression metadata
                offset_norm = abs(x_c_px - (iw / 2.0)) / iw

                regression_rows.append({
                    'image_id': image_id,
                    'cls_id': cls_id,
                    'w_px': w_px,
                    'h_px': h_px,
                    'offset_norm': offset_norm,
                    'distance_m': rec['distance_m'],
                    'split': split_name
                })

            # Write YOLO label file
            label_path = f"{OUTPUT_DIR}/{split_name}/labels/{image_id}.txt"
            with open(label_path, 'w') as f:
                for line in yolo_lines:
                    f.write(line + '\n')

        print(f"✓ Generated {len(per_image)} label files for {split_name}")

    # Save regression metadata
    df_regression = pd.DataFrame(regression_rows)
    df_regression.to_csv(f"{OUTPUT_DIR}/distance_regression_meta.csv", index=False)
    print(f"✓ Saved distance regression metadata")

    # Save dataset CSVs
    for split_name in ['train', 'val']:
        if split_records[split_name]:
            df = pd.DataFrame(split_records[split_name])
            df.to_csv(f"{OUTPUT_DIR}/{split_name}_dataset.csv", index=False)
            print(f"✓ Saved {split_name}_dataset.csv")

    # Save metadata
    metadata = {
        'dataset_name': 'Bicycle Safety Dataset (FIXED)',
        'version': '1.0',
        'created': time.strftime('%Y-%m-%d %H:%M:%S'),
        'source': 'nuScenes v1.0-trainval',
        'camera': 'CAM_BACK',
        'statistics': stats,
        'configuration': {
            'distance_range_m': [MIN_DISTANCE, MAX_DISTANCE],
            'min_box_size_px': MIN_BOX_SIZE_PX,
            'min_visibility': MIN_VISIBILITY,
            'min_lidar_points': MIN_LIDAR_POINTS,
            'random_seed': RANDOM_SEED,
            'val_split_ratio': VAL_SPLIT_RATIO
        },
        'classes': {v: k for k, v in CLASS_MAP.items()},
        'fixes_applied': [
            'Filter parked/stopped vehicles',
            'Filter non-moving vehicles',
            'Minimum bounding box size filtering (15px)',
            'Visibility filtering (min level 2)',
            'Occlusion filtering (min 5 lidar points)',
            'Distance range filtering (3-140m)'
        ]
    }

    with open(f"{OUTPUT_DIR}/metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"✓ Saved metadata.json")

    # Create data.yaml for YOLO
    data_yaml = f"""# Bicycle Safety Dataset - YOLO Configuration
# Only motorized vehicles that pose a threat to cyclists
path: {os.path.abspath(OUTPUT_DIR)}
train: train/images
val: val/images

# Classes (motorized vehicles only)
names:
  0: car
  1: truck

# Dataset info
nc: 2  # number of classes (bicycles and trailers excluded)
"""

    with open(f"{OUTPUT_DIR}/data.yaml", 'w') as f:
        f.write(data_yaml)

    print(f"✓ Saved data.yaml")

    # Final summary

    print("DATASET CREATION COMPLETE!")

    print(f"\nOutput directory: {OUTPUT_DIR}/")
    print(f"\nDataset summary:")
    print(f"  Train images:  {stats['images_train']}")
    print(f"  Val images:    {stats['images_val']}")
    print(f"  Total objects: {stats['used_objects']}")
    print(f"\nFiles created:")
    print(f"  - {OUTPUT_DIR}/train/images/ ({stats['images_train']} images)")
    print(f"  - {OUTPUT_DIR}/train/labels/ ({stats['images_train']} labels)")
    print(f"  - {OUTPUT_DIR}/val/images/ ({stats['images_val']} images)")
    print(f"  - {OUTPUT_DIR}/val/labels/ ({stats['images_val']} labels)")
    print(f"  - {OUTPUT_DIR}/distance_regression_meta.csv")
    print(f"  - {OUTPUT_DIR}/data.yaml")
    print(f"  - {OUTPUT_DIR}/metadata.json")


    print("NEXT STEPS")

    print("\n1. Inspect dataset:")
    print(f"   ls {OUTPUT_DIR}/train/images | wc -l")
    print(f"   head {OUTPUT_DIR}/distance_regression_meta.csv")

    print("\n2. Train YOLO model:")
    print(f"   yolo detect train \\")
    print(f"     data={OUTPUT_DIR}/data.yaml \\")
    print(f"     model=yolo11n.pt \\")
    print(f"     epochs=100 \\")
    print(f"     imgsz=640 \\")
    print(f"     rect=True \\")
    print(f"     batch=-1 \\")
    print(f"     device=0 \\")
    print(f"     seed=42")




if __name__ == "__main__":
    main()