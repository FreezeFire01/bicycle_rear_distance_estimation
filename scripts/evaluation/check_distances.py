import pandas as pd
import cv2
import os

df = pd.read_csv('/BP1/garmin_dataset/rf_dataset.csv')
meta = pd.read_csv('/BP1/garmin_dataset/metadata.csv')
meta_dict = {row['image']: row for _, row in meta.iterrows()}

images_dir = '/BP1/garmin_dataset/images_clean'
out_dir = '/home/jozef/Documents/FIIT/5thSemester/BP1/garmin_dataset/verify_distances'
os.makedirs(out_dir, exist_ok=True)

for image_name, group in df.groupby('image'):
    img = cv2.imread(os.path.join(images_dir, image_name))
    if img is None:
        continue

    h, w, _ = img.shape
    num_cars_radar = int(meta_dict.get(image_name, {}).get('num_cars', 0))
    num_labels = len(group)

    for _, bbox_row in group.iterrows():
        cx, cy, bw, bh = bbox_row['cx'], bbox_row['cy'], bbox_row['w'], bbox_row['h']
        dist = bbox_row['distance_m']
        cls_id = int(bbox_row['cls_id'])

        x1 = int((cx - bw/2) * w)
        y1 = int((cy - bh/2) * h)
        x2 = int((cx + bw/2) * w)
        y2 = int((cy + bh/2) * h)

        color = (0, 255, 0) if cls_id == 0 else (0, 0, 255)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, f"{dist:.0f}m", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    match = num_labels == num_cars_radar
    status = f"OK | radar={num_cars_radar} labels={num_labels}" if match else f"MISMATCH | radar={num_cars_radar} labels={num_labels}"
    color_status = (0, 200, 0) if match else (0, 0, 255)
    cv2.putText(img, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_status, 2)

    cv2.imwrite(os.path.join(out_dir, image_name), img)

print(f"Hotovo! Otvor: {out_dir}")