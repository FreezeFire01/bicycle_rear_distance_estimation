import pandas as pd
import os
import ast

labels_dir = '/BP1/garmin_dataset/labels'
meta = pd.read_csv('/BP1/garmin_dataset/metadata.csv')

rows = []

for _, row in meta.iterrows():
    image = row['image']

    # Preskočí ak radar nevidel nič
    if pd.isna(row['closest_m']) or row['num_cars'] == 0:
        continue

    label_path = os.path.join(labels_dir, image.replace('.jpg', '.txt'))
    if not os.path.exists(label_path):
        continue

    with open(label_path) as f:
        boxes = [l.strip().split() for l in f if l.strip()]

    if not boxes:
        continue

    # Vzdialenosti z radaru — vyfiltruj nuly
    ranges = sorted([r for r in ast.literal_eval(row['all_ranges']) if r > 0])

    if not ranges:
        continue

    # Zoraď boxy podľa plochy (väčší = bližší)
    boxes_sorted = sorted(boxes, key=lambda b: float(b[3]) * float(b[4]), reverse=True)

    for i, b in enumerate(boxes_sorted):
        if i >= len(ranges):
            break
        cls_id = int(b[0])
        cx, cy, w, h = float(b[1]), float(b[2]), float(b[3]), float(b[4])
        rows.append({
            'image': image,
            'cls_id': cls_id,
            'cx': cx,
            'cy': cy,
            'w': w,
            'h': h,
            'area': w * h,
            'aspect_ratio': w / h if h > 0 else 0,
            'y_bottom': cy + h / 2,  # kde auto "stojí" v obraze
            'distance_m': ranges[i]
        })

df = pd.DataFrame(rows)
df.to_csv('/home/jozef/Documents/FIIT/5thSemester/BP1/garmin_dataset/rf_dataset.csv', index=False)
print(f"Hotovo: {len(df)} vzoriek")
print(f"Dist range: {df['distance_m'].min()}m – {df['distance_m'].max()}m")
print(df.head(10))