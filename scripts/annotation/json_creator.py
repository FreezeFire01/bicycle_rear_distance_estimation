import os, json

IMAGES_DIR = '/BP1/garmin_dataset/images_clean'
LABELS_DIR = '/BP1/garmin_dataset/labels'
OUTPUT = '/home/jozef/Documents/FIIT/5thSemester/BP1/garmin_dataset/ls_all.json'

CLASS_NAMES = {0: 'car', 1: 'truck'}
tasks = []

for img in sorted(os.listdir(IMAGES_DIR)):
    if not img.endswith('.jpg'):
        continue
    label_path = os.path.join(LABELS_DIR, img.replace('.jpg', '.txt'))
    results = []
    if os.path.exists(label_path):
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls = int(parts[0])
                xc, yc, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                results.append({
                    'from_name': 'label', 'to_name': 'image',
                    'type': 'rectanglelabels',
                    'value': {
                        'x': (xc - w/2) * 100, 'y': (yc - h/2) * 100,
                        'width': w * 100, 'height': h * 100,
                        'rotation': 0, 'rectanglelabels': [CLASS_NAMES.get(cls, 'car')]
                    }
                })
    tasks.append({
        'data': {'image': f'/data/local-files/?d=images_clean/{img}'},
        'annotations': [{'result': results}] if results else []
    })

with open(OUTPUT, 'w') as f:
    json.dump(tasks, f, indent=2)
