import os
import torch
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# Pretrained YOLOv5m na COCO (pozná car=2, truck=7)
model = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True)
model.conf = 0.4
model.classes = [2, 7]  # len car a truck

COCO_TO_LABEL = {2: 0, 7: 1}  # car→0, truck→1

input_folder = '/tmp/new_20260412'
label_folder = '/home/jozef/Documents/FIIT/5thSemester/BP1/garmin_dataset/labels'
os.makedirs(label_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    img_path = os.path.join(input_folder, filename)
    results = model(img_path)

    label_path = os.path.join(label_folder, os.path.splitext(filename)[0] + '.txt')

    with open(label_path, 'w') as f:
        for *xyxy, conf, cls in results.xyxy[0].cpu().numpy():
            cls_id = COCO_TO_LABEL[int(cls)]
            # Konverzia na YOLO formát (normalizované)
            img_w, img_h = results.ims[0].shape[1], results.ims[0].shape[0]
            x1, y1, x2, y2 = xyxy
            cx = ((x1 + x2) / 2) / img_w
            cy = ((y1 + y2) / 2) / img_h
            w = (x2 - x1) / img_w
            h = (y2 - y1) / img_h
            f.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

    print(f"Labelované: {filename}")
