import os
import cv2
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Zariadenie: {device}")

# Správny spôsob načítania YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'custom', path='../../best.pt', force_reload=False)
model.to(device)
model.conf = 0.25

input_folder = '/tmp/new_20260412'
output_folder = '/home/jozef/Documents/FIIT/5thSemester/BP1/garmin_dataset/images1'
os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path)
        if img is None:
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = model(img_rgb)

        for *xyxy, conf, cls in results.xyxy[0].cpu().numpy():
            x1, y1, x2, y2 = map(int, xyxy)
            h, w, _ = img.shape
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
            roi = img[y1:y2, x1:x2]
            if roi.size > 0:
                roi = cv2.GaussianBlur(roi, (91, 91), 0)
                img[y1:y2, x1:x2] = roi

        cv2.imwrite(os.path.join(output_folder, filename), img)
        print(f"Spracované: {filename}")
