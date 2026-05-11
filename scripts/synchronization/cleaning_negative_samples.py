import os
import random

labels_dir = '/BP1/garmin_dataset/labels'

empty = [f for f in os.listdir(labels_dir)
         if f.endswith('.txt') and os.path.getsize(os.path.join(labels_dir, f)) == 0]

print(f"Prázdnych: {len(empty)}")

# Nechaj len 150 náhodných negatívnych
random.seed(69)
keep = set(random.sample(empty, min(150, len(empty))))
delete = [f for f in empty if f not in keep]

for f in delete:
    os.remove(os.path.join(labels_dir, f))

print(f"Zmazaných: {len(delete)}, ostalo negatívnych: 200")


import os
import shutil

images_dir = '/home/jozef/Documents/FIIT/5thSemester/BP1/garmin_dataset/images'
labels_dir = '/BP1/garmin_dataset/labels'
out_images = '/home/jozef/Documents/FIIT/5thSemester/BP1/garmin_dataset/images_clean'
os.makedirs(out_images, exist_ok=True)

labels = {f.replace('.txt', '.jpg') for f in os.listdir(labels_dir) if f.endswith('.txt')}

copied = 0
for filename in os.listdir(images_dir):
    if filename in labels:
        shutil.copy(os.path.join(images_dir, filename), os.path.join(out_images, filename))
        copied += 1

print(f"Skopírovaných: {copied} obrázkov")