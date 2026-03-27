import os
import shutil
import random

# Calea unde ai acum toate cele 19 foldere
source_dir = r'F:\FACULTATE\concurs severin bumbaru\antrenament_ai'
base_dir = 'data'

classes = os.listdir(source_dir)

for cls in classes:
    # Cream structura de foldere
    os.makedirs(os.path.join(base_dir, 'train', cls), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'val', cls), exist_ok=True)

    # Luam toate pozele din folderul plantei
    src_path = os.path.join(source_dir, cls)
    images = os.listdir(src_path)
    random.shuffle(images)  # Le amestecam sa nu fie toate din aceeasi serie

    # Calculam unde taiem (80%)
    split_idx = int(len(images) * 0.8)
    train_images = images[:split_idx]
    val_images = images[split_idx:]

    # Mutam fisierele
    for img in train_images:
        shutil.copy(os.path.join(src_path, img), os.path.join(base_dir, 'train', cls, img))
    for img in val_images:
        shutil.copy(os.path.join(src_path, img), os.path.join(base_dir, 'val', cls, img))

print("Gata! Datele au fost impartite corect.")