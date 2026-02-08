# detect_and_crop_faces.py
import os
import json
from PIL import Image
from facenet_pytorch import MTCNN
from tqdm import tqdm
import torch

RAW_DIR = "face_dataset"
CROPS_DIR = "face_crops"   # result: one folder per entity with face_<img>_<face_idx>.jpg
ENTITIES_JSON = "datasets/processed/all_data_unique_entities.json"

# Load valid entities from JSON
with open(ENTITIES_JSON, 'r') as f:
    valid_entities = json.load(f)

# Convert entity names to folder names (spaces -> underscores, lowercase)
valid_folder_names = {entity.replace(" ", "_") for entity in valid_entities}

# initialize MTCNN (use device if you have GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mtcnn = MTCNN(keep_all=True, device=device)

def create_dir(path):
    os.makedirs(path, exist_ok=True)

def detect_and_save_crops():
    create_dir(CROPS_DIR)

    for entity_folder in os.listdir(RAW_DIR):
        src_folder = os.path.join(RAW_DIR, entity_folder)
        dst_folder = os.path.join(CROPS_DIR, entity_folder)

        if not os.path.isdir(src_folder):
            continue
        
        # Only process folders that match entities in the JSON file
        if entity_folder not in valid_folder_names:
            continue

        create_dir(dst_folder)

        for img_name in tqdm(os.listdir(src_folder), desc=f"Processing {entity_folder}"):

            img_path = os.path.join(src_folder, img_name)

            # Skip non-image files
            if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            # Try opening the image
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f" Skipping (bad image): {img_path} -> {e}")
                continue

            # Run face detection safely
            try:
                boxes, probs = mtcnn.detect(img)
            except Exception as e:
                print(f"Skipping (MTCNN error): {img_path} -> {e}")
                continue

            # If no face detected, skip
            if boxes is None:
                # print(f" No face: {img_path}")
                continue

            # Crop and save detected faces
            for idx, box in enumerate(boxes):
                try:
                    x1, y1, x2, y2 = [int(max(0, b)) for b in box]
                    crop = img.crop((x1, y1, x2, y2))

                    crop_path = os.path.join(
                        dst_folder, f"{os.path.splitext(img_name)[0]}_face{idx+1}.jpg"
                    )

                    crop.save(crop_path)

                except Exception as e:
                    print(f" Failed to crop/save face for {img_path}: {e}")
                    continue

    print("\n Done! All detectable faces have been cropped and saved.")

if __name__ == "__main__":
    detect_and_save_crops()
