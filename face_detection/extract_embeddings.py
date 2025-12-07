# extract_embeddings.py
import os
import numpy as np
from facenet_pytorch import InceptionResnetV1
from PIL import Image
from tqdm import tqdm
import torch
import joblib

CROPS_DIR = "face_crops"
EMBED_DIR = "face_embeddings"  # will save .npy per face or a single dataset file

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load pretrained resnet (weights='vggface2' gives good embeddings)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def get_embedding(img_path):
    img = Image.open(img_path).convert("RGB").resize((160,160))
    # convert to tensor expecting shape (3,160,160) and normalized in facenet-pytorch's expected range
    import torchvision.transforms as transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
    ])
    x = transform(img).unsqueeze(0).to(device)  # (1,3,160,160)
    with torch.no_grad():
        emb = resnet(x)   # (1,512)
    return emb.detach().cpu().numpy()[0]

def main():
    X = []
    y = []
    paths = []

    for entity in os.listdir(CROPS_DIR):
        entity_folder = os.path.join(CROPS_DIR, entity)
        if not os.path.isdir(entity_folder): continue
        for img_name in os.listdir(entity_folder):
            img_path = os.path.join(entity_folder, img_name)
            try:
                emb = get_embedding(img_path)
                X.append(emb)
                y.append(entity)  # label is folder name (sanitized)
                paths.append(img_path)
            except Exception as e:
                print("Failed embedding for", img_path, e)

    X = np.array(X)
    y = np.array(y)

    os.makedirs(EMBED_DIR, exist_ok=True)
    np.save(os.path.join(EMBED_DIR, "X.npy"), X)
    joblib.dump(y, os.path.join(EMBED_DIR, "y.joblib"))
    print("Saved embeddings:", X.shape, "labels:", y.shape)

if __name__ == "__main__":
    main()
