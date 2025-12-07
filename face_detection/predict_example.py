# predict_with_bbox.py
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image, ImageDraw, ImageFont
import torch
import torchvision.transforms as transforms
import joblib
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mtcnn = MTCNN(keep_all=True, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

clf = joblib.load("face_model/svm_face_recognizer.joblib")
le = joblib.load("face_model/label_encoder.joblib")

def predict_image(path, output_path="output_with_boxes.jpg"):
    img = Image.open(path).convert("RGB")
    boxes, _ = mtcnn.detect(img)
    
    if boxes is None:
        print("No faces detected.")
        return
    
    # Create a drawing context
    draw = ImageDraw.Draw(img)
    
    # Try to load a font, fallback to default if not available
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = [int(b) for b in box]
        
        # Crop and prepare face for recognition
        crop = img.crop((x1, y1, x2, y2)).resize((160, 160))
        x = transform(crop).unsqueeze(0).to(device)
        
        with torch.no_grad():
            emb = resnet(x).cpu().numpy()
        
        # Predict identity
        probs = clf.predict_proba(emb)[0]
        pred_idx = np.argmax(probs)
        pred_label = le.inverse_transform([pred_idx])[0]
        confidence = probs[pred_idx]
        
        print(f"Face {i+1}: {pred_label} ({confidence:.2f})")
        
        # Draw bounding box
        draw.rectangle([(x1, y1), (x2, y2)], outline="green", width=3)
        
        # Prepare label text
        label_text = f"{pred_label}"
        
        # Get text bounding box for background
        bbox = draw.textbbox((x1, y1 - 20), label_text, font=font)
        
        # Draw background rectangle for text
        draw.rectangle(bbox, fill="green")
        
        # Draw label text
        draw.text((x1, y1 - 20), label_text, fill="white", font=font)
    
    # Save the output image
    img.save(output_path)
    print(f"\nOutput saved to: {output_path}")
    
    # Optionally display the image
    img.show()
    
    return img

if __name__ == "__main__":
    predict_image("public_images/public_image_set/1df1iy.jpg", "output_with_boxes.jpg")