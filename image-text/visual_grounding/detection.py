import torch
import cv2
import json
import supervision as sv
from groundingdino.util.inference import Model
from pathlib import Path
import numpy as np

# -----------------------------------------------------------
# Load Grounding DINO model
# -----------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model = Model(
    model_config_path="image-text/visual_grounding/GroundingDINO_SwinT_OGC.py",
    model_checkpoint_path="image-text/visual_grounding/groundingdino_swint_ogc.pth"
)

# -----------------------------------------------------------
# Function to create prompt from alignment tokens
# -----------------------------------------------------------
def create_grounding_prompt_from_alignment(data_entry):
    """
    Extract tokens from alignment field to create Grounding DINO prompt.
    Uses all tokens regardless of 'matched' status.
    """
    if "alignment" not in data_entry:
        return "person"  # Fallback

    stop_words = {"a", "an", "the", "with", "in", "on", "at", "to", "of"}

    # If person_names is present and not empty, create a single token for the full name
    person_names = data_entry.get("person_names", [])
    person_token = None
    alignment_tokens = [item["token"] for item in data_entry["alignment"]]

    if person_names:
        # Join all person_names into one token (e.g., "gillian_anderson")
        person_token = " ".join(person_names).replace("_", " ")
        # Remove all person name tokens from alignment_tokens (case-insensitive)
        person_name_tokens = set()
        for name in person_names:
            for part in name.split("_"):
                person_name_tokens.add(part.lower())
        alignment_tokens = [t for t in alignment_tokens if t.lower() not in person_name_tokens]

    # Remove stop words
    meaningful_tokens = [t for t in alignment_tokens if t.lower() not in stop_words]

    # Remove duplicates while preserving order
    seen = set()
    unique_tokens = []
    for token in meaningful_tokens:
        token_lower = token.lower()
        if token_lower not in seen:
            seen.add(token_lower)
            unique_tokens.append(token)

    # Add person_token at the front if present
    if person_token:
        unique_tokens = [person_token] + unique_tokens

    # Join with " . " (Grounding DINO format)
    prompt = " . ".join(unique_tokens)

    return prompt

# -----------------------------------------------------------
# NMS to remove duplicate detections
# -----------------------------------------------------------
def apply_class_aware_nms(detections, phrases, iou_threshold=0.5, score_threshold=0.3):
    """
    Apply NMS to remove overlapping detections across all phrases.
    Prioritizes person names over generic attributes.
    """
    if len(detections.xyxy) == 0:
        return detections, phrases

    boxes = detections.xyxy
    scores = detections.confidence

    # Determine if each phrase is a person name (contains multiple words/parts)
    is_person_name = [len(phrase.split()) > 1 for phrase in phrases]

    # Sort by: 1) Person names first, 2) Score descending
    sorted_indices = sorted(
        range(len(phrases)),
        key=lambda i: (-is_person_name[i], -scores[i])
    )

    keep_indices = []
    suppressed = np.zeros(len(phrases), dtype=bool)

    for i in sorted_indices:
        if suppressed[i] or scores[i] < score_threshold:
            continue

        keep_indices.append(i)
        boxA = boxes[i]

        # Suppress all overlapping boxes (regardless of phrase)
        for j in sorted_indices:
            if i == j or suppressed[j]:
                continue

            boxB = boxes[j]
            # IoU calculation
            xA = max(boxA[0], boxB[0])
            yA = max(boxA[1], boxB[1])
            xB = min(boxA[2], boxB[2])
            yB = min(boxA[3], boxB[3])
            interArea = max(0, xB - xA) * max(0, yB - yA)
            boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
            boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
            iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)

            if iou > iou_threshold:
                suppressed[j] = True

    keep_indices = sorted(keep_indices)

    filtered_detections = sv.Detections(
        xyxy=boxes[keep_indices],
        confidence=scores[keep_indices],
        class_id=detections.class_id[keep_indices] if detections.class_id is not None else None
    )

    filtered_phrases = [phrases[i] for i in keep_indices]

    return filtered_detections, filtered_phrases

# -----------------------------------------------------------
# Generate distinct colors for different classes
# -----------------------------------------------------------
def generate_color_palette(num_colors):
    """Generate distinct colors using HSV color space."""
    colors = []
    for i in range(num_colors):
        hue = int(180 * i / num_colors)  # Spread across hue spectrum
        # Convert HSV to BGR
        hsv = np.uint8([[[hue, 255, 255]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
        colors.append(tuple(map(int, bgr)))
    return colors

# -----------------------------------------------------------
# Function to process single image
# -----------------------------------------------------------
def process_image_with_grounding(image_path, data_entry, output_dir="grounded_outputs", 
                                use_nms=True, iou_threshold=0.5, score_threshold=0.3):
    """
    Process a single image using Grounding DINO with tokens from alignment field.
    """
    Path(output_dir).mkdir(exist_ok=True)
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return None
    
    # Create prompt from alignment tokens
    text_prompt = create_grounding_prompt_from_alignment(data_entry)
    print(f"\nProcessing: {image_path}")
    print(f"Prompt from alignment: {text_prompt}")
    
    try:
        detections, phrases = model.predict_with_caption(
            image=image,
            caption=text_prompt,
            box_threshold=0.2,
            text_threshold=0.2
        )
        
        print(f"Raw Detections: {len(detections.xyxy)}")
        
        if use_nms and len(detections.xyxy) > 0:
            detections, phrases = apply_class_aware_nms(
                detections, phrases, 
                iou_threshold=iou_threshold,
                score_threshold=score_threshold
            )
            print(f"After NMS: {len(detections.xyxy)}")
        
        if len(detections.xyxy) == 0:
            print("No detections found after filtering!")
            return None
        
        # Create unique color for each phrase
        unique_phrases = list(set(phrases))
        color_palette = generate_color_palette(len(unique_phrases))
        phrase_to_color = {phrase: color_palette[i] for i, phrase in enumerate(unique_phrases)}
        
        annotated_image = image.copy()
        
        for box, score, phrase in zip(detections.xyxy, detections.confidence, phrases):
            x1, y1, x2, y2 = box.astype(int)
            
            # Get color for this phrase
            color = phrase_to_color[phrase]
            
            # Draw bounding box
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
            
            # Prepare label (without score)
            label = phrase
            
            # Calculate text size
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 8)
            
            # Draw label background (same color as box)
            label_y1 = max(y1 - h - 10, 0)
            label_y2 = label_y1 + h + 10
            cv2.rectangle(annotated_image, (x1, label_y1), (x1 + w + 10, label_y2), color, -1)
            
            # Choose text color for contrast
            brightness = (color[0]*0.299 + color[1]*0.587 + color[2]*0.114)
            text_color = (255, 255, 255) if brightness < 128 else (0, 0, 0)
            
            # Draw label text
            cv2.putText(annotated_image, label, (x1 + 3, label_y1 + h + 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
            
            print(f"  - {phrase}: {score:.2f}")
        
        output_filename = Path(image_path).stem + "_grounded.jpg"
        output_path = Path(output_dir) / output_filename
        cv2.imwrite(str(output_path), annotated_image)
        print(f"Saved: {output_path}")
        
        return {
            "image_path": image_path,
            "prompt": text_prompt,
            "detections": len(detections.xyxy),
            "phrases": phrases,
            "boxes": detections.xyxy.tolist(),
            "scores": detections.confidence.tolist()
        }
        
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

# -----------------------------------------------------------
# Batch process entire dataset
# -----------------------------------------------------------
def process_dataset(json_path, image_dir, output_dir="grounded_outputs"):
    """
    Process entire dataset from JSON file using alignment tokens.
    """
    with open(json_path, 'r') as f:
        dataset = json.load(f)
    
    results = []
    
    for entry in dataset:
        index = entry.get("index")
        image_filename = f"{index}.jpg"
        image_path = Path(image_dir) / image_filename
        
        if not image_path.exists():
            print(f"Image not found: {image_path}")
            continue
        
        result = process_image_with_grounding(
            str(image_path), entry, output_dir,
            use_nms=True,
            iou_threshold=0.5,
            score_threshold=0.3
        )
        
        if result:
            result["index"] = index
            result["caption"] = entry.get("caption")
            results.append(result)
    
    output_json = Path(output_dir) / "grounding_results.json"
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Processed {len(results)} images")
    print(f"Results saved to: {output_json}")
    print(f"{'='*60}")
    
    return results

# -----------------------------------------------------------
# Example usage
# -----------------------------------------------------------
if __name__ == "__main__":
    data_entry = {
        "index": 2732,
        "person_names": ["gillian_anderson"],
        "caption": "gillian anderson a young woman with a banana in her mouth",
        "alignment": [
            {"token": "gillian", "matched": True},
            {"token": "anderson", "matched": True},
            {"token": "a", "matched": True},
            {"token": "young", "matched": False},
            {"token": "woman", "matched": False},
            {"token": "with", "matched": True},
            {"token": "a", "matched": True},
            {"token": "banana", "matched": True},
            {"token": "in", "matched": True},
            {"token": "her", "matched": True},
            {"token": "mouth", "matched": False}
        ]
    }
    
    image_path = "public_images/public_image_set/7rrsu4.jpg"
    
    result = process_image_with_grounding(
        image_path, data_entry,
        use_nms=True,
        iou_threshold=0.5,
        score_threshold=0.3
    )