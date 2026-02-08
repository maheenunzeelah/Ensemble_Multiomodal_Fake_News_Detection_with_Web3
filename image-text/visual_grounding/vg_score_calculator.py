"""
Visual Grounding Score Calculator using Grounding DINO
Separate module to keep main analyzer clean
"""

import torch
import cv2
import numpy as np
from groundingdino.util.inference import Model
from pathlib import Path
import re

class GroundingDINOVGS:
    """Visual Grounding Score calculator using Grounding DINO"""
    
    def __init__(self, 
                 model_config_path="image-text/visual_grounding/GroundingDINO_SwinT_OGC.py",
                 model_checkpoint_path="image-text/visual_grounding/groundingdino_swint_ogc.pth"):
        """
        Initialize Grounding DINO model
        
        Args:
            model_config_path: Path to model config
            model_checkpoint_path: Path to model checkpoint
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading Grounding DINO model on {self.device}...")
        
        self.model = Model(
            model_config_path=model_config_path,
            model_checkpoint_path=model_checkpoint_path
        )
        print("✅ Grounding DINO model loaded successfully")
    
    def tokenize(self, text):
        """Simple tokenization"""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        return [t for t in text.split() if t]
    
    def create_grounding_prompt(self, headline, person_names):
        """
        Create Grounding DINO prompt from headline and person names
        
        Args:
            headline: Text headline/caption
            person_names: List of person names (e.g., ["Donald_Trump"])
        
        Returns:
            Formatted prompt string
        """
        stop_words = {"a", "an", "the", "with", "in", "on", "at", "to", "of", "is", "are"}
        
        tokens = []
        
        # Add person names first (joined with spaces, not underscores)
        if person_names:
            for name in person_names:
                person_token = name.replace("_", " ")
                tokens.append(person_token)
        
        # Extract tokens from headline
        headline_tokens = self.tokenize(headline)
        
        # Remove person name parts from headline tokens
        person_name_parts = set()
        if person_names:
            for name in person_names:
                for part in name.lower().split("_"):
                    person_name_parts.add(part)
        
        # Filter headline tokens
        for token in headline_tokens:
            if token and token not in stop_words and token not in person_name_parts:
                if token not in [t.lower() for t in tokens]:
                    tokens.append(token)
        
        # Join with " . " (Grounding DINO format)
        prompt = " . ".join(tokens)
        return prompt
    
    def apply_nms(self, detections, phrases, iou_threshold=0.5, score_threshold=0.3):
        """
        Apply NMS to remove overlapping detections
        
        Args:
            detections: Detection results from model
            phrases: Corresponding phrase labels
            iou_threshold: IoU threshold for NMS
            score_threshold: Minimum confidence score
        
        Returns:
            Filtered detections and phrases
        """
        if len(detections.xyxy) == 0:
            return detections, phrases

        boxes = detections.xyxy
        scores = detections.confidence

        # Filter by score threshold first
        valid_indices = scores >= score_threshold
        boxes = boxes[valid_indices]
        scores = scores[valid_indices]
        phrases = [phrases[i] for i in range(len(phrases)) if valid_indices[i]]
        
        if len(boxes) == 0:
            return detections.__class__(xyxy=np.array([]), confidence=np.array([])), []

        # Sort by score descending
        sorted_indices = np.argsort(-scores)
        
        keep_indices = []
        suppressed = np.zeros(len(phrases), dtype=bool)

        for idx in sorted_indices:
            i = int(idx)
            if suppressed[i]:
                continue

            keep_indices.append(i)
            boxA = boxes[i]

            # Suppress overlapping boxes
            for jdx in sorted_indices:
                j = int(jdx)
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

        filtered_detections = detections.__class__(
            xyxy=boxes[keep_indices],
            confidence=scores[keep_indices]
        )
        filtered_phrases = [phrases[i] for i in keep_indices]

        return filtered_detections, filtered_phrases
    
    def compute_vgs(self,
                    image_path,
                    headline,
                    person_names=None,
                    box_threshold=0.2,
                    text_threshold=0.2,
                    iou_threshold=0.5,
                    score_threshold=0.3,
                    use_nms=True):
        """
        Compute visual grounding score for an image given headline and person names
        
        Args:
            image_path: Path to the image file
            headline: Text headline/caption
            person_names: List of person names (e.g., ["Donald_Trump"])
            box_threshold: Detection box threshold
            text_threshold: Detection text threshold
            iou_threshold: NMS IoU threshold
            score_threshold: Minimum confidence score
            use_nms: Whether to apply NMS
        
        Returns:
            float: Visual grounding score (0.0 to 1.0)
        """
        if person_names is None:
            person_names = []
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Failed to load image: {image_path}")
            return 0.0
        
        # Create prompt
        text_prompt = self.create_grounding_prompt(headline, person_names)
        
        # Extract entities we're looking for (for scoring)
        headline_entities = []
        if person_names:
            headline_entities.extend([name.replace("_", " ") for name in person_names])
        
        # Add other meaningful tokens from headline
        stop_words = {"a", "an", "the", "with", "in", "on", "at", "to", "of", "is", "are"}
        headline_tokens = self.tokenize(headline)
        person_name_parts = set()
        if person_names:
            for name in person_names:
                for part in name.lower().split("_"):
                    person_name_parts.add(part)
        
        for token in headline_tokens:
            if token and token not in stop_words and token not in person_name_parts:
                if token not in [e.lower() for e in headline_entities]:
                    headline_entities.append(token)
        
        if not headline_entities:
            print(f" No entities to detect for: {headline}")
            return 0.0
        
        # Run Grounding DINO
        try:
            detections, phrases = self.model.predict_with_caption(
                image=image,
                caption=text_prompt,
                box_threshold=box_threshold,
                text_threshold=text_threshold
            )
            
            # Apply NMS if requested
            if use_nms and len(detections.xyxy) > 0:
                detections, phrases = self.apply_nms(
                    detections, phrases,
                    iou_threshold=iou_threshold,
                    score_threshold=score_threshold
                )
            
            # Convert to detection list format
            detection_list = [
                {"label": phrase, "score": float(score)}
                for phrase, score in zip(phrases, detections.confidence)
            ]
            
            # Compute visual grounding score
            matched_scores = []
            
            for ent in headline_entities:
                ent_lower = ent.lower()
                found_scores = [
                    d["score"] for d in detection_list
                    if ent_lower in d["label"].lower()
                ]
                if found_scores:
                    matched_scores.append(max(found_scores))
                else:
                    matched_scores.append(0.0)
            
            # Coverage: fraction of entities detected
            coverage = sum(1 for s in matched_scores if s > 0) / len(headline_entities) if headline_entities else 0.0
            
            # Confidence: average confidence of matched entities
            matched_conf = [s for s in matched_scores if s > 0]
            confidence = sum(matched_conf) / len(matched_conf) if matched_conf else 0.0
            
            # Final visual grounding score
            vgs = coverage * confidence
            
            return vgs
            
        except Exception as e:
            print(f" Error processing image {image_path}: {e}")
            return 0.0


# Singleton instance to avoid reloading model
_vgs_calculator_instance = None

def get_vgs_calculator():
    """Get or create singleton VGS calculator instance"""
    global _vgs_calculator_instance
    if _vgs_calculator_instance is None:
        _vgs_calculator_instance = GroundingDINOVGS()
    return _vgs_calculator_instance


def calculate_vgs_for_sample(image_path, headline, person_names=None):
    """
    Convenience function to calculate VGS for a single sample
    
    Args:
        image_path: Path to image
        headline: Text headline
        person_names: List of person names (optional)
    
    Returns:
        float: VGS score
    """
    calculator = get_vgs_calculator()
    return calculator.compute_vgs(image_path, headline, person_names)


if __name__ == "__main__":
    # Test the calculator
    print("Testing Grounding DINO VGS Calculator...")
    
    calculator = GroundingDINOVGS()
    
    # Test case
    vgs = calculator.compute_vgs(
        image_path="public_images/public_image_set/7rrsu4.jpg",
        headline="Donald Trump with a banana",
        person_names=["Donald_Trump"]
    )
    
    print(f"\nTest VGS Score: {vgs:.3f}")