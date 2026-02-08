import os
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import torch
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm

# Face detection and recognition imports
from facenet_pytorch import MTCNN, InceptionResnetV1
import joblib

# Caption generation imports
from transformers import BlipProcessor, BlipForConditionalGeneration

class NewsImageCaptionGenerator:
    def __init__(self, face_model_path="face_model", device=None):
        """
        Initialize the caption generator with face detection and recognition models.
        
        Args:
            face_model_path: Path to directory containing SVM model and label encoder
            device: torch device (cuda/cpu). If None, automatically detects.
        """
        # Force CUDA if available
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda:0")
                print(f"✓ GPU Available: {torch.cuda.get_device_name(0)}")
                print(f"✓ CUDA Version: {torch.version.cuda}")
            else:
                self.device = torch.device("cpu")
                print("⚠ GPU not available, using CPU")
        else:
            self.device = device
        
        print(f"Using device: {self.device}")
        
        # Initialize face detection and recognition models
        print("Loading face detection models...")
        self.mtcnn = MTCNN(keep_all=True, device=self.device)
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        print(f"✓ Face models loaded on: {next(self.resnet.parameters()).device}")
        
        # Load trained face recognition models
        self.clf = joblib.load(f"{face_model_path}/svm_face_recognizer.joblib")
        self.le = joblib.load(f"{face_model_path}/label_encoder.joblib")
        
        # Initialize caption generation models
        print("Loading caption generation models...")
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        self.caption_model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-large"
        )
        self.caption_model = self.caption_model.to(self.device)
        self.caption_model.eval()  # Set to evaluation mode
        print(f"✓ Caption model loaded on: {next(self.caption_model.parameters()).device}")
        
        # Check GPU memory usage
        if self.device.type == 'cuda':
            print(f"✓ GPU Memory allocated: {torch.cuda.memory_allocated()/1024**2:.0f} MB")
            print(f"✓ GPU Memory reserved: {torch.cuda.memory_reserved()/1024**2:.0f} MB")
        
        # Face preprocessing transform
        self.face_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        print("Models loaded successfully!")
    
    def detect_faces(self, img):
        """
        Detect and recognize faces in an image.
        
        Args:
            img: PIL Image
            
        Returns:
            List of detected person names with confidence scores
        """
        boxes, _ = self.mtcnn.detect(img)
        
        if boxes is None:
            return []
        
        detected_persons = []
        
        for box in boxes:
            x1, y1, x2, y2 = [int(b) for b in box]
            
            # Crop and prepare face for recognition
            crop = img.crop((x1, y1, x2, y2)).resize((160, 160))
            x = self.face_transform(crop).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                emb = self.resnet(x).cpu().numpy()
            
            # Predict identity
            probs = self.clf.predict_proba(emb)[0]
            pred_idx = np.argmax(probs)
            pred_label = self.le.inverse_transform([pred_idx])[0]
            confidence = probs[pred_idx]
            
            detected_persons.append({
                'name': pred_label,
                'confidence': float(confidence)
            })
        
        return detected_persons
    
    def generate_caption(self, img, detected_persons=None):
        """
        Generate caption for an image, optionally using detected person names.
        
        Args:
            img: PIL Image
            detected_persons: List of detected person dictionaries (optional)
            
        Returns:
            Generated caption string
        """
        # Prepare text prompt with detected names if available
        if detected_persons and len(detected_persons) > 0:
            # Use names of detected persons as context
            # Replace underscores with spaces and capitalize properly
            names = []
            for p in detected_persons:
                name = p['name'].replace('_', ' ').title()
                names.append(name)
            
            # Remove duplicates while preserving order
            unique_names = []
            seen = set()
            for name in names:
                if name.lower() not in seen:
                    unique_names.append(name)
                    seen.add(name.lower())
            
            # Limit to first 3 unique names to avoid overwhelming the model
            text_prompt = ", ".join(unique_names[:3]) + ", "
        else:
            text_prompt = ""
        
        # Generate caption - ensure inputs are on the correct device
        inputs = self.processor(img, text=text_prompt, return_tensors="pt")
        
        # Move all input tensors to the device
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                 for k, v in inputs.items()}
        
        with torch.no_grad():
            out = self.caption_model.generate(**inputs, max_length=50, 
                    num_beams=3, repetition_penalty=2.0)
        
        caption = self.processor.decode(out[0], skip_special_tokens=True)
        
        # Clean up any remaining repetitions in the caption
        caption = self.clean_repetitive_caption(caption)
        
        return caption
    
    def clean_repetitive_caption(self, caption):
        """
        Remove excessive repetitions from generated captions.
        
        Args:
            caption: Generated caption string
            
        Returns:
            Cleaned caption string
        """
        # Split by common delimiters
        words = caption.replace(',', ' ').replace('.', ' ').split()
        
        # Remove consecutive duplicates
        cleaned_words = []
        prev_word = None
        consecutive_count = 0
        
        for word in words:
            word_lower = word.lower()
            if word_lower == prev_word:
                consecutive_count += 1
                # Only allow 1 repetition maximum
                if consecutive_count <= 1:
                    cleaned_words.append(word)
            else:
                cleaned_words.append(word)
                consecutive_count = 0
            prev_word = word_lower
        
        # Reconstruct caption
        cleaned_caption = ' '.join(cleaned_words)
        
        # Remove trailing commas or incomplete text
        cleaned_caption = cleaned_caption.rstrip(', ')
        
        return cleaned_caption
    
    def process_image(self, image_path):
        """
        Process a single image: detect faces and generate caption.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dictionary with detected persons and caption
        """
        try:
            # Check if file exists
            if not os.path.exists(image_path):
                return {
                    'image_path': image_path,
                    'detected_persons': [],
                    'num_faces': 0,
                    'person_names': [],
                    'caption': '',
                    'status': 'error: file not found'
                }
            
            img = Image.open(image_path).convert("RGB")
            
            # Detect faces
            detected_persons = self.detect_faces(img)
            
            # Generate caption
            caption = self.generate_caption(img, detected_persons)
            
            return {
                'image_path': image_path,
                'detected_persons': detected_persons,
                'num_faces': len(detected_persons),
                'person_names': [p['name'] for p in detected_persons],
                'caption': caption,
                'status': 'success'
            }
        
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            return {
                'image_path': image_path,
                'detected_persons': [],
                'num_faces': 0,
                'person_names': [],
                'caption': '',
                'status': f'error: {str(e)}'
            }

    def process_dataset(self, dataset_df, id_column='image_filename', image_base_path='allData_images', 
                       image_column=None, output_path='all_data_news_captions.csv'):
        """
        Process entire dataset of news images.
        
        Args:
            dataset_df: Pandas DataFrame containing image IDs or paths
            id_column: Name of column containing image IDs (used to construct path)
            image_base_path: Base directory path for images
            image_column: Name of column containing full image paths (optional, overrides id_column)
            output_path: Path to save output CSV
            
        Returns:
            DataFrame with added caption and face detection columns
        """
        results = []
        
        print(f"Processing {len(dataset_df)} images...")
        
        for idx, row in tqdm(dataset_df.iterrows(), total=len(dataset_df)):
            # Construct image path from ID if image_column not specified
            if image_column:
                image_path = row[image_column]
            else:
                image_id = row[id_column]
                # Try common image extensions
                extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
                image_path = None
                
                for ext in extensions:
                    potential_path = os.path.join(image_base_path, str(image_id) + ext)
                    if os.path.exists(potential_path):
                        image_path = potential_path
                        break
                
                # If no file found with extension, try without extension
                if image_path is None:
                    potential_path = os.path.join(image_base_path, str(image_id))
                    if os.path.exists(potential_path):
                        image_path = potential_path
                    else:
                        # File not found, will be handled in process_image
                        image_path = potential_path
            
            result = self.process_image(image_path)
            
            # Combine original row data with results
            result_row = row.to_dict()
            result_row.update(result)
            results.append(result_row)
        
        # Create output DataFrame
        output_df = pd.DataFrame(results)
        
        # Save to CSV
        output_df.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")
        
        # Print summary statistics
        self.print_summary(output_df)
        
        return output_df
    
    def process_directory(self, image_dir, output_path='all_data_news_captions.csv', 
                         image_extensions=['.jpg', '.jpeg', '.png', '.bmp']):
        """
        Process all images in a directory.
        
        Args:
            image_dir: Directory containing images
            output_path: Path to save output CSV
            image_extensions: List of valid image extensions
            
        Returns:
            DataFrame with caption and face detection results
        """
        # Find all images in directory
        image_paths = []
        for root, dirs, files in os.walk(image_dir):
            for file in files:
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    image_paths.append(os.path.join(root, file))
        
        print(f"Found {len(image_paths)} images in {image_dir}")
        
        # Create DataFrame
        df = pd.DataFrame({'image_path': image_paths})
        
        # Process dataset
        return self.process_dataset(df, output_path=output_path)
    
    def print_summary(self, results_df):
        """Print summary statistics of processing results."""
        print("\n" + "="*60)
        print("PROCESSING SUMMARY")
        print("="*60)
        print(f"Total images processed: {len(results_df)}")
        print(f"Successful: {(results_df['status'] == 'success').sum()}")
        print(f"Failed: {(results_df['status'] != 'success').sum()}")
        print(f"Images with faces: {(results_df['num_faces'] > 0).sum()}")
        print(f"Total faces detected: {results_df['num_faces'].sum()}")
        print(f"Average faces per image: {results_df['num_faces'].mean():.2f}")
        print("="*60)


# Example usage
if __name__ == "__main__":
    # Initialize the generator
    generator = NewsImageCaptionGenerator(face_model_path="face_model")
    
    # Option 1: Process from DataFrame with ID column
    # Dataset has 'id' column, images are at public_images/public_image_set/{id}
    df = pd.read_csv("datasets/processed/all_data_test.csv")
    results = generator.process_dataset(
        df, 
        id_column='image_filename',
        image_base_path='allData_images',
        output_path='all_data_news_captions.csv'
    )
    
    # Option 2: If you have full image paths in a column
    # results = generator.process_dataset(df, image_column='image_path', 
    #                                     output_path='news_with_captions.csv')
    
    # Option 3: Process all images in a directory
    # results = generator.process_directory(
    #     image_dir="public_images/public_image_set",
    #     output_path="news_captions.csv"
    # )
    
    # Option 4: Process single image
    # result = generator.process_image("public_images/public_image_set/1a7fgd.jpg")
    # print(f"Caption: {result['caption']}")
    # print(f"Detected persons: {result['person_names']}")
    
    # View sample results
    print("\nSample Results:")
    print(results[['id', 'image_path', 'num_faces', 'person_names', 'caption']].head(10))