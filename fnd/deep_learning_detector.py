import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, confusion_matrix
from PIL import Image
from tqdm import tqdm
import wandb

class MultimodalDataset(Dataset):
    """
    OPTIMIZED: Preload and cache all images in memory
    """
    
    def __init__(self, image_paths, headlines, labels, clip_processor, preload=True):
        self.image_paths = image_paths
        self.headlines = headlines
        self.labels = labels
        self.clip_processor = clip_processor
        
        # OPTIMIZATION 1: Preload all images into memory
        self.preloaded_images = None
        if preload:
            print("Preloading images into memory...")
            self.preloaded_images = []
            for img_path in tqdm(image_paths):
                try:
                    img = Image.open(img_path).convert('RGB')
                    # Process immediately to tensors
                    img_tensor = self.clip_processor(images=img, return_tensors="pt")['pixel_values'].squeeze(0)
                    self.preloaded_images.append(img_tensor)
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
                    # Use dummy tensor on error
                    self.preloaded_images.append(torch.zeros(3, 224, 224))
        
        # OPTIMIZATION 2: Preprocess all text with CLIP tokenizer
        print("Preprocessing text...")
        self.preloaded_text = self.clip_processor(
            text=headlines,
            padding='max_length',
            truncation=True,
            max_length=77,
            return_tensors='pt'
        )
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        if self.preloaded_images is not None:
            # Use preloaded data (MUCH faster)
            return {
                'pixel_values': self.preloaded_images[idx],
                'input_ids': self.preloaded_text['input_ids'][idx],
                'attention_mask': self.preloaded_text['attention_mask'][idx],
                'label': torch.tensor(self.labels[idx], dtype=torch.long)
            }
        else:
            # Fallback to on-the-fly loading
            image = Image.open(self.image_paths[idx]).convert('RGB')
            image_inputs = self.clip_processor(images=image, return_tensors="pt")
            
            text_inputs = self.clip_processor(
                text=self.headlines[idx],
                padding='max_length',
                truncation=True,
                max_length=77,
                return_tensors='pt'
            )
            
            return {
                'pixel_values': image_inputs['pixel_values'].squeeze(0),
                'input_ids': text_inputs['input_ids'].squeeze(0),
                'attention_mask': text_inputs['attention_mask'].squeeze(0),
                'label': torch.tensor(self.labels[idx], dtype=torch.long)
            }


class MultimodalClassifier(nn.Module):
    """
    OPTIMIZED: Freeze CLIP model entirely, only train fusion layers
    Uses CLIP for both image and text encoding
    """
    
    def __init__(self, clip_model_name='openai/clip-vit-base-patch32', 
                 freeze_encoders=True):
        super().__init__()
        
        # Load CLIP model for both modalities
        self.clip_model = CLIPModel.from_pretrained(clip_model_name)
        
        self.clip_dim = self.clip_model.config.projection_dim
        
        # OPTIMIZATION 3: Freeze ALL CLIP parameters
        if freeze_encoders:
            print("Freezing CLIP encoder...")
            for param in self.clip_model.parameters():
                param.requires_grad = False
            
            # Set to eval mode
            self.clip_model.eval()
        
        # OPTIMIZATION 4: Smaller, faster fusion network
        # Both image and text embeddings have the same dimension (projection_dim)
        self.fusion = nn.Sequential(
            nn.Linear(self.clip_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 2)
        )
        
    def forward(self, pixel_values, input_ids, attention_mask):
        # Use no_grad for frozen encoder (faster)
        with torch.no_grad():
            image_features = self.clip_model.get_image_features(pixel_values=pixel_values)
            text_features = self.clip_model.get_text_features(
                input_ids=input_ids, 
                attention_mask=attention_mask
            )
        
        # Only compute gradients for fusion layer
        combined = torch.cat([image_features, text_features], dim=1)
        logits = self.fusion(combined)
        return logits


class DeepLearningDetector:
    """
    OPTIMIZED version with WandB integration for tracking
    """
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu', 
                 freeze_encoders=True, use_wandb=True, wandb_project="fake-news-detection"):
        self.device = device
        self.use_wandb = use_wandb
        print(f"Using device: {device}")
        
        self.model = MultimodalClassifier(freeze_encoders=freeze_encoders).to(device)
        self.clip_processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
        
        # Only optimize trainable parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        total_trainable = sum(p.numel() for p in trainable_params)
        print(f"Trainable parameters: {total_trainable:,}")
        
        self.optimizer = torch.optim.AdamW(trainable_params, lr=1e-3)
        self.criterion = nn.CrossEntropyLoss()
        
        # Initialize WandB if enabled
        if self.use_wandb and not wandb.run:
            wandb.init(
                project=wandb_project,
                config={
                    "architecture": "CLIP Multimodal (Image + Text)",
                    "dataset": "FakeReddit",
                    "freeze_encoders": freeze_encoders,
                    "trainable_params": total_trainable,
                    "optimizer": "AdamW",
                    "learning_rate": 1e-3,
                    "device": str(device)
                }
            )
            # Watch model gradients
            wandb.watch(self.model, log="all", log_freq=100)
    
    def train(self, train_image_paths, train_headlines, train_labels, 
              val_image_paths=None, val_headlines=None, val_labels=None,
              batch_size=32, epochs=10, preload=True):
        """
        Train with WandB logging
        """
        
        print(f"Label distribution:")
        print(f"  FAKE (0): {train_labels.count(0)}")
        print(f"  REAL (1): {train_labels.count(1)}")
        
        # Log dataset info to WandB
        if self.use_wandb:
            wandb.config.update({
                "batch_size": batch_size,
                "epochs": epochs,
                "train_samples": len(train_labels),
                "train_fake": train_labels.count(0),
                "train_real": train_labels.count(1),
            })
            
            if val_labels is not None:
                wandb.config.update({
                    "val_samples": len(val_labels),
                    "val_fake": val_labels.count(0),
                    "val_real": val_labels.count(1),
                })
        
        # Create optimized dataset
        train_dataset = MultimodalDataset(
            train_image_paths, train_headlines, train_labels,
            self.clip_processor, preload=preload
        )
        
        num_workers = 0 if preload else 4
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        
        # Create validation dataset if provided
        val_loader = None
        if val_image_paths is not None:
            val_dataset = MultimodalDataset(
                val_image_paths, val_headlines, val_labels,
                self.clip_processor, preload=preload
            )
            val_loader = DataLoader(
                val_dataset, 
                batch_size=batch_size, 
                shuffle=False,
                pin_memory=True
            )
        
        # Training loop
        best_val_acc = 0.0
        for epoch in range(epochs):
            # ==================== TRAINING ====================
            self.model.train()
            total_loss = 0
            correct = 0
            total = 0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [TRAIN]")
            for batch_idx, batch in enumerate(pbar):
                # Move to device
                pixel_values = batch['pixel_values'].to(self.device, non_blocking=True)
                input_ids = batch['input_ids'].to(self.device, non_blocking=True)
                attention_mask = batch['attention_mask'].to(self.device, non_blocking=True)
                labels = batch['label'].to(self.device, non_blocking=True)
                
                self.optimizer.zero_grad()
                logits = self.model(pixel_values, input_ids, attention_mask)
                loss = self.criterion(logits, labels)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                
                # Update progress bar
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                
                # Log batch metrics to WandB
                if self.use_wandb:
                    wandb.log({
                        "batch/train_loss": loss.item(),
                        "batch/train_step": epoch * len(train_loader) + batch_idx
                    })
            
            # Calculate epoch metrics
            avg_train_loss = total_loss / len(train_loader)
            train_accuracy = correct / total
            
            print(f"\nEpoch {epoch+1}/{epochs} Summary:")
            print(f"  Train Loss: {avg_train_loss:.4f}")
            print(f"  Train Accuracy: {train_accuracy:.4f}")
            
            # ==================== VALIDATION ====================
            val_metrics = {}
            if val_loader is not None:
                val_metrics = self._validate(val_loader, epoch)
                print(f"  Val Loss: {val_metrics['val/loss']:.4f}")
                print(f"  Val Accuracy: {val_metrics['val/accuracy']:.4f}")
                print(f"  Val ROC-AUC: {val_metrics['val/roc_auc']:.4f}")
                
                # Save best model
                if val_metrics['val/accuracy'] > best_val_acc:
                    best_val_acc = val_metrics['val/accuracy']
                    self.save_model('models/best_model.pth')
                    if self.use_wandb:
                        wandb.run.summary["best_val_accuracy"] = best_val_acc
                        wandb.run.summary["best_epoch"] = epoch + 1
            
            # Log epoch metrics to WandB
            if self.use_wandb:
                log_dict = {
                    "epoch": epoch + 1,
                    "train/loss": avg_train_loss,
                    "train/accuracy": train_accuracy,
                    "learning_rate": self.optimizer.param_groups[0]['lr']
                }
                log_dict.update(val_metrics)
                wandb.log(log_dict)
            
            print("-" * 60)
    
    def _validate(self, val_loader, epoch):
        """
        Validation with comprehensive metrics
        """
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch+1} [VAL]")
            for batch in pbar:
                pixel_values = batch['pixel_values'].to(self.device, non_blocking=True)
                input_ids = batch['input_ids'].to(self.device, non_blocking=True)
                attention_mask = batch['attention_mask'].to(self.device, non_blocking=True)
                labels = batch['label'].to(self.device, non_blocking=True)
                
                logits = self.model(pixel_values, input_ids, attention_mask)
                loss = self.criterion(logits, labels)
                
                total_loss += loss.item()
                probabilities = torch.softmax(logits, dim=1)
                _, predicted = torch.max(logits, 1)
                
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities[:, 1].cpu().numpy())
                
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Calculate metrics
        avg_val_loss = total_loss / len(val_loader)
        val_accuracy = correct / total
        val_roc_auc = roc_auc_score(all_labels, all_probabilities)
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        
        # Per-class metrics
        tn, fp, fn, tp = cm.ravel()
        precision_fake = tn / (tn + fn) if (tn + fn) > 0 else 0
        recall_fake = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision_real = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall_real = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # Log confusion matrix to WandB
        if self.use_wandb:
            wandb.log({
                "val/confusion_matrix": wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=all_labels,
                    preds=all_predictions,
                    class_names=["Fake", "Real"]
                )
            })
        
        return {
            "val/loss": avg_val_loss,
            "val/accuracy": val_accuracy,
            "val/roc_auc": val_roc_auc,
            "val/precision_fake": precision_fake,
            "val/recall_fake": recall_fake,
            "val/precision_real": precision_real,
            "val/recall_real": recall_real,
            "val/true_negatives": int(tn),
            "val/false_positives": int(fp),
            "val/false_negatives": int(fn),
            "val/true_positives": int(tp)
        }
    
    def predict(self, image_paths, headlines, batch_size=32, preload=True):
        """
        Fast prediction with preloading
        """
        self.model.eval()
        
        dummy_labels = [0] * len(image_paths)
        test_dataset = MultimodalDataset(
            image_paths, headlines, dummy_labels,
            self.clip_processor, preload=preload
        )
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            pin_memory=True
        )
        
        all_predictions = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Predicting"):
                pixel_values = batch['pixel_values'].to(self.device, non_blocking=True)
                input_ids = batch['input_ids'].to(self.device, non_blocking=True)
                attention_mask = batch['attention_mask'].to(self.device, non_blocking=True)
                
                logits = self.model(pixel_values, input_ids, attention_mask)
                probabilities = torch.softmax(logits, dim=1)
                _, predicted = torch.max(logits, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_probabilities.extend(probabilities[:, 1].cpu().numpy())
        
        return np.array(all_predictions), np.array(all_probabilities)
    
    def evaluate(self, image_paths, headlines, labels, batch_size=32, preload=True, log_to_wandb=True):
        """
        Comprehensive evaluation with WandB logging
        """
        predictions, probabilities = self.predict(image_paths, headlines, batch_size, preload)
        accuracy = accuracy_score(labels, predictions)
        
        report = classification_report(
            labels, 
            predictions, 
            target_names=['Fake', 'Real'],
            zero_division=0,
            output_dict=True
        )
        
        try:
            roc_auc = roc_auc_score(labels, probabilities)
        except ValueError:
            roc_auc = 0.0
        
        # Confusion matrix
        cm = confusion_matrix(labels, predictions)
        
        # Print report
        print("\n" + "="*60)
        print("TEST SET EVALUATION")
        print("="*60)
        print(classification_report(labels, predictions, target_names=['Fake', 'Real'], zero_division=0))
        print(f"ROC-AUC Score: {roc_auc:.4f}")
        print(f"Overall Accuracy: {accuracy:.4f}")
        
        # Log to WandB
        if self.use_wandb and log_to_wandb:
            wandb.log({
                "test/accuracy": accuracy,
                "test/roc_auc": roc_auc,
                "test/precision_fake": report['Fake']['precision'],
                "test/recall_fake": report['Fake']['recall'],
                "test/f1_fake": report['Fake']['f1-score'],
                "test/precision_real": report['Real']['precision'],
                "test/recall_real": report['Real']['recall'],
                "test/f1_real": report['Real']['f1-score'],
                "test/confusion_matrix": wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=labels,
                    preds=predictions,
                    class_names=["Fake", "Real"]
                )
            })
            
            # Create a summary table
            test_table = wandb.Table(
                columns=["Metric", "Fake Class", "Real Class"],
                data=[
                    ["Precision", report['Fake']['precision'], report['Real']['precision']],
                    ["Recall", report['Fake']['recall'], report['Real']['recall']],
                    ["F1-Score", report['Fake']['f1-score'], report['Real']['f1-score']],
                    ["Support", report['Fake']['support'], report['Real']['support']]
                ]
            )
            wandb.log({"test/metrics_table": test_table})
        
        return accuracy, report
    
    def save_model(self, path='my_model.pth'):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path='my_model.pth'):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.model.eval()
        print(f"Model loaded from {path}")
    
    def finish_wandb(self):
        """Close WandB run"""
        if self.use_wandb:
            wandb.finish()


# USAGE EXAMPLE
"""
# Initialize with WandB
detector = DeepLearningDetector(
    device='cuda', 
    freeze_encoders=True,
    use_wandb=True,
    wandb_project="fake-news-detection"
)

# Train with validation
detector.train(
    train_images, 
    train_headlines, 
    train_labels,
    val_image_paths=val_images,
    val_headlines=val_headlines,
    val_labels=val_labels,
    batch_size=64,
    epochs=50,
    preload=True
)

# Evaluate on test set
detector.evaluate(test_images, test_headlines, test_labels)

# Close WandB
detector.finish_wandb()

# View results at: https://wandb.ai/your-username/fake-news-detection
"""