import numpy as np
import pandas as pd
import torch
import json
import os
import wandb
import pickle
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid tkinter issues
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, confusion_matrix

from deep_learning_detector import DeepLearningDetector
from rule_based_detector import RuleBasedDetectorWithExplainability

def get_image_path(image_id, base_path='allData_images'):
    """Find the actual image file with any extension"""
    extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    
    for ext in extensions:
        path = f"{base_path}/{image_id}{ext}"
        if os.path.exists(path):
            return path
    
    print(f"Warning: Image not found for ID {image_id}")
    return None

class EnsembleDetector:
    """
    Ensemble rule-based and deep learning predictions with WandB tracking
    Label Convention: 0 = FAKE, 1 = REAL
    """
    
    def __init__(self, rule_detector, dl_detector, ensemble_method='weighted_average', use_wandb=True):
        self.rule_detector = rule_detector
        self.dl_detector = dl_detector
        self.ensemble_method = ensemble_method
        self.use_wandb = use_wandb
        
        # Weights for ensemble (can be tuned based on validation performance)
        self.rule_weight = 0.2
        self.dl_weight = 0.8
        
        # Log ensemble configuration to WandB
        if self.use_wandb and wandb.run:
            wandb.config.update({
                "ensemble_method": ensemble_method,
                "rule_weight": self.rule_weight,
                "dl_weight": self.dl_weight
            })
        
    def predict(self, json_data_list, image_paths, headlines):
        """
        Ensemble predictions from both systems
        
        Args:
            json_data_list: List of JSON metadata (for rule-based)
            image_paths: List of image paths (for DL model)
            headlines: List of headlines (for DL model)
        
        Returns:
            Dictionary with predictions where 0=FAKE, 1=REAL
        """
        # Get rule-based predictions
        rule_preds, rule_probs, rule_details = self.rule_detector.predict_batch(json_data_list)
        
        # Get DL model predictions
        dl_preds, dl_probs = self.dl_detector.predict(image_paths, headlines)
        
        # Convert rule-based outputs
        rule_probs_real = 1 - rule_probs
        rule_preds_binary = (rule_probs_real > 0.5).astype(int)
        
        # Ensemble based on method
        if self.ensemble_method == 'weighted_average':
            ensemble_probs_real = (self.rule_weight * rule_probs_real + 
                                   self.dl_weight * dl_probs)
            ensemble_preds = (ensemble_probs_real > 0.5).astype(int)
            
        elif self.ensemble_method == 'voting':
            ensemble_preds = ((rule_preds_binary + dl_preds) >= 1).astype(int)
            ensemble_probs_real = (rule_probs_real + dl_probs) / 2
            
        elif self.ensemble_method == 'max':
            ensemble_probs_real = np.maximum(rule_probs_real, dl_probs)
            ensemble_preds = (ensemble_probs_real > 0.5).astype(int)
            
        elif self.ensemble_method == 'stacking':
            agreement = (rule_preds_binary == dl_preds)
            ensemble_probs_real = np.where(
                agreement,
                (rule_probs_real + dl_probs) / 2,
                np.maximum(rule_probs_real, dl_probs)
            )
            ensemble_preds = (ensemble_probs_real > 0.5).astype(int)
        
        return {
            'ensemble_predictions': ensemble_preds,
            'ensemble_probabilities': ensemble_probs_real,
            'rule_predictions': rule_preds_binary,
            'rule_probabilities': rule_probs_real,
            'dl_predictions': dl_preds,
            'dl_probabilities': dl_probs,
            'rule_details': rule_details,
            'rule_fake_probabilities': rule_probs
        }
    
    def evaluate(self, json_data_list, image_paths, headlines, true_labels, log_prefix="test"):
        """
        Evaluate ensemble performance with WandB logging
        
        Args:
            true_labels: Ground truth labels where 0=FAKE, 1=REAL
            log_prefix: Prefix for WandB logging (e.g., 'test', 'val')
        """
        results = self.predict(json_data_list, image_paths, headlines)
        
        # ==================== RULE-BASED METRICS ====================
        print("=" * 70)
        print("RULE-BASED SYSTEM PERFORMANCE")
        print("=" * 70)
        rule_report = classification_report(
            true_labels, results['rule_predictions'], 
            target_names=['Fake', 'Real'],
            output_dict=True,
            
        )
        print(classification_report(true_labels, results['rule_predictions'], 
                                   target_names=['Fake', 'Real']))
        rule_roc_auc = roc_auc_score(true_labels, results['rule_probabilities'])
        print(f"ROC-AUC: {rule_roc_auc:.3f}\n")
        
        rule_cm = confusion_matrix(true_labels, results['rule_predictions'])
        
        # ==================== DEEP LEARNING METRICS ====================
        print("=" * 70)
        print("DEEP LEARNING MODEL PERFORMANCE")
        print("=" * 70)
        dl_report = classification_report(
            true_labels, results['dl_predictions'],
            target_names=['Fake', 'Real'],
            output_dict=True,
           
        )
        print(classification_report(true_labels, results['dl_predictions'],
                                   target_names=['Fake', 'Real']))
        dl_roc_auc = roc_auc_score(true_labels, results['dl_probabilities'])
        print(f"ROC-AUC: {dl_roc_auc:.3f}\n")
        
        dl_cm = confusion_matrix(true_labels, results['dl_predictions'])
        
        # ==================== ENSEMBLE METRICS ====================
        print("=" * 70)
        print("ENSEMBLE SYSTEM PERFORMANCE")
        print("=" * 70)
        ensemble_report = classification_report(
            true_labels, results['ensemble_predictions'],
            target_names=['Fake', 'Real'],
            output_dict=True,
            
        )
        print(classification_report(true_labels, results['ensemble_predictions'],
                                   target_names=['Fake', 'Real']))
        ensemble_roc_auc = roc_auc_score(true_labels, results['ensemble_probabilities'])
        print(f"ROC-AUC: {ensemble_roc_auc:.3f}\n")
        
        ensemble_cm = confusion_matrix(true_labels, results['ensemble_predictions'])
        
        # ==================== LOG TO WANDB ====================
        if self.use_wandb and wandb.run:
            # Rule-based metrics
            wandb.log({
                f"{log_prefix}/rule_accuracy": round(rule_report['accuracy'], 3),
                f"{log_prefix}/rule_roc_auc": round(rule_roc_auc, 3),
                f"{log_prefix}/rule_precision_fake": round(rule_report['Fake']['precision'], 3),
                f"{log_prefix}/rule_recall_fake": round(rule_report['Fake']['recall'], 3),
                f"{log_prefix}/rule_f1_fake": round(rule_report['Fake']['f1-score'], 3),
                f"{log_prefix}/rule_precision_real": round(rule_report['Real']['precision'], 3),
                f"{log_prefix}/rule_recall_real": round(rule_report['Real']['recall'], 3),
                f"{log_prefix}/rule_f1_real": round(rule_report['Real']['f1-score'], 3),
                f"{log_prefix}/rule_confusion_matrix": wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=true_labels,
                    preds=results['rule_predictions'],
                    class_names=["Fake", "Real"],
                    title="Rule-Based System"
                )
            })
            
            # Deep learning metrics
            wandb.log({
                f"{log_prefix}/dl_accuracy": round(dl_report['accuracy'], 3),
                f"{log_prefix}/dl_roc_auc": round(dl_roc_auc, 3),
                f"{log_prefix}/dl_precision_fake": round(dl_report['Fake']['precision'], 3),
                f"{log_prefix}/dl_recall_fake": round(dl_report['Fake']['recall'], 3),
                f"{log_prefix}/dl_f1_fake": round(dl_report['Fake']['f1-score'], 3),
                f"{log_prefix}/dl_precision_real": round(dl_report['Real']['precision'], 3),
                f"{log_prefix}/dl_recall_real": round(dl_report['Real']['recall'], 3),
                f"{log_prefix}/dl_f1_real": round(dl_report['Real']['f1-score'], 3),
                f"{log_prefix}/dl_confusion_matrix": wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=true_labels,
                    preds=results['dl_predictions'],
                    class_names=["Fake", "Real"],
                    title="Deep Learning Model"
                )
            })
            
            # Ensemble metrics
            wandb.log({
                f"{log_prefix}/ensemble_accuracy": round(ensemble_report['accuracy'], 3),
                f"{log_prefix}/ensemble_roc_auc": round(ensemble_roc_auc, 3),
                f"{log_prefix}/ensemble_precision_fake": round(ensemble_report['Fake']['precision'], 3),
                f"{log_prefix}/ensemble_recall_fake": round(ensemble_report['Fake']['recall'], 3),
                f"{log_prefix}/ensemble_f1_fake": round(ensemble_report['Fake']['f1-score'], 3),
                f"{log_prefix}/ensemble_precision_real": round(ensemble_report['Real']['precision'], 3),
                f"{log_prefix}/ensemble_recall_real": round(ensemble_report['Real']['recall'], 3),
                f"{log_prefix}/ensemble_f1_real": round(ensemble_report['Real']['f1-score'], 3),
                f"{log_prefix}/ensemble_confusion_matrix": wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=true_labels,
                    preds=results['ensemble_predictions'],
                    class_names=["Fake", "Real"],
                    title="Ensemble System"
                )
            })
            
            # Comparative metrics table
            comparison_table = wandb.Table(
                columns=["System", "Accuracy", "ROC-AUC", "F1-Fake", "F1-Real"],
                data=[
                    ["Rule-Based", round(rule_report['accuracy'], 3), round(rule_roc_auc, 3), 
                     round(rule_report['Fake']['f1-score'], 3), round(rule_report['Real']['f1-score'], 3)],
                    ["Deep Learning", round(dl_report['accuracy'], 3), round(dl_roc_auc, 3),
                     round(dl_report['Fake']['f1-score'], 3), round(dl_report['Real']['f1-score'], 3)],
                    ["Ensemble", round(ensemble_report['accuracy'], 3), round(ensemble_roc_auc, 3),
                     round(ensemble_report['Fake']['f1-score'], 3), round(ensemble_report['Real']['f1-score'], 3)]
                ]
            )
            wandb.log({f"{log_prefix}/comparison_table": comparison_table})
            
            # Agreement analysis
            rule_dl_agreement = np.mean(results['rule_predictions'] == results['dl_predictions'])
            wandb.log({
                f"{log_prefix}/rule_dl_agreement": rule_dl_agreement
            })
        
        return results


# ============================================================================
# ENSEMBLE SAVE/LOAD FUNCTIONS
# ============================================================================

def save_ensemble_model(ensemble, rule_detector, dl_detector, save_dir='models'):
    """
    Save the complete ensemble model components for future use
    
    Args:
        ensemble: EnsembleDetector object
        rule_detector: RuleBasedDetectorWithExplainability object
        dl_detector: DeepLearningDetector object
        save_dir: Directory to save the ensemble components
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Save DL model
    dl_model_path = os.path.join(save_dir, 'dl_model.pth')
    dl_detector.save_model(dl_model_path)
    
    # Save rule detector and ensemble config
    ensemble_config = {
        'ensemble_method': ensemble.ensemble_method,
        'rule_weight': ensemble.rule_weight,
        'dl_weight': ensemble.dl_weight,
        'rule_optimal_weights': rule_detector.optimal_weights if hasattr(rule_detector, 'optimal_weights') else None,
        'rule_optimal_thresholds': rule_detector.optimal_thresholds if hasattr(rule_detector, 'optimal_thresholds') else None,
    }
    
    config_path = os.path.join(save_dir, 'ensemble_config.json')
    with open(config_path, 'w') as f:
        json.dump(ensemble_config, f, indent=2)
    
    # Save rule detector as pickle
    rule_detector_path = os.path.join(save_dir, 'rule_detector.pkl')
    with open(rule_detector_path, 'wb') as f:
        pickle.dump(rule_detector, f)
    
    print(f"✓ Ensemble components saved to {save_dir}/")
    print(f"  - DL model: {dl_model_path}")
    print(f"  - Rule detector: {rule_detector_path}")
    print(f"  - Configuration: {config_path}")


def load_ensemble_model(model_dir='models'):
    """
    Load a previously saved ensemble model components
    
    Args:
        model_dir: Directory containing the saved ensemble components
    
    Returns:
        Dictionary with reconstructed ensemble, rule_detector, and dl_detector
    """
    # Load ensemble config
    config_path = os.path.join(model_dir, 'ensemble_config.json')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Ensemble config not found at {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Load rule detector
    rule_detector_path = os.path.join(model_dir, 'rule_detector.pkl')
    with open(rule_detector_path, 'rb') as f:
        rule_detector = pickle.load(f)
    
    # Load DL detector
    dl_detector = DeepLearningDetector(device='cuda')
    dl_model_path = os.path.join(model_dir, 'dl_model.pth')
    dl_detector.load_model(dl_model_path)
    
    # Reconstruct ensemble
    ensemble = EnsembleDetector(
        rule_detector=rule_detector,
        dl_detector=dl_detector,
        ensemble_method=config['ensemble_method'],
        use_wandb=False
    )
    ensemble.rule_weight = config['rule_weight']
    ensemble.dl_weight = config['dl_weight']
    
    print(f"✓ Ensemble model loaded from {model_dir}/")
    print(f"  - Ensemble method: {config['ensemble_method']}")
    print(f"  - Rule weight: {config['rule_weight']}")
    print(f"  - DL weight: {config['dl_weight']}")
    
    return {
        'ensemble': ensemble,
        'rule_detector': rule_detector,
        'dl_detector': dl_detector,
        'config': config
    }


def predict_on_new_data(ensemble_package, json_data_list, image_paths, headlines):
    """
    Make predictions using a loaded ensemble model
    
    Args:
        ensemble_package: Dictionary returned by load_ensemble_model()
        json_data_list: List of JSON metadata
        image_paths: List of image paths
        headlines: List of headlines
    
    Returns:
        predictions dictionary
    """
    ensemble = ensemble_package['ensemble']
    return ensemble.predict(json_data_list, image_paths, headlines)


# ============================================================================
# MAIN PIPELINE WITH WANDB
# ============================================================================

if __name__ == "__main__":
    
    # ========================================================================
    # Initialize WandB
    # ========================================================================
    USE_WANDB = True
    WANDB_PROJECT = "fake-news-ensemble"
    
    if USE_WANDB:
        wandb.init(
            project=WANDB_PROJECT,
            name="ensemble-training-run",
            config={
                "framework": "ensemble",
                "dataset": "FakeReddit",
                "approach": "rule-based + deep-learning"
            }
        )
    
    # ========================================================================
    # STEP 1: Load and Prepare Data
    # ========================================================================
    print("="*80)
    print("STEP 1: LOADING DATA")
    print("="*80)
    
    df_train = pd.read_csv('datasets/processed/all_data_train.csv')
    df_test = pd.read_csv('datasets/processed/all_data_caption_analysis_summary.csv')
    print(len(df_test), len(df_train))
    
    # Load JSON metadata
    with open("datasets/processed/all_data_caption_analysis_results.json", "r") as file: 
        test_json_data = json.load(file)
    
    # Prepare test data
    test_images = [get_image_path(id) for id in df_test['image_filename']]
    test_headlines = df_test['clean_title'].tolist()
    test_labels = df_test['label'].tolist()

    train_labels = df_train['label'].tolist()
    
    print(f"Test samples: {len(test_labels)}")
    print(f"  FAKE (0): {test_labels.count(0)}")
    print(f"  REAL (1): {test_labels.count(1)}")
    
    print(f"Train samples: {len(train_labels)}")
    print(f"  FAKE (0): {train_labels.count(0)}")
    print(f"  REAL (1): {train_labels.count(1)}")
    # ========================================================================
    # STEP 2: Initialize Deep Learning Detector
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 2: INITIALIZING DEEP LEARNING MODEL")
    print("="*80)
    
    dl_detector = DeepLearningDetector(
        device='cuda',
        use_wandb=USE_WANDB,
        wandb_project=WANDB_PROJECT
    )
    
    # Load pre-trained model or train new one
    model_path = 'models/all_data_model.pth'
    train_model = False
    if os.path.exists(model_path):
        try:
            dl_detector.load_model(model_path)
            print("✓ Loaded pre-trained model")
        except Exception as e:
            print(f"⚠ Failed to load model: {e}")
            print("Training new model instead...")
            
    else:
        print(f"⚠ Model file '{model_path}' not found")
        print("Training new model...")
        train_model = True
    
    if train_model:
        train_images = [get_image_path(id) for id in df_train['image_filename']]
        train_headlines = df_train['clean_title'].tolist()
        train_labels = df_train['label'].tolist()
        
        # Split for validation (80-20)
        split_idx = int(0.7 * len(train_labels))
        val_images = train_images[split_idx:]
        val_headlines = train_headlines[split_idx:]
        val_labels = train_labels[split_idx:]
        train_images = train_images[:split_idx]
        train_headlines = train_headlines[:split_idx]
        train_labels = train_labels[:split_idx]
        
        dl_detector.train(
            train_images, 
            train_headlines, 
            train_labels,
            val_image_paths=val_images,
            val_headlines=val_headlines,
            val_labels=val_labels,
            batch_size=32,
            epochs=5,
            preload=True
        )
        dl_detector.save_model(model_path)
    
    # ========================================================================
    # STEP 3: Initialize Rule-Based Detector with SHAP
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 3: OPTIMIZING RULE-BASED DETECTOR")
    print("="*80)
    
    rule_detector = RuleBasedDetectorWithExplainability()
    
    print("\nAnalyzing features with SHAP...")
    shap_results = rule_detector.analyze_with_shap(
        test_json_data, 
        test_labels, 
        use_shap=True
    )
    
    print("\n✓ Rule detector optimized!")

    # ========================================================================
    # NEW: GENERATE SHAP VISUALIZATIONS
    # ========================================================================
    print("\n" + "="*80)
    print("GENERATING SHAP VISUALIZATIONS")
    print("="*80)

    from shap_visualizations import visualize_shap_analysis, visualize_shap_for_specific_samples

    # Generate all standard SHAP plots
    visualize_shap_analysis(rule_detector, save_dir='results/shap_plots')

    # Log SHAP plots to WandB
    if USE_WANDB:
        import os
        shap_dir = 'results/shap_plots'
        
        if os.path.exists(f'{shap_dir}/shap_summary_bar.png'):
            wandb.log({
                "shap/summary_bar": wandb.Image(f'{shap_dir}/shap_summary_bar.png'),
                "shap/summary_beeswarm": wandb.Image(f'{shap_dir}/shap_summary_beeswarm.png'),
                "shap/dependence_plots": wandb.Image(f'{shap_dir}/shap_dependence_plots.png'),
                "shap/decision_plot": wandb.Image(f'{shap_dir}/shap_decision_plot.png')
            })
            
            # Log individual force plots
            for i in range(5):
                if os.path.exists(f'{shap_dir}/shap_force_plot_sample_{i}.png'):
                    wandb.log({
                        f"shap/force_plot_sample_{i}": wandb.Image(f'{shap_dir}/shap_force_plot_sample_{i}.png')
                    })


    # Log rule configuration to WandB
    if USE_WANDB:
        wandb.config.update({
            "rule_optimization_method": shap_results['method'],
            "optimal_weights": rule_detector.optimal_weights,
            "optimal_thresholds": rule_detector.optimal_thresholds
        })
    
    # ========================================================================
    # STEP 4: Create and Evaluate Ensemble
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 4: ENSEMBLE EVALUATION")
    print("="*80)
    
    ensemble = EnsembleDetector(
        rule_detector, 
        dl_detector, 
        ensemble_method='weighted_average',
        use_wandb=USE_WANDB
    )
    
    results = ensemble.evaluate(
        test_json_data, 
        test_images, 
        test_headlines, 
        test_labels,
        log_prefix="test"
    )
    
    # ========================================================================
    # STEP 5: Save Results
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 5: SAVING RESULTS")
    print("="*80)
    
    # Save the complete ensemble model components
    save_ensemble_model(ensemble, rule_detector, dl_detector, save_dir='models')
    
    # Save configuration
    config = {
        'optimal_weights': rule_detector.optimal_weights,
        'optimal_thresholds': rule_detector.optimal_thresholds,
        'ensemble_weights': {
            'rule_weight': ensemble.rule_weight,
            'dl_weight': ensemble.dl_weight
        },
        'method': shap_results['method']
    }
    
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    with open('models/all_data_optimal_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Save predictions
    results_df = pd.DataFrame({
        'image_id': df_test['image_filename'],
        'headline': df_test['clean_title'],
        'true_label': test_labels,
        'ensemble_prediction': results['ensemble_predictions'],
        'ensemble_probability': results['ensemble_probabilities'],
        'rule_prediction': results['rule_predictions'],
        'dl_prediction': results['dl_predictions']
    })
    
    results_df.to_csv('results/all_data_ensemble_predictions.csv', index=False)
    
    # Log predictions as artifact to WandB
    if USE_WANDB:
        predictions_artifact = wandb.Artifact('predictions', type='predictions')
        predictions_artifact.add_file('results/all_data_ensemble_predictions.csv')
        wandb.log_artifact(predictions_artifact)
    
    print("✓ Results saved!")
    
    # ========================================================================
    # STEP 6: Detailed Sample Analysis
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 6: SAMPLE PREDICTIONS")
    print("="*80)
    
    # Create a table for WandB
    sample_predictions = []
    
    for i in range(min(10, len(test_labels))):
        pred_label = results['ensemble_predictions'][i]
        pred_prob = results['ensemble_probabilities'][i]
        true_label = test_labels[i]
        rule_detail = results['rule_details'][i]
        
        print(f"\n{'='*60}")
        print(f"Sample {i+1}: {df_test.iloc[i]['clean_title'][:60]}...")
        print(f"{'='*60}")
        print(f"True Label: {'REAL' if true_label == 1 else 'FAKE'}")
        print(f"Ensemble: {'REAL' if pred_label == 1 else 'FAKE'} ({pred_prob:.3f})")
        print(f"Rule-Based: {'REAL' if results['rule_predictions'][i] == 1 else 'FAKE'}")
        print(f"Deep Learning: {'REAL' if results['dl_predictions'][i] == 1 else 'FAKE'}")
        
        if rule_detail['reasons']:
            print(f"Suspicion Indicators: {', '.join(rule_detail['reasons'][:3])}")
        
        # Add to WandB table
        sample_predictions.append([
            i+1,
            df_test.iloc[i]['clean_title'][:50],
            'REAL' if true_label == 1 else 'FAKE',
            'REAL' if pred_label == 1 else 'FAKE',
            f"{pred_prob:.3f}",
            'REAL' if results['rule_predictions'][i] == 1 else 'FAKE',
            'REAL' if results['dl_predictions'][i] == 1 else 'FAKE',
            pred_label == true_label
        ])
    
    if USE_WANDB:
        samples_table = wandb.Table(
            columns=["Sample", "Headline", "True", "Ensemble", "Confidence", "Rule", "DL", "Correct"],
            data=sample_predictions
        )
        wandb.log({"sample_predictions": samples_table})
    
    # ========================================================================
    # Finish WandB
    # ========================================================================
    print("\n" + "="*80)
    print("PIPELINE COMPLETE!")
    print("="*80)
    
    if USE_WANDB:
        print(f"\nView results at: {wandb.run.get_url()}")
        wandb.finish()