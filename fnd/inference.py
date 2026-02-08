"""
Inference script to use the saved ensemble model on new test data
"""

import pandas as pd
import json
import os
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, confusion_matrix
from ensemble_detector import load_ensemble_model, predict_on_new_data, get_image_path


def run_inference(test_csv_path, json_metadata_path, model_dir='models'):
    """
    Load ensemble model and make predictions on new test set
    
    Args:
        test_csv_path: Path to test CSV file
        json_metadata_path: Path to JSON metadata file
        model_dir: Directory containing saved ensemble components
    """
    
    print("="*80)
    print("ENSEMBLE MODEL INFERENCE")
    print("="*80)
    
    # ========================================================================
    # Load test data
    # ========================================================================
    print("\nLoading test data...")
    df_test = pd.read_csv(test_csv_path)
    
    with open(json_metadata_path, "r") as file:
        test_json_data = json.load(file)
    
    # Prepare data
    test_images = [get_image_path(id) for id in df_test['image_filename']]
    test_headlines = df_test['clean_title'].tolist()
    
    if 'label' in df_test.columns:
        test_labels = df_test['label'].tolist()
        print(f"✓ Loaded {len(test_labels)} samples")
        print(f"  FAKE (0): {test_labels.count(0)}")
        print(f"  REAL (1): {test_labels.count(1)}")
    else:
        test_labels = None
        print(f"✓ Loaded {len(test_images)} samples (no labels provided)")
    
    # ========================================================================
    # Load ensemble model
    # ========================================================================
    print("\nLoading ensemble model...")
    try:
        ensemble_package = load_ensemble_model(model_dir)
    except FileNotFoundError as e:
        print(f"✗ Error: {e}")
        print(f"Make sure you've run ensemeble_detector.py to train and save the model first.")
        return
    
    ensemble = ensemble_package['ensemble']
    print(f"✓ Model loaded successfully")
    print(f"  Ensemble method: {ensemble.ensemble_method}")
    print(f"  Rule weight: {ensemble.rule_weight}")
    print(f"  DL weight: {ensemble.dl_weight}")
    
    # ========================================================================
    # Make predictions
    # ========================================================================
    print("\nMaking predictions...")
    results = predict_on_new_data(ensemble_package, test_json_data, test_images, test_headlines)
    
    # ========================================================================
    # Display results
    # ========================================================================
    print("\n" + "="*80)
    print("PREDICTION RESULTS")
    print("="*80)
    
    # Create results dataframe
    results_df = pd.DataFrame({
        'image_id': df_test['image_filename'],
        'headline': df_test['clean_title'],
        'ensemble_prediction': ['REAL' if p == 1 else 'FAKE' for p in results['ensemble_predictions']],
        'ensemble_probability': results['ensemble_probabilities'],
        'rule_prediction': ['REAL' if p == 1 else 'FAKE' for p in results['rule_predictions']],
        'dl_prediction': ['REAL' if p == 1 else 'FAKE' for p in results['dl_predictions']]
    })
    
    if test_labels is not None:
        results_df['true_label'] = ['REAL' if l == 1 else 'FAKE' for l in test_labels]
        results_df['correct'] = results['ensemble_predictions'] == test_labels
        
        # ====================================================================
        # Calculate and display metrics
        # ====================================================================
        y_true = test_labels
        y_pred = results['ensemble_predictions']
        
        accuracy = results_df['correct'].sum() / len(results_df)
        precision = precision_score(y_true, y_pred, average='binary')
        recall = recall_score(y_true, y_pred, average='binary')
        f1 = f1_score(y_true, y_pred, average='binary')
        
        print("\n" + "="*80)
        print("PERFORMANCE METRICS")
        print("="*80)
        print(f"Accuracy:  {accuracy:.3f}")
        print(f"Precision: {precision:.3f}")
        print(f"Recall:    {recall:.3f}")
        print(f"F1 Score:  {f1:.3f}")
        
        # Confusion matrix
        print("\n" + "="*80)
        print("CONFUSION MATRIX")
        print("="*80)
        cm = confusion_matrix(y_true, y_pred)
        print(f"                Predicted")
        print(f"                FAKE  REAL")
        print(f"Actual  FAKE    {cm[0][0]:4d}  {cm[0][1]:4d}")
        print(f"        REAL    {cm[1][0]:4d}  {cm[1][1]:4d}")
        
        # Detailed classification report
        print("\n" + "="*80)
        print("DETAILED CLASSIFICATION REPORT")
        print("="*80)
        target_names = ['FAKE', 'REAL']
        print(classification_report(y_true, y_pred, target_names=target_names))
        
        # Save metrics to file
        metrics_dict = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'confusion_matrix': cm.tolist(),
            'total_samples': len(y_true),
            'true_positives': int(cm[1][1]),
            'true_negatives': int(cm[0][0]),
            'false_positives': int(cm[0][1]),
            'false_negatives': int(cm[1][0])
        }
        
        os.makedirs('results', exist_ok=True)
        metrics_path = 'results/inference_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics_dict, f, indent=4)
        print(f"\n✓ Metrics saved to {metrics_path}")
    
    # Save results
    os.makedirs('results', exist_ok=True)
    output_path = 'results/inference_predictions.csv'
    results_df.to_csv(output_path, index=False)
    print(f"\n✓ Results saved to {output_path}")
    
    # Display sample predictions
    print("\n" + "="*80)
    print("SAMPLE PREDICTIONS (first 10)")
    print("="*80)
    
    for i in range(min(10, len(results_df))):
        row = results_df.iloc[i]
        print(f"\nSample {i+1}: {row['headline'][:60]}...")
        print(f"  Ensemble: {row['ensemble_prediction']} ({row['ensemble_probability']:.3f})")
        print(f"  Rule-Based: {row['rule_prediction']}")
        print(f"  Deep Learning: {row['dl_prediction']}")
        if 'true_label' in row:
            correct = "✓" if row['correct'] else "✗"
            print(f"  True Label: {row['true_label']} {correct}")
    
    print("\n" + "="*80)
    print("INFERENCE COMPLETE!")
    print("="*80)
    
    return results_df


if __name__ == "__main__":
    # Example usage
    test_csv = 'datasets/processed/all_data_caption_analysis_summary.csv'
    json_metadata = 'datasets/processed/all_data_caption_analysis_results.json'
    model_dir = 'models'
    
    results = run_inference(test_csv, json_metadata, model_dir)