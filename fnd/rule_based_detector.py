import numpy as np
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

class RuleBasedDetectorWithExplainability:
    """
    5-Feature Rule-based Fake News Detector
    
    LABEL CONVENTION: 0 = FAKE, 1 = REAL
    
    FEATURES:
    1. Visual Grounding Score (vgs_score)
    2. Lexical Similarity (rouge_avg)
    3. Semantic Similarity (semantic_similarity)
    4. Knowledge Graph Similarity (kg_similarity)
    5. Coverage Ratio (coverage_ratio)
    """
    
    def __init__(self):
        """Initialize with default thresholds for 5 features"""
        self.rules = {
            'vgs_threshold': 0.35,
            'rouge_threshold': 0.45,
            'semantic_threshold': 0.65,
            'kg_threshold': 0.55,
            'coverage_threshold': 0.55
        }
        
        self.feature_importance = None
        self.optimal_weights = None
        self.optimal_thresholds = None
        self.scaler_params = None
        self.decision_threshold = 0.5
        
    def extract_features_array(self, json_data_list):
        """Extract only the 5 core features"""
        features_list = []
        feature_names = [
            'vgs_score',
            'rouge_avg',
            'semantic_similarity',
            'kg_similarity',
            'coverage_ratio'
        ]
        
        for json_data in json_data_list:
            # Extract base metrics
            semantic = json_data['semantic']['semantic_similarity_score']
            kg_sim = json_data['kg_similarity']['kg_similarity_score']
            rouge1 = json_data['rouge1']['f1']
            rouge2 = json_data['rouge2']['f1']
            rougeL = json_data.get('rougeL', {}).get('f1', (rouge1 + rouge2) / 2)
            vgs = json_data['vgs']
            
            # Calculate coverage ratio
            covered = len(json_data['coverage']['covered'])
            missing = len(json_data['coverage']['missing'])
            extra = len(json_data['coverage']['extra'])
            total_tokens = covered + missing + extra + 1e-6
            coverage_ratio = covered / total_tokens
            
            # Calculate ROUGE average
            rouge_avg = (rouge1 + rouge2 + rougeL) / 3
            
            # Build feature vector
            features = [
                vgs,
                rouge_avg,
                semantic,
                kg_sim,
                coverage_ratio
            ]
            features_list.append(features)
        
        return np.array(features_list), feature_names
    
    def normalize_features(self, X, fit=True):
        """Normalize features using z-score normalization"""
        X = np.real(X)
        
        if fit:
            self.scaler_params = {
                'mean': np.mean(X, axis=0),
                'std': np.std(X, axis=0) + 1e-6
            }
        
        X_normalized = (X - self.scaler_params['mean']) / self.scaler_params['std']
        return np.real(X_normalized)
    
    def analyze_with_shap(self, json_data_list, true_labels, use_shap=False):
        """Train model and optimize weights/thresholds for 5 features"""
        print("\n" + "="*80)
        print(f"ANALYZING 5-FEATURE MODEL WITH {'SHAP' if use_shap else 'Random Forest'}")
        print("="*80)
        
        X, feature_names = self.extract_features_array(json_data_list)
        y = np.array(true_labels)
        
        print(f"\nDataset: {len(X)} samples, {X.shape[1]} features")
        print(f"Class distribution: FAKE={np.sum(y==0)} ({np.sum(y==0)/len(y)*100:.1f}%), "
              f"REAL={np.sum(y==1)} ({np.sum(y==1)/len(y)*100:.1f}%)")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
        
        # Normalize
        X_train_norm = self.normalize_features(X_train, fit=True)
        X_test_norm = self.normalize_features(X_test, fit=False)
        
        # Train Random Forest
        print("\n1. Training Random Forest Classifier...")
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='sqrt',
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train_norm, y_train)
        
        # Store trained model in analyze_with_shap()
        self._model = model
        
        # Evaluate
        train_pred = model.predict(X_train_norm)
        test_pred = model.predict(X_test_norm)
        
        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test, test_pred)
        
        print(f"   Train Accuracy: {train_acc:.3f}")
        print(f"   Test Accuracy: {test_acc:.3f}")
        
        precision, recall, f1, support = precision_recall_fscore_support(
            y_test, test_pred, average=None, labels=[0, 1]
        )
        
        print(f"\n   FAKE (0) - Precision: {precision[0]:.3f}, Recall: {recall[0]:.3f}, F1: {f1[0]:.3f}")
        print(f"   REAL (1) - Precision: {precision[1]:.3f}, Recall: {recall[1]:.3f}, F1: {f1[1]:.3f}")
        
        cm = confusion_matrix(y_test, test_pred)
        print(f"\n   Confusion Matrix:")
        print(f"   [[TN={cm[0,0]:3d}, FP={cm[0,1]:3d}]  <- FAKE predictions")
        print(f"    [FN={cm[1,0]:3d}, TP={cm[1,1]:3d}]]  <- REAL predictions")
        
        # Optimize decision threshold
        print("\n2. Optimizing decision threshold...")
        test_proba = model.predict_proba(X_test_norm)[:, 1]
        
        best_threshold = 0.5
        best_f1_fake = 0
        
        for threshold in np.arange(0.3, 0.7, 0.05):
            test_pred_adj = (test_proba >= threshold).astype(int)
            precision_adj, recall_adj, f1_adj, _ = precision_recall_fscore_support(
                y_test, test_pred_adj, average=None, labels=[0, 1]
            )
            if f1_adj[0] > best_f1_fake:
                best_f1_fake = f1_adj[0]
                best_threshold = threshold
        
        self.decision_threshold = best_threshold
        print(f"   Optimal threshold: {best_threshold:.3f} (FAKE F1: {best_f1_fake:.3f})")
        
        # Get feature importance
        if use_shap:
            try:
                results = self._analyze_with_shap_method(
                    model, X_train_norm, X_test_norm, y_train, y_test, feature_names
                )
            except ImportError:
                print("\n   WARNING: SHAP not installed. Using Random Forest importance instead.")
                print("           Install SHAP with: pip install shap")
                results = self._analyze_with_feature_importance(model, feature_names)
        else:
            results = self._analyze_with_feature_importance(model, feature_names)
        
        # Compute optimal thresholds
        print("\n3. Computing optimal thresholds using ROC analysis...")
        optimal_thresholds = self._compute_optimal_thresholds(X, y, feature_names)
        results['optimal_thresholds'] = optimal_thresholds
        
        # Store results
        self.feature_importance = results['importance_scores']
        self.optimal_weights = results['normalized_weights']
        self.optimal_thresholds = optimal_thresholds
        
        # Update rules
        self._update_rules_from_thresholds(optimal_thresholds)
        
        # Print summary
        self._print_summary(results, optimal_thresholds)
        
        return results
    
    def _compute_optimal_thresholds(self, X, y, feature_names):
        """Compute optimal thresholds for each feature using ROC analysis"""
        from sklearn.metrics import roc_curve
        
        optimal_thresholds = {}
        
        print("\n   Computing optimal thresholds for 5 features:")
        
        for i, feat_name in enumerate(feature_names):
            feature_values = X[:, i]
            
            # ROC curve analysis
            fpr, tpr, thresholds = roc_curve(y, feature_values, pos_label=1)
            
            # Youden's J statistic: maximize (TPR - 0.7*FPR)
            j_scores = tpr - 0.7 * fpr
            best_idx = np.argmax(j_scores)
            best_threshold = thresholds[best_idx]
            
            optimal_thresholds[feat_name] = float(best_threshold)
            print(f"   {feat_name:25s}: {best_threshold:.3f} (J={j_scores[best_idx]:.3f})")
        
        return optimal_thresholds
    
    def _analyze_with_shap_method(self, model, X_train, X_test, y_train, y_test, feature_names):
        """Analyze using SHAP (if available)"""
        try:
            import shap
            
            print("\n   Computing SHAP values...")
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
            
            if isinstance(shap_values, list):
                shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            
            if len(shap_values.shape) == 3:
                shap_values = shap_values[:, :, 1]
            
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            normalized_weights = mean_abs_shap / mean_abs_shap.sum()
            
            importance_scores = {name: float(score) for name, score in zip(feature_names, mean_abs_shap)}
            normalized_weights_dict = {name: float(weight) for name, weight in zip(feature_names, normalized_weights)}
            
            print("\n   SHAP Feature Importance:")
            for name, score in importance_scores.items():
                print(f"   {name:25s}: {score:.4f} ({normalized_weights_dict[name]*100:.1f}%)")
            
            # Store SHAP values for visualization
            self._shap_values = shap_values
            self._X_test = X_test
            self._feature_names = feature_names
            self._explainer = explainer
            
            return {
                'method': 'SHAP',
                'importance_scores': importance_scores,
                'normalized_weights': normalized_weights_dict,
                'shap_values': shap_values,
                'explainer': explainer
            }
            
        except ImportError:
            print("\n   SHAP not available, using Random Forest importance")
            return self._analyze_with_feature_importance(model, feature_names)
    
    def _analyze_with_feature_importance(self, model, feature_names):
        """Use Random Forest feature importance"""
        importance_scores = dict(zip(feature_names, model.feature_importances_))
        normalized_weights = dict(zip(feature_names, 
                                     model.feature_importances_ / model.feature_importances_.sum()))
        
        print("\n   Random Forest Feature Importance:")
        for name, score in importance_scores.items():
            print(f"   {name:25s}: {score:.4f} ({normalized_weights[name]*100:.1f}%)")
        
        print("\n   NOTE: For SHAP visualizations, run analyze_with_shap(use_shap=True)")
        print("         Install SHAP first: pip install shap")
        
        return {
            'method': 'Random Forest',
            'importance_scores': importance_scores,
            'normalized_weights': normalized_weights
        }
    
    def _update_rules_from_thresholds(self, optimal_thresholds):
        """Update internal rules from computed thresholds"""
        threshold_map = {
            'vgs_score': 'vgs_threshold',
            'rouge_avg': 'rouge_threshold',
            'semantic_similarity': 'semantic_threshold',
            'kg_similarity': 'kg_threshold',
            'coverage_ratio': 'coverage_threshold'
        }
        
        for feat_name, threshold in optimal_thresholds.items():
            if feat_name in threshold_map:
                self.rules[threshold_map[feat_name]] = threshold
    
    def _print_summary(self, results, optimal_thresholds):
        """Print final summary"""
        print("\n" + "="*80)
        print("OPTIMAL CONFIGURATION SUMMARY (5 FEATURES)")
        print("="*80)
        
        print("\nOptimized Feature Weights:")
        sorted_weights = sorted(results['normalized_weights'].items(), 
                               key=lambda x: x[1], reverse=True)
        for name, weight in sorted_weights:
            print(f"  {name:25s}: {weight:.3f} ({weight*100:.1f}%)")
        
        print("\nOptimized Thresholds:")
        for feat_name in ['vgs_score', 'rouge_avg', 'semantic_similarity', 
                          'kg_similarity', 'coverage_ratio']:
            if feat_name in optimal_thresholds:
                print(f"  {feat_name:25s}: {optimal_thresholds[feat_name]:.3f}")
        
        print(f"\nDecision Threshold: {self.decision_threshold:.3f}")
    
    def apply_rules_with_learned_weights(self, json_data):
        """
        Apply weighted scoring using trained Random Forest model
        """
        if self.optimal_weights is None:
            print("Warning: No learned weights. Run analyze_with_shap() first.")
            return self._apply_basic_rules(json_data)
        
        if not hasattr(self, '_model') or self._model is None:
            print("Warning: No trained model available.")
            return self._apply_basic_rules(json_data)
        
        features = self._extract_features(json_data)
        
        # Convert to feature array (in same order as training)
        feature_array = np.array([
            [
                features['vgs_score'],
                features['rouge_avg'],
                features['semantic_similarity'],
                features['kg_similarity'],
                features['coverage_ratio']
            ]
        ])
        
        # Normalize using training parameters
        if self.scaler_params:
            feature_array_norm = (feature_array - self.scaler_params['mean']) / self.scaler_params['std']
        else:
            feature_array_norm = feature_array
        
        # Get probability from trained Random Forest
        real_probability = self._model.predict_proba(feature_array_norm)[0, 1]
        fake_probability = 1 - real_probability
        
        # Make prediction using optimized threshold
        prediction = 1 if real_probability >= self.decision_threshold else 0
        confidence = abs(real_probability - self.decision_threshold)
        
        # Generate reasons based on feature thresholds
        reasons = []
        for feat_name in ['vgs_score', 'rouge_avg', 'semantic_similarity', 'kg_similarity', 'coverage_ratio']:
            threshold_key = feat_name.replace('_score', '_threshold').replace('_ratio', '_threshold')
            threshold = self.rules.get(threshold_key, 0.5)
            feat_value = features[feat_name]
            
            if feat_value < threshold:
                deficit = (threshold - feat_value) / (threshold + 1e-6)
                reasons.append(f"{feat_name}: {feat_value:.3f} < {threshold:.3f} (deficit: {deficit:.1%})")
        
        return {
            'fake_probability': fake_probability,
            'real_probability': real_probability,
            'prediction': prediction,
            'confidence': min(confidence, 1.0),
            'reasons': reasons if prediction == 0 else ["All checks passed"],
            'features': features,
            'weights_used': self.optimal_weights
        }
    
    def _extract_features(self, json_data):
        """Extract the 5 features from JSON data"""
        semantic = json_data['semantic']['semantic_similarity_score']
        kg_sim = json_data['kg_similarity']['kg_similarity_score']
        rouge1 = json_data['rouge1']['f1']
        rouge2 = json_data['rouge2']['f1']
        rougeL = json_data.get('rougeL', {}).get('f1', (rouge1 + rouge2) / 2)
        vgs = json_data['vgs']
        
        covered = len(json_data['coverage']['covered'])
        missing = len(json_data['coverage']['missing'])
        extra = len(json_data['coverage']['extra'])
        total_tokens = covered + missing + extra + 1e-6
        
        coverage_ratio = covered / total_tokens
        rouge_avg = (rouge1 + rouge2 + rougeL) / 3
        
        return {
            'vgs_score': vgs,
            'rouge_avg': rouge_avg,
            'semantic_similarity': semantic,
            'kg_similarity': kg_sim,
            'coverage_ratio': coverage_ratio
        }
    
    def _apply_basic_rules(self, json_data):
        """Fallback basic rules for 5 features"""
        features = self._extract_features(json_data)
        
        suspicion_score = 0
        reasons = []
        weights = [0.20, 0.20, 0.20, 0.20, 0.20]  # Equal weights
        
        checks = [
            ('vgs_score', 0.35),
            ('rouge_avg', 0.45),
            ('semantic_similarity', 0.65),
            ('kg_similarity', 0.55),
            ('coverage_ratio', 0.55)
        ]
        
        for (feat_name, threshold), weight in zip(checks, weights):
            if features[feat_name] < threshold:
                suspicion_score += weight
                reasons.append(f"{feat_name}: {features[feat_name]:.3f} < {threshold:.3f}")
        
        prediction = 0 if suspicion_score > 0.5 else 1
        
        return {
            'fake_probability': suspicion_score,
            'prediction': prediction,
            'confidence': abs(suspicion_score - 0.5) * 2,
            'reasons': reasons,
            'features': features
        }
    
    def predict_batch(self, json_data_list):
        """Batch prediction"""
        results = [self.apply_rules_with_learned_weights(json_data) 
                   for json_data in json_data_list]
        
        predictions = np.array([r['prediction'] for r in results])
        probabilities = np.array([r['fake_probability'] for r in results])
        
        return predictions, probabilities, results