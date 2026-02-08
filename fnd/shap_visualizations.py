import numpy as np
import matplotlib.pyplot as plt
import shap

def visualize_shap_analysis(detector, save_dir='results/shap_plots'):
    """
    Generate comprehensive SHAP visualizations for 5-feature model
    
    Features visualized:
    1. vgs_score (Visual Grounding Score)
    2. rouge_avg (Lexical Similarity)
    3. semantic_similarity (Semantic Similarity)
    4. kg_similarity (Knowledge Graph Similarity)
    5. coverage_ratio (Coverage Ratio)
    
    Args:
        detector: RuleBasedDetectorWithExplainability instance (after analyze_with_shap)
        save_dir: Directory to save plots
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    # Check if SHAP analysis was performed
    if not hasattr(detector, '_shap_values'):
        print("ERROR: No SHAP values found. Run analyze_with_shap(use_shap=True) first!")
        return
    
    shap_values_raw = detector._shap_values
    X_test = detector._X_test
    feature_names = detector._feature_names
    explainer = detector._explainer
    
    # Verify we have exactly 5 features
    assert len(feature_names) == 5, f"Expected 5 features, got {len(feature_names)}"
    
    print(f"\nGenerating SHAP visualizations for 5 features in '{save_dir}/'...")
    print(f"Features: {', '.join(feature_names)}")
    
    # FIX: Handle multi-output SHAP values (binary classification)
    if len(shap_values_raw.shape) == 3:
        print(f"  Detected multi-output SHAP values with shape {shap_values_raw.shape}")
        print(f"  Using SHAP values for class 1 (REAL)")
        shap_values = shap_values_raw[:, :, 1]  # Select class 1
    else:
        shap_values = shap_values_raw
    
    # Get base value
    if isinstance(explainer.expected_value, (list, np.ndarray)):
        base_value = explainer.expected_value[1] if len(explainer.expected_value) > 1 else explainer.expected_value[0]
    else:
        base_value = explainer.expected_value
    
    print(f"  SHAP values shape: {shap_values.shape}")
    print(f"  Base value: {base_value:.4f}")
    
    # ========================================================================
    # 1. SUMMARY PLOT (Bar) - Feature Importance
    # ========================================================================
    print("\n  1. Summary plot (bar) - Feature importance ranking...")
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values, 
        X_test, 
        feature_names=feature_names,
        plot_type="bar",
        show=False
    )
    plt.title('SHAP Feature Importance (5 Features)', fontsize=14, pad=15)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/shap_summary_bar.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # ========================================================================
    # 2. SUMMARY PLOT (Beeswarm) - Feature Impact Distribution
    # ========================================================================
    print("  2. Summary plot (beeswarm) - Impact distribution...")
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values, 
        X_test, 
        feature_names=feature_names,
        show=False
    )
    plt.title('SHAP Feature Impact Distribution (5 Features)', fontsize=14, pad=15)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/shap_summary_beeswarm.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # ========================================================================
    # 3. DEPENDENCE PLOTS - All 5 Features
    # ========================================================================
    print("  3. Dependence plots for all 5 features...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i in range(5):
        shap.dependence_plot(
            i,
            shap_values,
            X_test,
            feature_names=feature_names,
            ax=axes[i],
            show=False
        )
        axes[i].set_title(f'{feature_names[i]}', fontsize=12, fontweight='bold')
    
    # Hide the 6th subplot (we only have 5 features)
    axes[5].axis('off')
    
    plt.suptitle('SHAP Dependence Plots (All 5 Features)', fontsize=14, y=1.00)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/shap_dependence_plots.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # ========================================================================
    # 4. FORCE PLOTS - Individual Predictions
    # ========================================================================
    print("  4. Force plots for sample predictions...")
    
    num_samples = min(5, len(X_test))
    
    for i in range(num_samples):
        try:
            plt.figure(figsize=(20, 3))
            shap.plots.force(
                base_value,
                shap_values[i],
                X_test[i],
                feature_names=feature_names,
                matplotlib=True,
                show=False
            )
            plt.title(f'Force Plot - Sample {i}', fontsize=12, pad=10)
            plt.tight_layout()
            plt.savefig(f'{save_dir}/shap_force_plot_sample_{i}.png', dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"    Warning: Could not generate force plot for sample {i}: {e}")
            continue
    
    # ========================================================================
    # 5. WATERFALL PLOTS - Detailed Individual Explanations
    # ========================================================================
    print("  5. Waterfall plots for sample predictions...")
    for i in range(num_samples):
        try:
            plt.figure(figsize=(10, 7))
            
            # Create Explanation object for waterfall plot
            explanation = shap.Explanation(
                values=shap_values[i],
                base_values=base_value,
                data=X_test[i],
                feature_names=feature_names
            )
            
            shap.plots.waterfall(explanation, show=False)
            plt.title(f'Waterfall Plot - Sample {i}', fontsize=12, pad=10)
            plt.tight_layout()
            plt.savefig(f'{save_dir}/shap_waterfall_sample_{i}.png', dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"    Warning: Could not generate waterfall plot for sample {i}: {e}")
            continue
    
    # ========================================================================
    # 6. DECISION PLOT - Multiple Predictions
    # ========================================================================
    print("  6. Decision plot...")
    try:
        plt.figure(figsize=(10, 8))
        shap.decision_plot(
            base_value,
            shap_values[:20],  # First 20 samples
            X_test[:20],
            feature_names=feature_names,
            show=False
        )
        plt.title('SHAP Decision Plot (20 Samples)', fontsize=14, pad=15)
        plt.tight_layout()
        plt.savefig(f'{save_dir}/shap_decision_plot.png', dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"    Warning: Could not generate decision plot: {e}")
    
    # ========================================================================
    # 7. HEATMAP - Feature Values for Multiple Samples (All 5 Features)
    # ========================================================================
    print("  7. Heatmap of feature values (all 5 features)...")
    try:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Use all 5 features and first 20 samples
        heatmap_data = X_test[:20, :]
        
        im = ax.imshow(heatmap_data.T, aspect='auto', cmap='RdYlGn')
        
        # Set ticks
        ax.set_yticks(np.arange(5))
        ax.set_yticklabels(feature_names, fontsize=11)
        ax.set_xticks(np.arange(min(20, len(X_test))))
        ax.set_xticklabels(np.arange(min(20, len(X_test))))
        
        ax.set_xlabel('Sample Index', fontsize=12)
        ax.set_ylabel('Features', fontsize=12)
        ax.set_title('Feature Values Heatmap (5 Features, 20 Samples)', fontsize=14, pad=15)
        
        plt.colorbar(im, ax=ax, label='Feature Value')
        plt.tight_layout()
        plt.savefig(f'{save_dir}/shap_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"    Warning: Could not generate heatmap: {e}")
    
    # ========================================================================
    # 8. CUSTOM: Feature Weights Bar Chart
    # ========================================================================
    print("  8. Feature weights comparison...")
    try:
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        normalized_weights = mean_abs_shap / mean_abs_shap.sum()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, 5))
        bars = ax.barh(feature_names, normalized_weights * 100, color=colors)
        
        # Add value labels
        for i, (bar, weight) in enumerate(zip(bars, normalized_weights)):
            ax.text(weight * 100 + 1, i, f'{weight*100:.1f}%', 
                   va='center', fontsize=11, fontweight='bold')
        
        ax.set_xlabel('Feature Weight (%)', fontsize=12)
        ax.set_ylabel('Features', fontsize=12)
        ax.set_title('Optimized Feature Weights (5 Features)', fontsize=14, pad=15)
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/feature_weights.png', dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"    Warning: Could not generate weights plot: {e}")
    
    # ========================================================================
    # 9. CUSTOM: Feature Thresholds Comparison
    # ========================================================================
    print("  9. Feature thresholds visualization...")
    try:
        if detector.optimal_thresholds:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            thresholds = [detector.optimal_thresholds.get(name, 0) for name in feature_names]
            
            colors = plt.cm.plasma(np.linspace(0.3, 0.9, 5))
            bars = ax.barh(feature_names, thresholds, color=colors)
            
            # Add value labels
            for i, (bar, thresh) in enumerate(zip(bars, thresholds)):
                ax.text(thresh + 0.02, i, f'{thresh:.3f}', 
                       va='center', fontsize=11, fontweight='bold')
            
            ax.set_xlabel('Threshold Value', fontsize=12)
            ax.set_ylabel('Features', fontsize=12)
            ax.set_title('Optimized Feature Thresholds (5 Features)', fontsize=14, pad=15)
            ax.grid(axis='x', alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f'{save_dir}/feature_thresholds.png', dpi=300, bbox_inches='tight')
            plt.close()
    except Exception as e:
        print(f"    Warning: Could not generate thresholds plot: {e}")
    
    print(f"\n✓ All plots saved to '{save_dir}/'")
    print(f"\n  Generated plots:")
    print(f"  - shap_summary_bar.png (feature importance ranking)")
    print(f"  - shap_summary_beeswarm.png (impact distribution)")
    print(f"  - shap_dependence_plots.png (all 5 feature interactions)")
    print(f"  - shap_force_plot_sample_*.png (individual explanations)")
    print(f"  - shap_waterfall_sample_*.png (detailed breakdowns)")
    print(f"  - shap_decision_plot.png (multiple predictions)")
    print(f"  - shap_heatmap.png (feature values heatmap)")
    print(f"  - feature_weights.png (optimized weights)")
    print(f"  - feature_thresholds.png (optimized thresholds)")


def visualize_shap_for_specific_samples(detector, sample_indices, X_test, y_test, 
                                        predictions, save_dir='results/shap_plots'):
    """
    Generate SHAP visualizations for specific samples (e.g., misclassified ones)
    
    Args:
        detector: RuleBasedDetectorWithExplainability instance
        sample_indices: List of indices to visualize
        X_test: Test features
        y_test: True labels
        predictions: Model predictions
        save_dir: Directory to save plots
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    if not hasattr(detector, '_shap_values'):
        print("ERROR: No SHAP values found. Run analyze_with_shap(use_shap=True) first!")
        return
    
    shap_values_raw = detector._shap_values
    feature_names = detector._feature_names
    explainer = detector._explainer
    
    # Verify we have exactly 5 features
    assert len(feature_names) == 5, f"Expected 5 features, got {len(feature_names)}"
    
    # FIX: Handle multi-output SHAP values
    if len(shap_values_raw.shape) == 3:
        shap_values = shap_values_raw[:, :, 1]  # Select class 1
    else:
        shap_values = shap_values_raw
    
    # Get base value
    if isinstance(explainer.expected_value, (list, np.ndarray)):
        base_value = explainer.expected_value[1] if len(explainer.expected_value) > 1 else explainer.expected_value[0]
    else:
        base_value = explainer.expected_value
    
    print(f"\nGenerating SHAP plots for {len(sample_indices)} specific samples...")
    
    for idx in sample_indices:
        true_label = "REAL" if y_test[idx] == 1 else "FAKE"
        pred_label = "REAL" if predictions[idx] == 1 else "FAKE"
        status = "CORRECT" if y_test[idx] == predictions[idx] else "MISCLASSIFIED"
        
        print(f"\n  Sample {idx}: {status} - True: {true_label}, Pred: {pred_label}")
        
        # Waterfall plot
        try:
            plt.figure(figsize=(10, 7))
            
            explanation = shap.Explanation(
                values=shap_values[idx],
                base_values=base_value,
                data=X_test[idx],
                feature_names=feature_names
            )
            
            shap.plots.waterfall(explanation, show=False)
            plt.title(f"Sample {idx}: {status}\nTrue: {true_label}, Pred: {pred_label}", 
                     fontsize=12, pad=15)
            plt.tight_layout()
            plt.savefig(f'{save_dir}/sample_{idx}_{status.lower()}_waterfall.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"    Warning: Could not generate waterfall plot: {e}")
        
        # Force plot
        try:
            plt.figure(figsize=(20, 3))
            shap.plots.force(
                base_value,
                shap_values[idx],
                X_test[idx],
                feature_names=feature_names,
                matplotlib=True,
                show=False
            )
            plt.title(f"Sample {idx}: {status} - True: {true_label}, Pred: {pred_label}", 
                     fontsize=12, pad=10)
            plt.tight_layout()
            plt.savefig(f'{save_dir}/sample_{idx}_{status.lower()}_force.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"    Warning: Could not generate force plot: {e}")
    
    print(f"\n✓ Sample-specific plots saved to '{save_dir}/'")


def create_feature_comparison_plot(detector, sample_idx, save_dir='results/shap_plots'):
    """
    Create a detailed comparison plot showing feature values and SHAP contributions
    for all 5 features
    
    Args:
        detector: RuleBasedDetectorWithExplainability instance
        sample_idx: Index of sample to visualize
        save_dir: Directory to save plot
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    if not hasattr(detector, '_shap_values'):
        print("ERROR: No SHAP values found. Run analyze_with_shap(use_shap=True) first!")
        return
    
    shap_values_raw = detector._shap_values
    X_test = detector._X_test
    feature_names = detector._feature_names
    
    # Verify we have exactly 5 features
    assert len(feature_names) == 5, f"Expected 5 features, got {len(feature_names)}"
    
    # FIX: Handle multi-output SHAP values
    if len(shap_values_raw.shape) == 3:
        shap_values = shap_values_raw[:, :, 1]  # Select class 1
    else:
        shap_values = shap_values_raw
    
    shap_values_sample = shap_values[sample_idx]
    X_test_sample = X_test[sample_idx]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Sort by absolute SHAP value (all 5 features)
    sorted_idx = np.argsort(np.abs(shap_values_sample))[::-1]
    
    # Plot 1: Feature values
    colors1 = plt.cm.viridis(np.linspace(0.3, 0.9, 5))
    bars1 = ax1.barh(range(5), X_test_sample[sorted_idx], color=colors1)
    ax1.set_yticks(range(5))
    ax1.set_yticklabels([feature_names[i] for i in sorted_idx], fontsize=11)
    ax1.set_xlabel('Feature Value', fontsize=12)
    ax1.set_title('Feature Values\n(All 5 Features)', fontsize=13, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, bar in enumerate(bars1):
        width = bar.get_width()
        ax1.text(width + 0.01, i, f'{width:.3f}', 
                va='center', fontsize=10)
    
    # Plot 2: SHAP values
    colors2 = ['#d62728' if v > 0 else '#1f77b4' for v in shap_values_sample[sorted_idx]]
    bars2 = ax2.barh(range(5), shap_values_sample[sorted_idx], color=colors2)
    ax2.set_yticks(range(5))
    ax2.set_yticklabels([feature_names[i] for i in sorted_idx], fontsize=11)
    ax2.set_xlabel('SHAP Value (Impact on Prediction)', fontsize=12)
    ax2.set_title('SHAP Contributions\n(Red=Push to REAL, Blue=Push to FAKE)', 
                  fontsize=13, fontweight='bold')
    ax2.axvline(x=0, color='black', linestyle='--', linewidth=1.2)
    ax2.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, bar in enumerate(bars2):
        width = bar.get_width()
        label_x = width + (0.01 if width > 0 else -0.01)
        ha = 'left' if width > 0 else 'right'
        ax2.text(label_x, i, f'{width:.3f}', 
                va='center', ha=ha, fontsize=10)
    
    plt.suptitle(f'Feature Analysis for Sample {sample_idx}', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/feature_comparison_sample_{sample_idx}.png', 
               dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Feature comparison plot saved for sample {sample_idx}")


def create_5_feature_summary(detector, save_dir='results/shap_plots'):
    """
    Create a comprehensive summary visualization for all 5 features
    
    Args:
        detector: RuleBasedDetectorWithExplainability instance
        save_dir: Directory to save plot
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    if detector.optimal_weights is None or detector.optimal_thresholds is None:
        print("ERROR: No optimized weights/thresholds found. Run analyze_with_shap() first!")
        return
    
    feature_names = ['vgs_score', 'rouge_avg', 'semantic_similarity', 
                     'kg_similarity', 'coverage_ratio']
    
    # Get weights and thresholds
    weights = [detector.optimal_weights.get(name, 0.2) * 100 for name in feature_names]
    thresholds = [detector.optimal_thresholds.get(name, 0.5) for name in feature_names]
    
    # Create 2x2 subplot
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # 1. Feature Weights
    ax1 = fig.add_subplot(gs[0, 0])
    colors1 = plt.cm.viridis(np.linspace(0.3, 0.9, 5))
    bars1 = ax1.barh(feature_names, weights, color=colors1)
    for i, (bar, w) in enumerate(zip(bars1, weights)):
        ax1.text(w + 1, i, f'{w:.1f}%', va='center', fontsize=11, fontweight='bold')
    ax1.set_xlabel('Weight (%)', fontsize=12)
    ax1.set_title('Optimized Feature Weights', fontsize=13, fontweight='bold', pad=15)
    ax1.grid(axis='x', alpha=0.3)
    
    # 2. Feature Thresholds
    ax2 = fig.add_subplot(gs[0, 1])
    colors2 = plt.cm.plasma(np.linspace(0.3, 0.9, 5))
    bars2 = ax2.barh(feature_names, thresholds, color=colors2)
    for i, (bar, t) in enumerate(zip(bars2, thresholds)):
        ax2.text(t + 0.02, i, f'{t:.3f}', va='center', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Threshold Value', fontsize=12)
    ax2.set_title('Optimized Feature Thresholds', fontsize=13, fontweight='bold', pad=15)
    ax2.grid(axis='x', alpha=0.3)
    
    # 3. Feature Importance (if SHAP available)
    ax3 = fig.add_subplot(gs[1, 0])
    if hasattr(detector, 'feature_importance') and detector.feature_importance:
        importance = [detector.feature_importance.get(name, 0) for name in feature_names]
        colors3 = plt.cm.coolwarm(np.linspace(0.3, 0.9, 5))
        bars3 = ax3.barh(feature_names, importance, color=colors3)
        for i, (bar, imp) in enumerate(zip(bars3, importance)):
            ax3.text(imp + 0.001, i, f'{imp:.4f}', va='center', fontsize=11, fontweight='bold')
        ax3.set_xlabel('Importance Score', fontsize=12)
        ax3.set_title('Feature Importance (SHAP)', fontsize=13, fontweight='bold', pad=15)
        ax3.grid(axis='x', alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'Run analyze_with_shap(use_shap=True)\nfor importance scores', 
                ha='center', va='center', fontsize=12)
        ax3.set_title('Feature Importance', fontsize=13, fontweight='bold', pad=15)
    
    # 4. Summary Table
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('tight')
    ax4.axis('off')
    
    table_data = []
    for i, name in enumerate(feature_names):
        table_data.append([
            name,
            f'{weights[i]:.1f}%',
            f'{thresholds[i]:.3f}'
        ])
    
    table = ax4.table(cellText=table_data,
                     colLabels=['Feature', 'Weight', 'Threshold'],
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.5, 0.25, 0.25])
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    # Style header
    for i in range(3):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style rows
    for i in range(1, 6):
        for j in range(3):
            table[(i, j)].set_facecolor('#E7E6E6' if i % 2 == 0 else 'white')
    
    ax4.set_title('Configuration Summary', fontsize=13, fontweight='bold', pad=20)
    
    plt.suptitle('5-Feature Fake News Detector - Complete Summary', 
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.savefig(f'{save_dir}/5_feature_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ 5-feature summary visualization saved to '{save_dir}/5_feature_summary.png'")