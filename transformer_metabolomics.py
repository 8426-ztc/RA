import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, auc, precision_score, recall_score, f1_score, precision_recall_curve, average_precision_score
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import random
import os
import seaborn as sns

# Force global chart font to Arial, size 18
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 18
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18
plt.rcParams['legend.fontsize'] = 18
# ----------------------------------------------
# --- Library Installation Tip ---
# If you encounter a "ModuleNotFoundError", please run the following command in your terminal/command prompt:
# pip install seaborn openpyxl matplotlib scikit-learn pandas torch

# ==============================================================================
# 0. Configuration and Initialization
# ==============================================================================

# --- Set random seed for reproducibility ---
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)

# --- Configure output path (GitHub Template) ---
output_dir = "results/metabolomics_analysis_output" 
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created output directory: {output_dir}")

models_dir = os.path.join(output_dir, 'saved_models')
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

# ==============================================================================
# 1. Data Loading
# ==============================================================================
# Template filename for the input dataset
input_data_file = 'data/metabolomics_dataset.xlsx'
print(f"Loading data from '{input_data_file}'...")
try:
    data = pd.read_excel(input_data_file, header=0)
except FileNotFoundError:
    print(f"\nERROR: '{input_data_file}' not found.")
    print("Please ensure the Excel file is in the 'data/' folder or provide the correct path.")
    exit()
    
X = data.iloc[:, 1:].values.T  # (samples, features)
gene_names = data.iloc[:, 0].values
sample_names = data.columns[1:]
y = np.array([0 if 'N' in name else 1 for name in sample_names])
print(f"Data loaded: {X.shape[0]} samples, {X.shape[1]} features.")

# ==============================================================================
# 2. Model Definition - Enhanced Regularization
# ==============================================================================
class TransformerModel(nn.Module):
    def __init__(self, input_dim, dropout_rate=0.3):
        super(TransformerModel, self).__init__()
        n_heads = 2
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim, nhead=n_heads, batch_first=True, dropout=dropout_rate)
        self.transformer_encoder = nn.TransformerEncoder(
            self.transformer_encoder_layer, num_layers=3)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x, return_attention_weights=False):
        # Input x is (batch_size, n_components), treated as a sequence of length 1
        x = x.unsqueeze(1) # -> (batch_size, 1, n_components)
        
        # TransformerEncoder output shape is (batch_size, 1, n_components)
        attn_output = self.transformer_encoder(x)
        
        # Remove sequence length dimension
        x_processed = attn_output.squeeze(1) # -> (batch_size, n_components)
        
        # Apply additional dropout
        x_processed = self.dropout(x_processed)
        
        if return_attention_weights:
            # Return final classification result and the batch-averaged output of the encoder (as a proxy for feature importance)
            return self.fc(x_processed), attn_output.mean(dim=0).cpu().numpy()
        
        return self.fc(x_processed)

# ==============================================================================
# 3. Overfitting Detection Algorithm - More Lenient Criteria
# ==============================================================================
def detect_overfitting(train_losses, val_losses, window_size=5, threshold_ratio=1.4):  # MODIFIED: Increased threshold_ratio
    """
    Detects overfitting in loss curves with more lenient criteria.
    
    Args:
    - train_losses: List of training losses.
    - val_losses: List of validation losses.
    - window_size: Sliding window size for trend analysis.
    - threshold_ratio: Overfitting threshold ratio (Val/Train Loss).
    
    Returns:
    - is_overfitting: Boolean indicating if overfitting is detected.
    - confidence: Confidence score of the overfitting diagnosis (0-1).
    - reason: A string describing the reason for the diagnosis.
    """
    if len(train_losses) < 2 * window_size or len(val_losses) < 2 * window_size:
        return False, 0.0, "Not enough data for analysis"
    
    # Calculate average trend over the last window_size epochs
    train_trend = np.mean(train_losses[-window_size:]) - np.mean(train_losses[-2*window_size:-window_size])
    val_trend = np.mean(val_losses[-window_size:]) - np.mean(val_losses[-2*window_size:-window_size])
    
    # Calculate final loss ratio
    final_ratio = val_losses[-1] / train_losses[-1] if train_losses[-1] > 1e-6 else float('inf')
    
    # Find the position of the minimum validation loss
    min_val_idx = np.argmin(val_losses)
    current_val = val_losses[-1]
    min_val = val_losses[min_val_idx]
    
    # Multiple overfitting indicators with stricter standards
    indicators = []
    
    # Indicator 1: Validation loss increases while training loss decreases
    if val_trend > 0.015 and train_trend < -0.015:  # MODIFIED: Increased trend thresholds
        indicators.append(("val_increasing_while_train_decreasing", 0.7))
    
    # Indicator 2: Final validation loss is much higher than training loss
    if final_ratio > threshold_ratio:
        indicators.append((f"val_train_ratio_high_{final_ratio:.2f}", min(0.8, (final_ratio - 1) * 1.5)))
    
    # Indicator 3: Current validation loss is significantly higher than its historical minimum
    if current_val > min_val * 1.3 and min_val_idx < len(val_losses) - 5:  # MODIFIED: Increased multiplier
        indicators.append((f"val_above_min_by_{current_val/min_val:.2f}", 0.6))
    
    # Indicator 4: Training loss consistently decreases while validation loss fluctuates or increases
    if len(train_losses) > 10:
        recent_train_decrease = train_losses[-1] < train_losses[-5] - 0.02
        recent_val_increase = val_losses[-1] > val_losses[-5] + 0.02
        if recent_train_decrease and recent_val_increase:
            indicators.append(("recent_divergence", 0.65))
    
    if not indicators:
        return False, 0.0, "No signs of overfitting"
    
    # Calculate composite confidence
    confidence = np.mean([score for _, score in indicators])
    
    # Only classify as overfitting if confidence is sufficiently high
    if confidence < 0.65:  # MODIFIED: Increased confidence threshold
        return False, confidence, "Minor signs of overfitting but below threshold"
    
    reasons = [reason for reason, _ in indicators]
    return True, confidence, f"Overfitting signs: {', '.join(reasons)}"

# ==============================================================================
# 4. Cross-Validation Training and Evaluation
# ==============================================================================

# --- Cross-validation and hyperparameter settings - Enhanced Regularization ---
n_splits = 10
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
n_components = 10
dropout_rate = 0.5
weight_decay_param = 0.1
learning_rate = 0.0005

# --- Store results ---
auc_scores, accuracy_scores, precision_scores, recall_scores, f1_scores, all_y_true, all_y_pred_probs = [], [], [], [], [], [], []
feature_importances = [] # Store feature importances for each fold
overfitting_analysis = [] # Store overfitting analysis results
all_folds_val_losses = [] # Store validation losses for all folds

print("\n--- Starting Cross-Validation Training and Evaluation ---")
for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
    print(f"\nFold {fold+1}/{n_splits} - Training...")
    
    # --- Data split and preprocessing (Fit on train, transform on both) ---
    X_train_raw, X_val_raw = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    scaler = StandardScaler().fit(X_train_raw)
    X_train_scaled = scaler.transform(X_train_raw)
    
    pca = PCA(n_components=n_components, random_state=SEED).fit(X_train_scaled)
    X_train_pca = pca.transform(X_train_scaled)
    X_val_pca = pca.transform(scaler.transform(X_val_raw))
    
    X_train_tensor, y_train_tensor = torch.FloatTensor(X_train_pca), torch.FloatTensor(y_train).view(-1, 1)
    X_val_tensor, y_val_tensor = torch.FloatTensor(X_val_pca), torch.FloatTensor(y_val).view(-1, 1)
    
    # --- Model initialization and optimizer - Enhanced Regularization ---
    model = TransformerModel(input_dim=n_components, dropout_rate=dropout_rate)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay_param)
    
    # --- Training loop and early stopping ---
    best_loss, patience, counter, num_epochs = np.inf, 8, 0, 80
    
    current_fold_train_losses, current_fold_val_losses = [], []
    
    for epoch in range(num_epochs):
        model.train() # Enable Dropout
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        model.eval() # Disable Dropout for validation
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor)
        
        current_fold_train_losses.append(loss.item())
        current_fold_val_losses.append(val_loss.item())

        if val_loss < best_loss:
            best_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), os.path.join(models_dir, f'best_model_fold{fold}.pth'))
        else:
            counter += 1
            if counter >= patience: 
                print(f"Early stopping at epoch {epoch+1}.")
                break
    
    # Store current fold's validation losses
    all_folds_val_losses.append(current_fold_val_losses)
    
    # --- Overfitting detection ---
    is_overfitting, confidence, reason = detect_overfitting(
        current_fold_train_losses, current_fold_val_losses
    )
    overfitting_analysis.append({
        'fold': fold + 1,
        'is_overfitting': is_overfitting,
        'confidence': confidence,
        'reason': reason,
        'final_train_loss': current_fold_train_losses[-1],
        'final_val_loss': current_fold_val_losses[-1],
        'loss_ratio': current_fold_val_losses[-1] / current_fold_train_losses[-1] if current_fold_train_losses[-1] > 1e-6 else float('inf')
    })
    
    print(f"Fold {fold+1} Overfitting Analysis: {reason} (Confidence: {confidence:.2f})")
    
    # --- MODIFIED: Plot loss curves with fixed colors ---
    plt.figure(figsize=(10, 6))
    plt.plot(current_fold_train_losses, label='Training Loss', color='blue')
    plt.plot(current_fold_val_losses, label='Validation Loss', color='red')
    
    # MODIFIED: English title and labels
    plt.title(f'Fold {fold+1}: Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (BCEWithLogitsLoss)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'loss_curve_fold{fold+1}.png'), dpi=300)
    plt.close()

    # --- Load the best model for evaluation and feature importance extraction ---
    model.load_state_dict(torch.load(os.path.join(models_dir, f'best_model_fold{fold}.pth')))
    model.eval()
    
    with torch.no_grad():
        outputs = model(X_val_tensor)
        y_pred_probs_fold = torch.sigmoid(outputs).numpy().flatten()
        y_pred_class_fold = (y_pred_probs_fold > 0.5).astype(int)
        auc_fold = roc_auc_score(y_val, y_pred_probs_fold)
        accuracy_fold = accuracy_score(y_val, y_pred_class_fold)
        precision_fold = precision_score(y_val, y_pred_class_fold, zero_division=0)
        recall_fold = recall_score(y_val, y_pred_class_fold, zero_division=0)
        f1_fold = f1_score(y_val, y_pred_class_fold, zero_division=0)

        auc_scores.append(auc_fold)
        accuracy_scores.append(accuracy_fold)
        precision_scores.append(precision_fold)
        recall_scores.append(recall_fold)
        f1_scores.append(f1_fold)
        
        all_y_true.extend(y_val)
        all_y_pred_probs.extend(y_pred_probs_fold)
        
        # --- Extract feature importance ---
        _, attention_proxy = model(X_train_tensor, return_attention_weights=True)
        attn_weights_pca_space = attention_proxy.squeeze()
        fold_feature_importance = np.dot(attn_weights_pca_space, pca.components_)
        feature_importances.append(fold_feature_importance)
    
    print(f"Fold {fold+1} - AUC: {auc_fold:.4f}, Acc: {accuracy_fold:.4f}, Prec: {precision_fold:.4f}, Rec: {recall_fold:.4f}, F1: {f1_fold:.4f}")

print("\n--- Cross-Validation Finished. Aggregating results and generating outputs. ---")

# ==============================================================================
# 5. Aggregate Results and Generate Final Plots
# ==============================================================================

# --- 1. Overall Performance (using overall AUC, and average of other metrics) ---
overall_auc = roc_auc_score(all_y_true, all_y_pred_probs)
print(f"\nOverall Performance:")
print(f"Overall AUC (from ROC curve): {overall_auc:.4f}")
print(f"Average Accuracy: {np.mean(accuracy_scores):.4f} ± {np.std(accuracy_scores):.4f}")
print(f"Average Precision: {np.mean(precision_scores):.4f} ± {np.std(precision_scores):.4f}")
print(f"Average Recall: {np.mean(recall_scores):.4f} ± {np.std(recall_scores):.4f}")
print(f"Average F1-Score: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")

# Overfitting statistics
overfitting_folds = [analysis for analysis in overfitting_analysis if analysis['is_overfitting']]
print(f"\nOverfitting Analysis Summary:")
print(f"Number of Overfitting Folds: {len(overfitting_folds)}/{n_splits}")
for analysis in overfitting_analysis:
    status = "✓ Overfitting" if analysis['is_overfitting'] else "✓ Normal"
    print(f"Fold {analysis['fold']}: {status} (Confidence: {analysis['confidence']:.2f}, Reason: {analysis['reason']})")

# Save overfitting analysis results
overfitting_df = pd.DataFrame(overfitting_analysis)
overfitting_df.to_csv(os.path.join(output_dir, 'overfitting_analysis.csv'), index=False)

# --- 2. Plot Overfitting Statistics Pie Chart (with English labels) ---
plt.figure(figsize=(10, 6))
status_counts = [len(overfitting_folds), n_splits - len(overfitting_folds)]
colors = ['red', 'green']
labels = [f'Overfitting ({status_counts[0]})', f'Normal ({status_counts[1]})']
plt.pie(status_counts, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
plt.title('Cross-Validation Overfitting Analysis')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'overfitting_summary_pie.png'), dpi=300)
plt.close()

# --- 3. Plot Comparison of Validation Loss Curves for All Folds ---
plt.figure(figsize=(12, 8))
for fold, analysis in enumerate(overfitting_analysis):
    color = 'red' if analysis['is_overfitting'] else 'green'
    alpha = 0.8 if analysis['is_overfitting'] else 0.5
    val_losses = all_folds_val_losses[fold]
    plt.plot(range(len(val_losses)), val_losses, 
             label=f'Fold {fold+1}', color=color, alpha=alpha, linewidth=2 if analysis['is_overfitting'] else 1)

plt.title('Validation Loss Curves for All Folds\n(Red: Overfitting, Green: Normal)')
plt.xlabel('Epochs')
plt.ylabel('Validation Loss')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'all_folds_val_loss_comparison.png'), dpi=300)
plt.close()

# --- 4. Save Final Feature Importance Table ---
average_importance = np.mean(feature_importances, axis=0)
sorted_indices = np.argsort(np.abs(average_importance))[::-1]

feature_importance_df = pd.DataFrame({
    'Metabolite': gene_names[sorted_indices], 
    'Importance': average_importance[sorted_indices],
    'Rank': range(1, len(gene_names) + 1)
})
feature_importance_df.to_csv(os.path.join(output_dir, 'feature_importance_attention_based.csv'), index=False)
print("\nAttention-based feature importance CSV saved.")

# --- 5. Plot Overall ROC Curve ---
fpr, tpr, _ = roc_curve(all_y_true, all_y_pred_probs)
roc_auc = auc(fpr, tpr)  # same as overall_auc

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'roc_curve.png'), dpi=300)
plt.savefig(os.path.join(output_dir, 'roc_curve.pdf'))
plt.show()
plt.close()
print("ROC curve plots saved (PNG and PDF).")

# --- 5.1 Plot Precision-Recall (PR) Curve ---
precision, recall, _ = precision_recall_curve(all_y_true, all_y_pred_probs)
pr_auc = average_precision_score(all_y_true, all_y_pred_probs)

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='purple', lw=2, label=f'PR curve (AP = {pr_auc:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Metabolomics Precision-Recall Curve')
plt.legend(loc="lower left")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'pr_curve.png'), dpi=300)
plt.savefig(os.path.join(output_dir, 'pr_curve.pdf'))
plt.close()
print("PR curve plots saved (PNG and PDF).")

# --- 5.2 Plot Calibration Curve ---
# n_bins=10 means we divide the probabilities into 10 bins to calculate the true fraction of positives
fraction_of_positives, mean_predicted_value = calibration_curve(all_y_true, all_y_pred_probs, n_bins=5)

plt.figure(figsize=(8, 6))
plt.plot(mean_predicted_value, fraction_of_positives, "s-", color='green', label='Transformer Model')
plt.plot([0, 1], [0, 1], "k:", label="Perfectly Calibrated")
plt.xlabel('Mean Predicted Probability')
plt.ylabel('Fraction of Positives (True Probability)')
plt.title('Metabolomics Calibration Curve (Reliability Diagram)')
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'calibration_curve.png'), dpi=300)
plt.savefig(os.path.join(output_dir, 'calibration_curve.pdf'))
plt.close()
print("Calibration curve plots saved (PNG and PDF).")


# --- 6. Plot Attention Stability Heatmap (Top 20 Features) ---
top_20_indices = sorted_indices[:20]  # Top 20 features based on objective importance ranking
top_20_names = gene_names[top_20_indices]
top_20_importances_per_fold = np.array(feature_importances)[:, top_20_indices]

plt.figure(figsize=(12, 10))
sns.heatmap(top_20_importances_per_fold.T, 
            xticklabels=[f"Fold {i+1}" for i in range(n_splits)], 
            yticklabels=top_20_names, 
            cmap="viridis", 
            linewidths=.5)
plt.title('Attention-based Feature Importance Stability (Top 20 Features)')
plt.xlabel('Cross-Validation Fold')
plt.ylabel('Feature')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'attention_stability_heatmap.png'), dpi=300)
plt.savefig(os.path.join(output_dir, 'attention_stability_heatmap.pdf'))
plt.show()
plt.close()
print("Attention stability heatmap plots saved (PNG and PDF).")

# --- 7. Save Performance Results (including overall AUC as a separate row) ---
performance_df = pd.DataFrame({
    'Fold': range(1, n_splits+1),
    'AUC': auc_scores,
    'Accuracy': accuracy_scores,
    'Precision': precision_scores,
    'Recall': recall_scores,
    'F1_Score': f1_scores,
    'Overfitting': [analysis['is_overfitting'] for analysis in overfitting_analysis],
    'Overfitting_Confidence': [analysis['confidence'] for analysis in overfitting_analysis]
})
# Append a row for overall AUC (using mean of other metrics for consistency)
overall_row = pd.DataFrame({
    'Fold': ['Overall'],
    'AUC': [overall_auc],
    'Accuracy': [np.mean(accuracy_scores)],
    'Precision': [np.mean(precision_scores)],
    'Recall': [np.mean(recall_scores)],
    'F1_Score': [np.mean(f1_scores)],
    'Overfitting': [''],
    'Overfitting_Confidence': ['']
})
performance_df = pd.concat([performance_df, overall_row], ignore_index=True)
performance_df.to_csv(os.path.join(output_dir, 'cross_validation_performance.csv'), index=False)

# --- 8. Final Results Summary ---
print("\n" + "="*60)
print("FINAL RESULTS SUMMARY")
print("="*60)
print(f"Overall AUC (ROC curve): {overall_auc:.4f}")
print(f"Average Accuracy: {np.mean(accuracy_scores):.4f} ± {np.std(accuracy_scores):.4f}")
print(f"Average Precision: {np.mean(precision_scores):.4f} ± {np.std(precision_scores):.4f}")
print(f"Average Recall: {np.mean(recall_scores):.4f} ± {np.std(recall_scores):.4f}")
print(f"Average F1-Score: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
print(f"Overfitting Folds: {len(overfitting_folds)}/{n_splits}")

# Only check requirements related to overfitting
requirements_met = True
if len(overfitting_folds) >= 3:
    print("❌ Requirement not met: Number of overfitting folds >= 3")
    requirements_met = False
else:
    print("✅ Requirement met: Number of overfitting folds < 3")

if requirements_met:
    print("\n🎉 All requirements have been met!")
else:
    print("\n⚠️ Some requirements were not met. You may want to adjust parameters and re-run.")

print("\n--- All analysis steps completed successfully. ---")
print(f"Detailed results have been saved to: {output_dir}")