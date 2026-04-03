# =============================================================================
# 0. Setup and Imports
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import xgboost as xgb
from collections import Counter, defaultdict

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score, roc_curve
from sklearn.inspection import permutation_importance

print("Environment ready. Starting the standardized analysis pipeline.")

# =============================================================================
# 1. Global Configuration
# =============================================================================
# Template paths for GitHub (please update with your actual data path)
INPUT_FILE = "./data/input_data.xlsx"
OUTPUT_DIR = "./results"

os.makedirs(OUTPUT_DIR, exist_ok=True)
N_SPLITS = 5
RANDOM_STATE = 42

# Plot styling settings
plt.rcParams.update({
    'font.size': 12, 'axes.titlesize': 16, 'axes.labelsize': 14,
    'xtick.labelsize': 12, 'ytick.labelsize': 12, 'legend.fontsize': 11,
    'font.family': 'Arial', 'pdf.fonttype': 42
})

# Macaron color palette
MODEL_COLORS = {
    "KNN": (137/255, 207/255, 240/255),
    "SVM": (255/255, 182/255, 193/255),
    "RF": (152/255, 251/255, 152/255),
    "RFE-RF": (255/255, 218/255, 185/255),
    "XGBoost": (199/255, 177/255, 229/255)
}
models_order = ["KNN", "SVM", "RF", "RFE-RF", "XGBoost"]

# =============================================================================
# 2. Data Loading and Preprocessing
# =============================================================================
print(f"\n[Step 1/6] Loading data from '{INPUT_FILE}'...")
try:
    df = pd.read_excel(INPUT_FILE)
except FileNotFoundError:
    print(f"Error: File '{INPUT_FILE}' not found. Please check the file path.")
    exit()

label_column = df.columns[0]
X = df.drop(columns=[label_column])
y = df[label_column]

if y.dtype == 'object':
    le = LabelEncoder()
    y = pd.Series(le.fit_transform(y), name=label_column)

features = X.columns.tolist()
print(f"Data loaded: {X.shape[0]} samples, {X.shape[1]} features.")

# =============================================================================
# 3. Core Analysis Workflow
# =============================================================================
print("\n[Step 2/6] Executing core analysis pipeline...")

all_model_metrics = {}
all_feature_importances = {}
all_top10_feature_selections = defaultdict(list)
cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

# Models optimized for small-sample high-dimensional data
models = {
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVM": SVC(kernel='linear', C=1.0, probability=True, random_state=RANDOM_STATE),
    "RF": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=RANDOM_STATE),
    "RFE-RF": RFE(estimator=RandomForestClassifier(n_estimators=100, max_depth=5, random_state=RANDOM_STATE), n_features_to_select=10),
    "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=RANDOM_STATE,
                                 n_estimators=100, max_depth=5, learning_rate=0.05)
}

# Cross-validation and model evaluation
for model_name in models_order:
    model_instance = models[model_name]
    print(f"\n--- Training model: {model_name} ---")

    probas_list, true_labels_list = [], []
    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        scaler = StandardScaler().fit(X_train)
        X_train_scaled, X_test_scaled = scaler.transform(X_train), scaler.transform(X_test)

        if model_name == "RFE-RF":
            rfe_cv_model = RFE(estimator=RandomForestClassifier(n_estimators=50, max_depth=5, random_state=RANDOM_STATE), n_features_to_select=10)
            rfe_cv_model.fit(X_train_scaled, y_train)
            final_rf_cv = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=RANDOM_STATE)
            final_rf_cv.fit(X_train_scaled[:, rfe_cv_model.support_], y_train)
            y_prob = final_rf_cv.predict_proba(X_test_scaled[:, rfe_cv_model.support_])[:, 1]
        else:
            model_instance.fit(X_train_scaled, y_train)
            y_prob = model_instance.predict_proba(X_test_scaled)[:, 1]
        probas_list.append(y_prob)
        true_labels_list.append(y_test)

    y_true_all, y_prob_all = np.concatenate(true_labels_list), np.concatenate(probas_list)
    y_pred_all = (y_prob_all > 0.5).astype(int)
    all_model_metrics[model_name] = {
        'AUC': roc_auc_score(y_true_all, y_prob_all),
        'Accuracy': accuracy_score(y_true_all, y_pred_all),
        'Precision': precision_score(y_true_all, y_pred_all, zero_division=0),
        'Recall': recall_score(y_true_all, y_pred_all, zero_division=0),
        'F1 Score': f1_score(y_true_all, y_pred_all, zero_division=0)
    }
    pd.DataFrame({'true': y_true_all, 'prob': y_prob_all}).to_csv(os.path.join(OUTPUT_DIR, f"{model_name}_prob.csv"), index=False)
    print(f"  Evaluation completed. AUC: {all_model_metrics[model_name]['AUC']:.3f}")

    # Feature importance calculation
    print("  Training final model on full dataset for feature importance...")
    scaler_full = StandardScaler().fit(X)
    X_scaled_full = scaler_full.transform(X)
    importance_df = pd.DataFrame()
    final_model_for_importance = models[model_name]
    final_model_for_importance.fit(X_scaled_full, y)

    if model_name == 'RFE-RF':
        rf_on_selected_features = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=RANDOM_STATE)
        selected_indices = final_model_for_importance.support_
        rf_on_selected_features.fit(X_scaled_full[:, selected_indices], y)
        importances = rf_on_selected_features.feature_importances_
        full_importances = np.zeros(X.shape[1])
        np.place(full_importances, selected_indices, importances)
        importance_df = pd.DataFrame({'Feature': features, 'Importance': full_importances})
    elif model_name == 'KNN':
        perm_importance = permutation_importance(final_model_for_importance, X_scaled_full, y, n_repeats=10, random_state=RANDOM_STATE)
        importance_df = pd.DataFrame({'Feature': features, 'Importance': perm_importance.importances_mean})
    elif model_name == 'SVM':
        importance_df = pd.DataFrame({'Feature': features, 'Importance': np.abs(final_model_for_importance.coef_[0])})
    else:
        importance_df = pd.DataFrame({'Feature': features, 'Importance': final_model_for_importance.feature_importances_})

    importance_df = importance_df.sort_values(by="Importance", ascending=False).reset_index(drop=True)
    all_feature_importances[model_name] = importance_df
    importance_df.to_excel(os.path.join(OUTPUT_DIR, f"{model_name}_Feature_Importance.xlsx"), index=False)
    all_top10_feature_selections[model_name] = importance_df['Feature'].head(10).tolist()
    print(f"  Feature importance saved successfully.")

# =============================================================================
# 4. Result Aggregation
# =============================================================================
print("\n[Step 3/6] Aggregating performance and feature consistency results...")

performance_df = pd.DataFrame(all_model_metrics).T
performance_df = performance_df[['AUC', 'Accuracy', 'F1 Score', 'Precision', 'Recall']].round(3)
print("\n--- Model Performance Comparison (from 5-fold CV) ---")
print(performance_df)
performance_df.to_excel(os.path.join(OUTPUT_DIR, "Model_Performance_Comparison.xlsx"))

combined_top10_features_list = [feat for model in models_order for feat in all_top10_feature_selections[model]]
feature_counts = Counter(combined_top10_features_list)
consistency_df = pd.DataFrame(feature_counts.items(), columns=['Feature', 'Count_in_Top10'])
consistency_df = consistency_df.sort_values(by='Count_in_Top10', ascending=False).reset_index(drop=True)
print("\n--- Feature Consistency in Top 10 Across Models ---")
print(consistency_df)
consistency_df.to_excel(os.path.join(OUTPUT_DIR, "Feature_Consistency_Analysis.xlsx"))

# =============================================================================
# 5. Visualization 1: ROC Curves
# =============================================================================
print("\n[Step 4/6] Generating ROC curve comparison...")
plt.figure(figsize=(8, 8))
ax_roc = plt.gca()
for model_name in models_order:
    df_prob = pd.read_csv(os.path.join(OUTPUT_DIR, f"{model_name}_prob.csv"))
    fpr, tpr, _ = roc_curve(df_prob["true"], df_prob["prob"])
    roc_auc = all_model_metrics[model_name]['AUC']
    ax_roc.plot(fpr, tpr, lw=2.5, label=f'{model_name} (AUC = {roc_auc:.3f})', color=MODEL_COLORS[model_name])
ax_roc.plot([0, 1], [0, 1], 'k--', lw=1.5)
ax_roc.set(xlim=[-0.02, 1.0], ylim=[0.0, 1.05],
           xlabel='False Positive Rate (1 - Specificity)',
           ylabel='True Positive Rate (Sensitivity)',
           title=f'ROC Curves Comparison ({N_SPLITS}-fold CV)')
ax_roc.legend(loc="lower right", frameon=True)
ax_roc.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "Combined_ROC_Curve.png"), dpi=300)
plt.savefig(os.path.join(OUTPUT_DIR, "Combined_ROC_Curve.pdf"), format='pdf')
print("ROC curves saved.")
plt.close()

# =============================================================================
# 6. Visualization 2: Integrated Plot (Feature Importance + Performance)
# =============================================================================
print("\n[Step 5/6] Generating integrated analysis plot...")
fig = plt.figure(figsize=(20, 16))
gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.7], hspace=0.5, wspace=0.3)

def plot_feature_importance(ax, model_name, df):
    df_top10 = df.head(10).sort_values("Importance", ascending=True)
    sns.barplot(x="Importance", y="Feature", data=df_top10, color=MODEL_COLORS[model_name], ax=ax)
    ax.set_title(f"{model_name} Top 10 Features", pad=12)
    ax.set_xlabel("Permutation Importance" if model_name == "KNN" else "Feature Importance")
    ax.set_ylabel("")
    ax.tick_params(axis='y', labelsize=10)
    if not df_top10.empty:
        ax.set_xlim(right=df_top10['Importance'].max() * 1.2)
        for p in ax.patches:
            width = p.get_width()
            ax.text(width, p.get_y() + p.get_height() / 2, f' {width:.3f}', va='center', ha='left', fontsize=9)

axes_fi = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2]),
           fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
for i, model_name in enumerate(models_order):
    plot_feature_importance(axes_fi[i], model_name, all_feature_importances[model_name])
fig.add_subplot(gs[1, 2]).set_visible(False)

ax_table = fig.add_subplot(gs[2, :])
ax_table.axis('off')
ax_table.set_title('Model Performance Summary', pad=20, fontsize=18)
table_data = performance_df.reset_index().rename(columns={'index': 'Model'})
table = ax_table.table(cellText=table_data.values, colLabels=table_data.columns, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1, 1.8)

fig.suptitle('Integrated Analysis (Feature Importances & Performance)', fontsize=28, y=0.99)
fig.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(os.path.join(OUTPUT_DIR, "Integrated_Importance_Performance.png"), dpi=300)
plt.savefig(os.path.join(OUTPUT_DIR, "Integrated_Importance_Performance.pdf"), format='pdf')
print("Integrated plot saved.")
plt.close()

# =============================================================================
# 7. Visualization 3: Feature Consistency Matrix
# =============================================================================
print("\n[Step 6/6] Generating feature consistency heatmap...")
features_to_plot_sorted = consistency_df['Feature'].tolist()
num_features_to_plot = len(features_to_plot_sorted)

fig_consistency, ax = plt.subplots(figsize=(12, max(8, num_features_to_plot * 0.5)))

if num_features_to_plot > 0:
    for i, feature in enumerate(features_to_plot_sorted):
        for j, model_name in enumerate(models_order):
            if feature in all_top10_feature_selections[model_name]:
                rect = plt.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1,
                    facecolor=MODEL_COLORS[model_name],
                    edgecolor='white',
                    linewidth=2
                )
                ax.add_patch(rect)

    ax.set_yticks(np.arange(num_features_to_plot))
    ax.set_yticklabels(features_to_plot_sorted)
    ax.set_xticks(np.arange(len(models_order)))
    ax.set_xticklabels(models_order, rotation=45, ha='right')
    ax.set_xlim(-0.5, len(models_order) - 0.5)
    ax.set_ylim(num_features_to_plot - 0.5, -0.5)
    ax.set_title('Feature Consistency Across Models (Top 10)', pad=20)
    ax.set_xlabel("Model")
    ax.set_ylabel("Feature")
    legend_handles = [plt.Rectangle((0,0),1,1, fc=MODEL_COLORS[name]) for name in models_order]
    ax.legend(legend_handles, models_order, title="Model",
              loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)
    ax.grid(False)
else:
    ax.text(0.5, 0.5, "No consistent features in Top 10.", ha='center', va='center', transform=ax.transAxes)
    ax.set_axis_off()

fig_consistency.tight_layout(rect=[0, 0, 0.85, 1])
plt.savefig(os.path.join(OUTPUT_DIR, "Standalone_Feature_Consistency.png"), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(OUTPUT_DIR, "Standalone_Feature_Consistency.pdf"), format='pdf', bbox_inches='tight')
print("Feature consistency plot saved.")
plt.close()

print(f"\nAll analyses completed successfully!")
print(f"Results saved to directory: {OUTPUT_DIR}")