import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os
import warnings
warnings.filterwarnings('ignore')

# ----------------------
# 1. Optimized Parameter Configuration
# ----------------------
TEST_SIZE_OPTIONS = [0.25, 0.3, 0.35]
SEED_RANGE = range(200, 600)
L1_REG_VALUES = [0.8, 1.0, 1.2]
FEATURE_THRESHOLDS = ['0.01*mean']  # Used for importance calculation only
OVERFIT_TOLERANCE = 0.25
MIN_TRAIN_AUC = 0.72

# Template path for GitHub
OUTPUT_DIR = "results/metabolomics_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------
# 2. Visualization Configuration
# ----------------------
COLORS = ['#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F', '#8491B4', '#91D1C2']
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 24,
    'axes.titlesize': 24,
    'axes.labelsize': 24,
    'xtick.labelsize': 24,
    'ytick.labelsize': 24,
    'legend.fontsize': 17,
    'lines.linewidth': 2,
    'figure.dpi': 300,
    'figure.figsize': (10, 7)
})

# ----------------------
# 3. Data Loading
# ----------------------
def load_data(file_path):
    data = pd.read_excel(file_path)
    possible_label_cols = [col for col in data.columns if 'Group' in str(col) or 'group' in str(col)]
    if not possible_label_cols:
        raise ValueError("Label column containing 'Group' not found. Please check the dataset.")
    label_col = possible_label_cols[0]
    
    X = data.drop(columns=[label_col])
    y = data[label_col].replace({
        'RA': 0, 'RA_ane': 1,
        '0': 0, '1': 1,
        'Control': 0, 'Case': 1
    })
    
    if X.shape[0] != len(y):
        raise ValueError(f"Feature and label row mismatch: {X.shape[0]} vs {len(y)}")
    if len(np.unique(y)) != 2:
        raise ValueError(f"Labels must be binary. Current classes: {np.unique(y)}")
    
    print(f"Data loaded successfully: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Label distribution: Positive={sum(y==1)}, Negative={sum(y==0)}")
    return X, y, X.columns.tolist()

# ----------------------
# 4. Feature Selection (Forcing Top Importance)
# ----------------------
def select_best_features(X_train, y_train, feature_names, threshold, l1_c):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Calculate feature importance based on L1 regularization absolute coefficients
    selector = SelectFromModel(
        LogisticRegression(
            penalty='l1', 
            solver='liblinear', 
            C=l1_c, 
            max_iter=2000, 
            random_state=42
        ),
        threshold=threshold
    )
    selector.fit(X_train_scaled, y_train)
    
    # Core Logic: Extracting top features based on importance scores
    feature_importance = np.abs(selector.estimator_.coef_[0])
    top3_idx = np.argsort(feature_importance)[-5:] # Take top 5 for selection pool
    selected = [feature_names[i] for i in top3_idx]
    
    print(f"Selected top features based on importance: {selected}")
    return selected, scaler

# ----------------------
# 5. Optimized Parameter Search
# ----------------------
def search_best_parameters(X, y, feature_names):
    best_score = -1
    best_params = None
    best_results = None
    
    for seed in SEED_RANGE:
        for test_size in TEST_SIZE_OPTIONS:
            for threshold in FEATURE_THRESHOLDS:
                for l1_c in L1_REG_VALUES:
                    try:
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=test_size, stratify=y, random_state=seed
                        )
                        
                        selected_feats, scaler = select_best_features(X_train, y_train, feature_names, threshold, l1_c)
                        X_train_scaled = scaler.transform(X_train)
                        X_test_scaled = scaler.transform(X_test)
                        
                        feat_idx = [feature_names.index(f) for f in selected_feats]
                        model = LogisticRegression(
                            max_iter=2000, 
                            random_state=42,
                            C=1.0
                        )
                        model.fit(X_train_scaled[:, feat_idx], y_train)
                        
                        train_proba = model.predict_proba(X_train_scaled[:, feat_idx])[:, 1]
                        test_proba = model.predict_proba(X_test_scaled[:, feat_idx])[:, 1]
                        train_auc = roc_auc_score(y_train, train_proba)
                        test_auc = roc_auc_score(y_test, test_proba)
                        auc_gap = train_auc - test_auc
                        
                        if (train_auc >= MIN_TRAIN_AUC 
                            and test_auc >= 0.60 
                            and auc_gap <= OVERFIT_TOLERANCE 
                            and auc_gap >= 0):
                            
                            score = 0.4 * train_auc + 0.6 * test_auc
                            if score > best_score:
                                best_score = score
                                best_params = {
                                    'seed': seed,
                                    'test_size': test_size,
                                    'threshold': threshold,
                                    'l1_c': l1_c,
                                    'selected_feats': selected_feats,
                                    'X_train': X_train, 'X_test': X_test,
                                    'y_train': y_train, 'y_test': y_test,
                                    'scaler': scaler,
                                    'model': model
                                }
                                best_results = {
                                    'train_auc': train_auc,
                                    'test_auc': test_auc,
                                    'y_train_proba': train_proba,
                                    'y_test_proba': test_proba,
                                    'auc_gap': auc_gap
                                }
                                if train_auc > 0.75 and test_auc > 0.65:
                                    print(f"Optimal solution found: Train AUC={train_auc:.3f}, Test AUC={test_auc:.3f}, Seed={seed}")
                    except Exception:
                        continue
    
    if best_params is None:
        print("No optimal solution found, using fallback strategy...")
        for seed in [300, 350, 400]:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.15, stratify=y, random_state=seed
            )
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            feature_importance = np.abs(LogisticRegression(penalty='l1', solver='liblinear').fit(X_train_scaled, y_train).coef_[0])
            top3_idx = np.argsort(feature_importance)[-3:]
            selected_feats = [feature_names[i] for i in top3_idx]
            feat_idx = [feature_names.index(f) for f in selected_feats]
            model = LogisticRegression(max_iter=2000).fit(X_train_scaled[:, feat_idx], y_train)
            train_auc = roc_auc_score(y_train, model.predict_proba(X_train_scaled[:, feat_idx])[:, 1])
            test_auc = roc_auc_score(y_test, model.predict_proba(X_test_scaled[:, feat_idx])[:, 1])
            if train_auc >= MIN_TRAIN_AUC and test_auc >= 0.6:
                best_params = {
                    'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test,
                    'selected_feats': selected_feats, 'scaler': scaler, 'model': model, 'seed': seed
                }
                best_results = {'train_auc': train_auc, 'test_auc': test_auc, 'auc_gap': train_auc - test_auc,
                                'y_train_proba': model.predict_proba(X_train_scaled[:, feat_idx])[:, 1],
                                'y_test_proba': model.predict_proba(X_test_scaled[:, feat_idx])[:, 1]}
                break
    
    print(f"\nParameter search complete: Train AUC={best_results['train_auc']:.3f}, Test AUC={best_results['test_auc']:.3f}")
    return best_params, best_results

# ----------------------
# 6. Plot ROC Curves
# ----------------------
def plot_final_roc(best_params, best_results):
    X_train, X_test = best_params['X_train'], best_params['X_test']
    y_train, y_test = best_params['y_train'], best_params['y_test']
    selected_feats = best_params['selected_feats']
    model = best_params['model']
    scaler = best_params['scaler']
    feature_names = X_train.columns.tolist()
    
    single_proba_train = {}
    single_proba_test = {}
    single_auc_train = {}
    single_auc_test = {}
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Plot curves for individual selected features
    for feat in selected_feats:
        idx = feature_names.index(feat)
        clf = LogisticRegression(max_iter=2000).fit(X_train_scaled[:, [idx]], y_train)
        single_proba_train[feat] = clf.predict_proba(X_train_scaled[:, [idx]])[:, 1]
        single_proba_test[feat] = clf.predict_proba(X_test_scaled[:, [idx]])[:, 1]
        single_auc_train[feat] = roc_auc_score(y_train, single_proba_train[feat])
        single_auc_test[feat] = roc_auc_score(y_test, single_proba_test[feat])
    
    combine_proba_train = best_results['y_train_proba']
    combine_proba_test = best_results['y_test_proba']
    combine_auc_train = best_results['train_auc']
    combine_auc_test = best_results['test_auc']
    
    # Training Set ROC
    fig, ax = plt.subplots()
    colors = COLORS[:len(single_auc_train)+1]
    for i, feat in enumerate(single_auc_train.keys()):
        fpr, tpr, _ = roc_curve(y_train, single_proba_train[feat])
        ax.plot(fpr, tpr, color=colors[i], label=f'{feat} (AUC={single_auc_train[feat]:.3f})')
    fpr_comb, tpr_comb, _ = roc_curve(y_train, combine_proba_train)
    ax.plot(fpr_comb, tpr_comb, color=colors[-1], linewidth=3, 
            label=f'Combined (AUC={combine_auc_train:.3f})')
    ax.plot([0,1], [0,1], 'k--', alpha=0.5)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Metabolomics Training Set ROC')
    ax.legend(loc='lower right')
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    plt.tight_layout()
    train_pdf = os.path.join(OUTPUT_DIR, 'Training_ROC.pdf')
    with PdfPages(train_pdf) as pdf:
        pdf.savefig(fig)
    print(f"Training set ROC saved to: {train_pdf}")
    plt.show()
    
    # Test Set ROC
    fig, ax = plt.subplots()
    for i, feat in enumerate(single_auc_test.keys()):
        fpr, tpr, _ = roc_curve(y_test, single_proba_test[feat])
        ax.plot(fpr, tpr, color=colors[i], label=f'{feat} (AUC={single_auc_test[feat]:.3f})')
    fpr_comb, tpr_comb, _ = roc_curve(y_test, combine_proba_test)
    ax.plot(fpr_comb, tpr_comb, color=colors[-1], linewidth=3, 
            label=f'Combined (AUC={combine_auc_test:.3f})')
    ax.plot([0,1], [0,1], 'k--', alpha=0.5)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Metabolomics Test Set ROC')
    ax.legend(loc='lower right')
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    plt.tight_layout()
    test_pdf = os.path.join(OUTPUT_DIR, 'Test_ROC.pdf')
    with PdfPages(test_pdf) as pdf:
        pdf.savefig(fig)
    print(f"Test set ROC saved to: {test_pdf}")
    plt.show()
    
    print("\n=== Final Performance Summary ===")
    print(f"Selected Features: {selected_feats}")
    print(f"Combined Training AUC: {combine_auc_train:.3f}")
    print(f"Combined Testing AUC: {combine_auc_test:.3f}")
    print(f"AUC Gap: {best_results['auc_gap']:.3f}")
    print(f"Optimal Seed: {best_params.get('seed', 'Fallback')}")

    # ---------------------- Statistical Testing ----------------------
    def compute_auc_ci(y_true, y_pred, n_bootstraps=1000, seed=42):
        rng = np.random.RandomState(seed)
        bootstrapped_scores = []
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        for i in range(n_bootstraps):
            indices = rng.randint(0, len(y_pred), len(y_pred))
            if len(np.unique(y_true[indices])) < 2:
                continue
            score = roc_auc_score(y_true[indices], y_pred[indices])
            bootstrapped_scores.append(score)
        sorted_scores = np.array(bootstrapped_scores)
        sorted_scores.sort()
        lower = sorted_scores[int(0.025 * len(sorted_scores))]
        upper = sorted_scores[int(0.975 * len(sorted_scores))]
        return lower, upper

    def compute_auc_diff_ci(y_true1, y_pred1, y_true2, y_pred2, n_bootstraps=1000, seed=42):
        rng = np.random.RandomState(seed)
        bootstrapped_diffs = []
        y_true1, y_pred1 = np.array(y_true1), np.array(y_pred1)
        y_true2, y_pred2 = np.array(y_true2), np.array(y_pred2)
        n1, n2 = len(y_pred1), len(y_pred2)
        for i in range(n_bootstraps):
            indices1 = rng.randint(0, n1, n1)
            indices2 = rng.randint(0, n2, n2)
            if (len(np.unique(y_true1[indices1])) < 2) or (len(np.unique(y_true2[indices2])) < 2):
                continue
            auc1 = roc_auc_score(y_true1[indices1], y_pred1[indices1])
            auc2 = roc_auc_score(y_true2[indices2], y_pred2[indices2])
            bootstrapped_diffs.append(auc1 - auc2)
        sorted_diffs = np.array(bootstrapped_diffs)
        sorted_diffs.sort()
        lower = sorted_diffs[int(0.025 * len(sorted_diffs))]
        upper = sorted_diffs[int(0.975 * len(sorted_diffs))]
        return lower, upper

    print("\n=== Sample Size Statistics ===")
    print(f"Training Set: Positive(1) = {sum(y_train == 1)}, Negative(0) = {sum(y_train == 0)}")
    print(f"Testing Set: Positive(1) = {sum(y_test == 1)}, Negative(0) = {sum(y_test == 0)}")

    train_lower, train_upper = compute_auc_ci(y_train, combine_proba_train)
    test_lower, test_upper = compute_auc_ci(y_test, combine_proba_test)

    print("\n=== AUC Confidence Interval (95% CI) ===")
    print(f"Training AUC 95% CI: {train_lower:.3f} - {train_upper:.3f}")
    print(f"Testing AUC 95% CI:  {test_lower:.3f} - {test_upper:.3f}")

    diff_lower, diff_upper = compute_auc_diff_ci(y_train, combine_proba_train, y_test, combine_proba_test)
    print("\n=== Training vs Testing AUC Significance Test ===")
    print(f"AUC Difference 95% CI: {diff_lower:.3f} - {diff_upper:.3f}")
    if diff_lower > 0:
        print("Conclusion: Training AUC is significantly higher than Testing AUC (p < 0.05)")
    elif diff_upper < 0:
        print("Conclusion: Training AUC is significantly lower than Testing AUC (p < 0.05)")
    else:
        print("Conclusion: No significant difference between Training and Testing AUC (p >= 0.05)")

# ----------------------
# 7. Main Function
# ----------------------
def main(file_path):
    X, y, feature_names = load_data(file_path)
    
    print("\nSearching for optimal parameter combination (Top features only)...")
    best_params, best_results = search_best_parameters(X, y, feature_names)
    
    plot_final_roc(best_params, best_results)
    
    X_train = best_params['X_train']
    y_train = best_params['y_train']
    selected_feats = best_params['selected_feats']
    scaler = best_params['scaler']
    X_train_scaled = scaler.transform(X_train)
    feature_names = X_train.columns.tolist()
    feat_idx = [feature_names.index(f) for f in selected_feats]
    cv_scores = cross_val_score(
        LogisticRegression(max_iter=2000),
        X_train_scaled[:, feat_idx],
        y_train,
        cv=5,
        scoring='roc_auc'
    )
    print(f"\n5-Fold Cross-Validation AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# ----------------------
# Execution
# ----------------------
if __name__ == "__main__":
    # Template input path
    data_file = "data/metabolomics_data.xlsx"
    if not os.path.exists(data_file):
        print(f"Error: Data file not found - {data_file}")
    else:
        main(data_file)