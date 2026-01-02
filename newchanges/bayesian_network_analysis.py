import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (classification_report, confusion_matrix, 
                             accuracy_score, precision_score, recall_score, 
                             f1_score, roc_curve, auc, roc_auc_score)
from sklearn.naive_bayes import GaussianNB, CategoricalNB
from sklearn.preprocessing import KBinsDiscretizer
import warnings
warnings.filterwarnings('ignore')

# Load the cleaned dataset
df = pd.read_excel('dataset_limpo.xlsx')

print("="*80)
print("BAYESIAN NETWORK ANALYSIS - PREDICTING CARDIAC DISEASES IN CHILDREN")
print("="*80)

# ============================================================================
# 1. FEATURE SELECTION & DATA PREPARATION
# ============================================================================
print("\n" + "="*80)
print("1. FEATURE SELECTION & DATA PREPARATION")
print("="*80)

# Use same features as Decision Tree for fair comparison
selected_features = ['SOPRO', 'FC', 'IMC', 'MOTIVO1', 'MOTIVO2', 'B2', 'PA_SISTOLICA']

# Prepare dataset
df_model = df[selected_features + ['NORMAL_X_ANORMAL_BIN']].copy()

# Remove rows with missing target variable
df_model = df_model[df_model['NORMAL_X_ANORMAL_BIN'].notna()]

print(f"\nDataset size: {len(df_model)}")
print(f"\nClass distribution:")
print(df_model['NORMAL_X_ANORMAL_BIN'].value_counts())
print(f"\nClass balance: {df_model['NORMAL_X_ANORMAL_BIN'].value_counts(normalize=True).round(3).to_dict()}")

# Separate numeric and categorical features
numeric_features = ['FC', 'IMC', 'PA_SISTOLICA']
categorical_features = ['SOPRO', 'MOTIVO1', 'MOTIVO2', 'B2']

print(f"\nNumeric features: {numeric_features}")
print(f"Categorical features: {categorical_features}")

# ============================================================================
# 2. DATA PREPROCESSING
# ============================================================================
print("\n" + "="*80)
print("2. DATA PREPROCESSING")
print("="*80)

# Handle missing values
# For numeric: impute with median
for col in numeric_features:
    if df_model[col].isnull().any():
        median_val = df_model[col].median()
        df_model[col].fillna(median_val, inplace=True)
        print(f"Imputed {col} with median: {median_val:.2f}")

# For categorical: impute with 'Missing' category
for col in categorical_features:
    if df_model[col].isnull().any():
        df_model[col].fillna('Missing', inplace=True)
        print(f"Imputed {col} with 'Missing' category")

# Encode categorical variables
le_dict = {}
for col in categorical_features:
    le = LabelEncoder()
    df_model[f'{col}_encoded'] = le.fit_transform(df_model[col].astype(str))
    le_dict[col] = le
    print(f"\n{col}: {len(le.classes_)} categories")

# ============================================================================
# 3. DISCRETIZATION OF CONTINUOUS VARIABLES
# ============================================================================
print("\n" + "="*80)
print("3. DISCRETIZATION (For Categorical Naive Bayes)")
print("="*80)

print("\nBayesian Networks often work better with discretized variables.")
print("Creating discretized versions of numeric features...")

# Create discretized versions of numeric features
discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')

numeric_cols_discrete = []
for col in numeric_features:
    df_model[f'{col}_discrete'] = discretizer.fit_transform(df_model[[col]]).astype(int)
    numeric_cols_discrete.append(f'{col}_discrete')
    
    # Show bin edges
    bins = discretizer.bin_edges_[0]
    print(f"\n{col} discretized into 5 bins:")
    for i in range(len(bins)-1):
        print(f"  Bin {i}: [{bins[i]:.2f}, {bins[i+1]:.2f})")

# ============================================================================
# 4. TRAIN-TEST SPLIT
# ============================================================================
print("\n" + "="*80)
print("4. TRAIN-TEST SPLIT")
print("="*80)

# Prepare feature matrices
categorical_encoded = [f'{col}_encoded' for col in categorical_features]

# For Gaussian NB (continuous features)
X_continuous = df_model[numeric_features + categorical_encoded].values

# For Categorical NB (all discrete)
X_discrete = df_model[numeric_cols_discrete + categorical_encoded].values

# Target variable
y = df_model['NORMAL_X_ANORMAL_BIN'].values

# Split data
X_cont_train, X_cont_test, y_train, y_test = train_test_split(
    X_continuous, y, test_size=0.3, random_state=42, stratify=y
)

X_disc_train, X_disc_test, _, _ = train_test_split(
    X_discrete, y, test_size=0.3, random_state=42, stratify=y
)

print(f"Training set size: {len(X_cont_train)} ({len(X_cont_train)/len(X_continuous)*100:.1f}%)")
print(f"Test set size: {len(X_cont_test)} ({len(X_cont_test)/len(X_continuous)*100:.1f}%)")

# ============================================================================
# 5. MODEL 1: GAUSSIAN NAIVE BAYES
# ============================================================================
print("\n" + "="*80)
print("5. GAUSSIAN NAIVE BAYES (Continuous Features)")
print("="*80)

print("\nGaussian Naive Bayes assumes features follow normal distributions.")
print("Suitable for continuous numeric variables.\n")

# Train model
gnb = GaussianNB()
gnb.fit(X_cont_train, y_train)

# Predictions
y_pred_gnb = gnb.predict(X_cont_test)
y_pred_proba_gnb = gnb.predict_proba(X_cont_test)[:, 1]

# Evaluation metrics
print("--- Gaussian Naive Bayes Performance ---")
print(f"Training Accuracy: {gnb.score(X_cont_train, y_train):.4f}")
print(f"Test Accuracy: {accuracy_score(y_test, y_pred_gnb):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_gnb):.4f}")
print(f"Recall (Sensitivity): {recall_score(y_test, y_pred_gnb):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred_gnb):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba_gnb):.4f}")

print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred_gnb, 
                          target_names=['Normal (0)', 'Anormal (1)']))

# ============================================================================
# 6. MODEL 2: CATEGORICAL NAIVE BAYES
# ============================================================================
print("\n" + "="*80)
print("6. CATEGORICAL NAIVE BAYES (Discretized Features)")
print("="*80)

print("\nCategorical Naive Bayes works with discrete/categorical features.")
print("Uses discretized versions of continuous variables.\n")

# Train model
cnb = CategoricalNB()
cnb.fit(X_disc_train, y_train)

# Predictions
y_pred_cnb = cnb.predict(X_disc_test)
y_pred_proba_cnb = cnb.predict_proba(X_disc_test)[:, 1]

# Evaluation metrics
print("--- Categorical Naive Bayes Performance ---")
print(f"Training Accuracy: {cnb.score(X_disc_train, y_train):.4f}")
print(f"Test Accuracy: {accuracy_score(y_test, y_pred_cnb):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_cnb):.4f}")
print(f"Recall (Sensitivity): {recall_score(y_test, y_pred_cnb):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred_cnb):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba_cnb):.4f}")

print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred_cnb, 
                          target_names=['Normal (0)', 'Anormal (1)']))

# ============================================================================
# 7. CONFUSION MATRICES
# ============================================================================
print("\n" + "="*80)
print("7. CONFUSION MATRICES")
print("="*80)

cm_gnb = confusion_matrix(y_test, y_pred_gnb)
cm_cnb = confusion_matrix(y_test, y_pred_cnb)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Gaussian NB
sns.heatmap(cm_gnb, annot=True, fmt='d', cmap='Blues', ax=axes[0], 
            xticklabels=['Normal', 'Anormal'], yticklabels=['Normal', 'Anormal'])
axes[0].set_title('Gaussian Naive Bayes\nConfusion Matrix', fontsize=12, fontweight='bold')
axes[0].set_ylabel('True Label', fontsize=11)
axes[0].set_xlabel('Predicted Label', fontsize=11)

# Categorical NB
sns.heatmap(cm_cnb, annot=True, fmt='d', cmap='Greens', ax=axes[1],
            xticklabels=['Normal', 'Anormal'], yticklabels=['Normal', 'Anormal'])
axes[1].set_title('Categorical Naive Bayes\nConfusion Matrix', fontsize=12, fontweight='bold')
axes[1].set_ylabel('True Label', fontsize=11)
axes[1].set_xlabel('Predicted Label', fontsize=11)

plt.tight_layout()
plt.savefig('bn_confusion_matrices.png', dpi=300, bbox_inches='tight')
print("\n✓ Confusion matrices saved as 'bn_confusion_matrices.png'")

# Print confusion matrix details
print("\n--- Gaussian NB Confusion Matrix ---")
print(f"True Negatives (TN): {cm_gnb[0,0]}")
print(f"False Positives (FP): {cm_gnb[0,1]}")
print(f"False Negatives (FN): {cm_gnb[1,0]}")
print(f"True Positives (TP): {cm_gnb[1,1]}")
print(f"Specificity: {cm_gnb[0,0]/(cm_gnb[0,0]+cm_gnb[0,1]):.4f}")

print("\n--- Categorical NB Confusion Matrix ---")
print(f"True Negatives (TN): {cm_cnb[0,0]}")
print(f"False Positives (FP): {cm_cnb[0,1]}")
print(f"False Negatives (FN): {cm_cnb[1,0]}")
print(f"True Positives (TP): {cm_cnb[1,1]}")
print(f"Specificity: {cm_cnb[0,0]/(cm_cnb[0,0]+cm_cnb[0,1]):.4f}")

# ============================================================================
# 8. ROC CURVES
# ============================================================================
print("\n" + "="*80)
print("8. ROC CURVES")
print("="*80)

# Calculate ROC curves
fpr_gnb, tpr_gnb, _ = roc_curve(y_test, y_pred_proba_gnb)
fpr_cnb, tpr_cnb, _ = roc_curve(y_test, y_pred_proba_cnb)

auc_gnb = auc(fpr_gnb, tpr_gnb)
auc_cnb = auc(fpr_cnb, tpr_cnb)

# Plot ROC curves
plt.figure(figsize=(10, 8))
plt.plot(fpr_gnb, tpr_gnb, color='blue', lw=2, 
         label=f'Gaussian NB (AUC = {auc_gnb:.3f})')
plt.plot(fpr_cnb, tpr_cnb, color='green', lw=2, 
         label=f'Categorical NB (AUC = {auc_cnb:.3f})')
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--', label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
plt.title('ROC Curves - Bayesian Network Models', fontsize=14, fontweight='bold')
plt.legend(loc="lower right", fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('bn_roc_curves.png', dpi=300, bbox_inches='tight')
print("\n✓ ROC curves saved as 'bn_roc_curves.png'")

# ============================================================================
# 9. PROBABILITY CALIBRATION ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("9. PROBABILITY CALIBRATION ANALYSIS")
print("="*80)

print("\nNaive Bayes models output class probabilities.")
print("Analyzing probability distributions for both classes...\n")

# Create probability distribution plots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Gaussian NB - Normal class
axes[0, 0].hist(y_pred_proba_gnb[y_test == 0], bins=30, alpha=0.7, color='blue', edgecolor='black')
axes[0, 0].set_title('Gaussian NB: Predicted Probabilities\n(True Normal Cases)', fontsize=11, fontweight='bold')
axes[0, 0].set_xlabel('P(Anormal)', fontsize=10)
axes[0, 0].set_ylabel('Frequency', fontsize=10)
axes[0, 0].axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Decision Threshold')
axes[0, 0].legend()

# Gaussian NB - Anormal class
axes[0, 1].hist(y_pred_proba_gnb[y_test == 1], bins=30, alpha=0.7, color='orange', edgecolor='black')
axes[0, 1].set_title('Gaussian NB: Predicted Probabilities\n(True Anormal Cases)', fontsize=11, fontweight='bold')
axes[0, 1].set_xlabel('P(Anormal)', fontsize=10)
axes[0, 1].set_ylabel('Frequency', fontsize=10)
axes[0, 1].axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Decision Threshold')
axes[0, 1].legend()

# Categorical NB - Normal class
axes[1, 0].hist(y_pred_proba_cnb[y_test == 0], bins=30, alpha=0.7, color='green', edgecolor='black')
axes[1, 0].set_title('Categorical NB: Predicted Probabilities\n(True Normal Cases)', fontsize=11, fontweight='bold')
axes[1, 0].set_xlabel('P(Anormal)', fontsize=10)
axes[1, 0].set_ylabel('Frequency', fontsize=10)
axes[1, 0].axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Decision Threshold')
axes[1, 0].legend()

# Categorical NB - Anormal class
axes[1, 1].hist(y_pred_proba_cnb[y_test == 1], bins=30, alpha=0.7, color='purple', edgecolor='black')
axes[1, 1].set_title('Categorical NB: Predicted Probabilities\n(True Anormal Cases)', fontsize=11, fontweight='bold')
axes[1, 1].set_xlabel('P(Anormal)', fontsize=10)
axes[1, 1].set_ylabel('Frequency', fontsize=10)
axes[1, 1].axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Decision Threshold')
axes[1, 1].legend()

plt.tight_layout()
plt.savefig('bn_probability_distributions.png', dpi=300, bbox_inches='tight')
print("\n✓ Probability distributions saved as 'bn_probability_distributions.png'")

# ============================================================================
# 10. FEATURE PROBABILITY ANALYSIS (GAUSSIAN NB)
# ============================================================================
print("\n" + "="*80)
print("10. FEATURE STATISTICS PER CLASS (GAUSSIAN NB)")
print("="*80)

print("\nGaussian NB learns mean and variance for each feature per class.")
print("\n--- Class 0 (Normal) - Feature Means ---")
feature_names_cont = numeric_features + categorical_features
for i, feature in enumerate(feature_names_cont):
    print(f"{feature}: μ = {gnb.theta_[0, i]:.4f}, σ² = {gnb.var_[0, i]:.4f}")

print("\n--- Class 1 (Anormal) - Feature Means ---")
for i, feature in enumerate(feature_names_cont):
    print(f"{feature}: μ = {gnb.theta_[1, i]:.4f}, σ² = {gnb.var_[1, i]:.4f}")

# Calculate feature importance based on mean differences
print("\n--- Feature Discriminative Power (Mean Difference) ---")
mean_diffs = np.abs(gnb.theta_[1] - gnb.theta_[0])
feature_importance_gnb = pd.DataFrame({
    'Feature': feature_names_cont,
    'Mean_Diff': mean_diffs,
    'Normalized_Importance': mean_diffs / mean_diffs.sum()
}).sort_values('Mean_Diff', ascending=False)

print(feature_importance_gnb.to_string(index=False))

# Visualize
plt.figure(figsize=(10, 6))
colors = ['steelblue' if imp > 0.1 else 'lightsteelblue' 
          for imp in feature_importance_gnb['Normalized_Importance']]
plt.barh(feature_importance_gnb['Feature'], 
         feature_importance_gnb['Normalized_Importance'], color=colors)
plt.xlabel('Normalized Importance (Mean Difference)', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.title('Gaussian Naive Bayes - Feature Discriminative Power', fontsize=14, fontweight='bold')
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('bn_feature_importance.png', dpi=300, bbox_inches='tight')
print("\n✓ Feature importance plot saved as 'bn_feature_importance.png'")

# ============================================================================
# 11. CROSS-VALIDATION
# ============================================================================
print("\n" + "="*80)
print("11. CROSS-VALIDATION ANALYSIS")
print("="*80)

# Cross-validation for Gaussian NB
cv_scores_gnb = cross_val_score(gnb, X_cont_train, y_train, cv=5, scoring='f1')
print("\n--- Gaussian NB: 5-Fold Cross-Validation ---")
print(f"F1-Scores: {cv_scores_gnb}")
print(f"Mean F1-Score: {cv_scores_gnb.mean():.4f}")
print(f"Standard Deviation: {cv_scores_gnb.std():.4f}")
print(f"95% CI: [{cv_scores_gnb.mean() - 1.96*cv_scores_gnb.std():.4f}, "
      f"{cv_scores_gnb.mean() + 1.96*cv_scores_gnb.std():.4f}]")

# Cross-validation for Categorical NB
cv_scores_cnb = cross_val_score(cnb, X_disc_train, y_train, cv=5, scoring='f1')
print("\n--- Categorical NB: 5-Fold Cross-Validation ---")
print(f"F1-Scores: {cv_scores_cnb}")
print(f"Mean F1-Score: {cv_scores_cnb.mean():.4f}")
print(f"Standard Deviation: {cv_scores_cnb.std():.4f}")
print(f"95% CI: [{cv_scores_cnb.mean() - 1.96*cv_scores_cnb.std():.4f}, "
      f"{cv_scores_cnb.mean() + 1.96*cv_scores_cnb.std():.4f}]")

# ============================================================================
# 12. MODEL COMPARISON
# ============================================================================
print("\n" + "="*80)
print("12. MODEL COMPARISON SUMMARY")
print("="*80)

comparison_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 
               'Specificity', 'CV Mean F1', 'CV Std F1'],
    'Gaussian NB': [
        accuracy_score(y_test, y_pred_gnb),
        precision_score(y_test, y_pred_gnb),
        recall_score(y_test, y_pred_gnb),
        f1_score(y_test, y_pred_gnb),
        roc_auc_score(y_test, y_pred_proba_gnb),
        cm_gnb[0,0]/(cm_gnb[0,0]+cm_gnb[0,1]),
        cv_scores_gnb.mean(),
        cv_scores_gnb.std()
    ],
    'Categorical NB': [
        accuracy_score(y_test, y_pred_cnb),
        precision_score(y_test, y_pred_cnb),
        recall_score(y_test, y_pred_cnb),
        f1_score(y_test, y_pred_cnb),
        roc_auc_score(y_test, y_pred_proba_cnb),
        cm_cnb[0,0]/(cm_cnb[0,0]+cm_cnb[0,1]),
        cv_scores_cnb.mean(),
        cv_scores_cnb.std()
    ]
})

comparison_df['Difference'] = comparison_df['Categorical NB'] - comparison_df['Gaussian NB']

print("\n--- Performance Comparison ---")
print(comparison_df.to_string(index=False))

# Visualize comparison
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
gnb_values = comparison_df[comparison_df['Metric'].isin(metrics)]['Gaussian NB'].values
cnb_values = comparison_df[comparison_df['Metric'].isin(metrics)]['Categorical NB'].values

x = np.arange(len(metrics))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 6))
bars1 = ax.bar(x - width/2, gnb_values, width, label='Gaussian NB', color='steelblue', alpha=0.8)
bars2 = ax.bar(x + width/2, cnb_values, width, label='Categorical NB', color='seagreen', alpha=0.8)

ax.set_xlabel('Metric', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Bayesian Network Models - Performance Comparison', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()
ax.grid(axis='y', alpha=0.3)
ax.set_ylim([0.75, 1.0])

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('bn_model_comparison.png', dpi=300, bbox_inches='tight')
print("\n✓ Model comparison plot saved as 'bn_model_comparison.png'")

# ============================================================================
# 13. NAIVE BAYES ASSUMPTIONS ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("13. NAIVE BAYES ASSUMPTIONS")
print("="*80)

print("\n--- Key Assumption: Feature Independence ---")
print("\nNaive Bayes assumes features are conditionally independent given the class.")
print("Let's check if this assumption holds for our data:\n")

# Calculate pairwise correlations for numeric features
print("Correlation Matrix (Numeric Features):")
corr_matrix = df_model[numeric_features].corr()
print(corr_matrix.round(3))

print("\nInterpretation:")
high_corr = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        if abs(corr_matrix.iloc[i, j]) > 0.3:
            high_corr.append((corr_matrix.columns[i], 
                            corr_matrix.columns[j], 
                            corr_matrix.iloc[i, j]))

if high_corr:
    print("Features with correlation > 0.3 (violates independence assumption):")
    for f1, f2, corr in high_corr:
        print(f"  • {f1} <-> {f2}: {corr:.3f}")
else:
    print("No strong correlations detected. Independence assumption approximately holds.")

# ============================================================================
# 14. KEY INSIGHTS
# ============================================================================
print("\n" + "="*80)
print("14. KEY INSIGHTS & RECOMMENDATIONS")
print("="*80)

print("\n1. MODEL SELECTION:")
if accuracy_score(y_test, y_pred_gnb) > accuracy_score(y_test, y_pred_cnb):
    print(f"   • Gaussian NB performs better ({accuracy_score(y_test, y_pred_gnb):.4f} vs {accuracy_score(y_test, y_pred_cnb):.4f} accuracy)")
else:
    print(f"   • Categorical NB performs better ({accuracy_score(y_test, y_pred_cnb):.4f} vs {accuracy_score(y_test, y_pred_gnb):.4f} accuracy)")

print("\n2. MOST DISCRIMINATIVE FEATURES:")
top_3 = feature_importance_gnb.head(3)
for idx, row in top_3.iterrows():
    print(f"   • {row['Feature']}: {row['Normalized_Importance']:.4f}")

print("\n3. CLINICAL IMPLICATIONS:")
best_model = 'Gaussian NB' if accuracy_score(y_test, y_pred_gnb) > accuracy_score(y_test, y_pred_cnb) else 'Categorical NB'
best_recall = recall_score(y_test, y_pred_gnb) if best_model == 'Gaussian NB' else recall_score(y_test, y_pred_cnb)
best_spec = cm_gnb[0,0]/(cm_gnb[0,0]+cm_gnb[0,1]) if best_model == 'Gaussian NB' else cm_cnb[0,0]/(cm_cnb[0,0]+cm_cnb[0,1])

print(f"   • Best model ({best_model}):")
print(f"     - Sensitivity: {best_recall:.2%} (detects pathology)")
print(f"     - Specificity: {best_spec:.2%} (identifies healthy)")

print("\n4. ADVANTAGES OF BAYESIAN NETWORKS:")
print("   • Provides probabilistic predictions (uncertainty quantification)")
print("   • Fast training and prediction")
print("   • Works well with small datasets")
print("   • Handles missing data naturally")
print("   • Interpretable: based on probability theory")

print("\n5. LIMITATIONS:")
print("   • Independence assumption often violated in practice")
print("   • May underperform if features are highly correlated")
if high_corr:
    print(f"   • ⚠ Detected {len(high_corr)} correlated feature pairs in this dataset")
print("   • Gaussian NB assumes normal distributions (may not hold)")

print("\n" + "="*80)
print("BAYESIAN NETWORK ANALYSIS COMPLETE")
print("="*80)
print("\nGenerated files:")
print("  1. bn_confusion_matrices.png")
print("  2. bn_roc_curves.png")
print("  3. bn_probability_distributions.png")
print("  4. bn_feature_importance.png")
print("  5. bn_model_comparison.png")