import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix, 
                             accuracy_score, precision_score, recall_score, 
                             f1_score, roc_curve, auc, roc_auc_score)
from sklearn.utils.class_weight import compute_class_weight
import warnings
warnings.filterwarnings('ignore')

# Load the cleaned dataset
df = pd.read_excel('dataset_limpo.xlsx')

print("="*80)
print("DECISION TREE ANALYSIS - PREDICTING CARDIAC DISEASES IN CHILDREN")
print("="*80)

# ============================================================================
# 1. FEATURE SELECTION & DATA PREPARATION
# ============================================================================
print("\n" + "="*80)
print("1. FEATURE SELECTION & DATA PREPARATION")
print("="*80)

# Based on multivariate analysis, select optimal features
# Tier 1 + Tier 2 features
selected_features = ['SOPRO', 'FC', 'IMC', 'MOTIVO1', 'MOTIVO2', 'B2', 'PA_SISTOLICA']

# Prepare dataset
df_model = df[selected_features + ['NORMAL_X_ANORMAL_BIN']].copy()

# Remove rows with missing target variable
df_model = df_model[df_model['NORMAL_X_ANORMAL_BIN'].notna()]

print(f"\nOriginal dataset size: {len(df)}")
print(f"After removing missing target: {len(df_model)}")
print(f"\nClass distribution:")
print(df_model['NORMAL_X_ANORMAL_BIN'].value_counts())
print(f"\nClass balance: {df_model['NORMAL_X_ANORMAL_BIN'].value_counts(normalize=True).round(3).to_dict()}")

# Check missing values in features
print(f"\nMissing values in selected features:")
missing = df_model[selected_features].isnull().sum()
print(missing[missing > 0])

# ============================================================================
# 2. ENCODING & PREPROCESSING
# ============================================================================
print("\n" + "="*80)
print("2. ENCODING & PREPROCESSING")
print("="*80)

# Separate numeric and categorical features
numeric_features = ['FC', 'IMC', 'PA_SISTOLICA']
categorical_features = ['SOPRO', 'MOTIVO1', 'MOTIVO2', 'B2']

# Handle missing values
# For numeric: impute with median
for col in numeric_features:
    if df_model[col].isnull().any():
        median_val = df_model[col].median()
        df_model[col].fillna(median_val, inplace=True)
        print(f"Imputed {col} with median: {median_val:.2f}")

# For categorical: impute with mode or 'Missing' category
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
    print(f"\n{col} encoding:")
    for i, class_name in enumerate(le.classes_):
        print(f"  {i}: {class_name}")

# Create feature matrix
feature_names = numeric_features + [f'{col}_encoded' for col in categorical_features]
X = df_model[feature_names].values
y = df_model['NORMAL_X_ANORMAL_BIN'].values

print(f"\nFinal dataset shape: {X.shape}")
print(f"Features: {feature_names}")

# ============================================================================
# 3. TRAIN-TEST SPLIT
# ============================================================================
print("\n" + "="*80)
print("3. TRAIN-TEST SPLIT")
print("="*80)

# Split data: 70% training, 30% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"Training set size: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
print(f"Test set size: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")
print(f"\nTraining set class distribution:")
unique, counts = np.unique(y_train, return_counts=True)
for u, c in zip(unique, counts):
    print(f"  Class {int(u)}: {c} ({c/len(y_train)*100:.1f}%)")

# ============================================================================
# 4. BASELINE MODEL (DEFAULT PARAMETERS)
# ============================================================================
print("\n" + "="*80)
print("4. BASELINE DECISION TREE (DEFAULT PARAMETERS)")
print("="*80)

# Train baseline model
dt_baseline = DecisionTreeClassifier(random_state=42)
dt_baseline.fit(X_train, y_train)

# Predictions
y_pred_baseline = dt_baseline.predict(X_test)
y_pred_proba_baseline = dt_baseline.predict_proba(X_test)[:, 1]

# Evaluation metrics
print("\n--- Baseline Model Performance ---")
print(f"Training Accuracy: {dt_baseline.score(X_train, y_train):.4f}")
print(f"Test Accuracy: {accuracy_score(y_test, y_pred_baseline):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_baseline):.4f}")
print(f"Recall (Sensitivity): {recall_score(y_test, y_pred_baseline):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred_baseline):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba_baseline):.4f}")

print(f"\nTree Depth: {dt_baseline.get_depth()}")
print(f"Number of Leaves: {dt_baseline.get_n_leaves()}")

print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred_baseline, 
                          target_names=['Normal (0)', 'Anormal (1)']))

# ============================================================================
# 5. HYPERPARAMETER TUNING
# ============================================================================
print("\n" + "="*80)
print("5. HYPERPARAMETER TUNING (GRID SEARCH)")
print("="*80)

# Define parameter grid
param_grid = {
    'max_depth': [3, 5, 7, 10, 15, None],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 5, 10],
    'criterion': ['gini', 'entropy'],
    'class_weight': [None, 'balanced']
}

print("Parameter grid:")
for key, values in param_grid.items():
    print(f"  {key}: {values}")

# Grid search with cross-validation
print("\nPerforming Grid Search with 5-fold cross-validation...")
grid_search = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print("\n--- Best Parameters Found ---")
for param, value in grid_search.best_params_.items():
    print(f"  {param}: {value}")

print(f"\nBest Cross-Validation F1-Score: {grid_search.best_score_:.4f}")

# ============================================================================
# 6. OPTIMIZED MODEL
# ============================================================================
print("\n" + "="*80)
print("6. OPTIMIZED DECISION TREE")
print("="*80)

# Train optimized model
dt_optimized = grid_search.best_estimator_

# Predictions
y_pred_opt = dt_optimized.predict(X_test)
y_pred_proba_opt = dt_optimized.predict_proba(X_test)[:, 1]

# Evaluation metrics
print("\n--- Optimized Model Performance ---")
print(f"Training Accuracy: {dt_optimized.score(X_train, y_train):.4f}")
print(f"Test Accuracy: {accuracy_score(y_test, y_pred_opt):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_opt):.4f}")
print(f"Recall (Sensitivity): {recall_score(y_test, y_pred_opt):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred_opt):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba_opt):.4f}")

print(f"\nTree Depth: {dt_optimized.get_depth()}")
print(f"Number of Leaves: {dt_optimized.get_n_leaves()}")

print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred_opt, 
                          target_names=['Normal (0)', 'Anormal (1)']))

# ============================================================================
# 7. FEATURE IMPORTANCE
# ============================================================================
print("\n" + "="*80)
print("7. FEATURE IMPORTANCE ANALYSIS")
print("="*80)

# Get feature importances
importances = dt_optimized.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values('Importance', ascending=False)

# Clean feature names for display
feature_importance_df['Feature_Clean'] = feature_importance_df['Feature'].str.replace('_encoded', '')

print("\n--- Feature Importance Ranking ---")
print(feature_importance_df[['Feature_Clean', 'Importance']].to_string(index=False))

# Visualize feature importance
plt.figure(figsize=(10, 6))
colors = ['steelblue' if imp > 0.1 else 'lightsteelblue' for imp in feature_importance_df['Importance']]
plt.barh(feature_importance_df['Feature_Clean'], feature_importance_df['Importance'], color=colors)
plt.xlabel('Feature Importance', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.title('Decision Tree - Feature Importance', fontsize=14, fontweight='bold')
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('dt_feature_importance.png', dpi=300, bbox_inches='tight')
print("\n✓ Feature importance plot saved as 'dt_feature_importance.png'")

# ============================================================================
# 8. CONFUSION MATRICES
# ============================================================================
print("\n" + "="*80)
print("8. CONFUSION MATRICES")
print("="*80)

# Confusion matrices
cm_baseline = confusion_matrix(y_test, y_pred_baseline)
cm_optimized = confusion_matrix(y_test, y_pred_opt)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Baseline
sns.heatmap(cm_baseline, annot=True, fmt='d', cmap='Blues', ax=axes[0], 
            xticklabels=['Normal', 'Anormal'], yticklabels=['Normal', 'Anormal'])
axes[0].set_title('Baseline Model\nConfusion Matrix', fontsize=12, fontweight='bold')
axes[0].set_ylabel('True Label', fontsize=11)
axes[0].set_xlabel('Predicted Label', fontsize=11)

# Optimized
sns.heatmap(cm_optimized, annot=True, fmt='d', cmap='Greens', ax=axes[1],
            xticklabels=['Normal', 'Anormal'], yticklabels=['Normal', 'Anormal'])
axes[1].set_title('Optimized Model\nConfusion Matrix', fontsize=12, fontweight='bold')
axes[1].set_ylabel('True Label', fontsize=11)
axes[1].set_xlabel('Predicted Label', fontsize=11)

plt.tight_layout()
plt.savefig('dt_confusion_matrices.png', dpi=300, bbox_inches='tight')
print("\n✓ Confusion matrices saved as 'dt_confusion_matrices.png'")

# Print confusion matrix analysis
print("\n--- Baseline Model Confusion Matrix ---")
print(f"True Negatives (TN): {cm_baseline[0,0]}")
print(f"False Positives (FP): {cm_baseline[0,1]}")
print(f"False Negatives (FN): {cm_baseline[1,0]}")
print(f"True Positives (TP): {cm_baseline[1,1]}")

print("\n--- Optimized Model Confusion Matrix ---")
print(f"True Negatives (TN): {cm_optimized[0,0]}")
print(f"False Positives (FP): {cm_optimized[0,1]}")
print(f"False Negatives (FN): {cm_optimized[1,0]}")
print(f"True Positives (TP): {cm_optimized[1,1]}")

# ============================================================================
# 9. ROC CURVES
# ============================================================================
print("\n" + "="*80)
print("9. ROC CURVES")
print("="*80)

# Calculate ROC curves
fpr_baseline, tpr_baseline, _ = roc_curve(y_test, y_pred_proba_baseline)
fpr_opt, tpr_opt, _ = roc_curve(y_test, y_pred_proba_opt)

auc_baseline = auc(fpr_baseline, tpr_baseline)
auc_opt = auc(fpr_opt, tpr_opt)

# Plot ROC curves
plt.figure(figsize=(10, 8))
plt.plot(fpr_baseline, tpr_baseline, color='blue', lw=2, 
         label=f'Baseline (AUC = {auc_baseline:.3f})')
plt.plot(fpr_opt, tpr_opt, color='green', lw=2, 
         label=f'Optimized (AUC = {auc_opt:.3f})')
plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--', label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
plt.title('ROC Curves - Decision Tree Models', fontsize=14, fontweight='bold')
plt.legend(loc="lower right", fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('dt_roc_curves.png', dpi=300, bbox_inches='tight')
print("\n✓ ROC curves saved as 'dt_roc_curves.png'")

# ============================================================================
# 10. VISUALIZE DECISION TREE
# ============================================================================
print("\n" + "="*80)
print("10. DECISION TREE VISUALIZATION")
print("="*80)

# Visualize optimized tree (if not too large)
if dt_optimized.get_depth() <= 6:
    plt.figure(figsize=(20, 12))
    plot_tree(dt_optimized, 
              feature_names=[name.replace('_encoded', '') for name in feature_names],
              class_names=['Normal', 'Anormal'],
              filled=True,
              rounded=True,
              fontsize=10)
    plt.title('Optimized Decision Tree Structure', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('dt_tree_structure.png', dpi=300, bbox_inches='tight')
    print("\n✓ Full tree structure saved as 'dt_tree_structure.png'")
else:
    # For deep trees, show only top levels
    plt.figure(figsize=(20, 12))
    plot_tree(dt_optimized, 
              feature_names=[name.replace('_encoded', '') for name in feature_names],
              class_names=['Normal', 'Anormal'],
              filled=True,
              rounded=True,
              fontsize=9,
              max_depth=4)
    plt.title('Optimized Decision Tree Structure (Top 4 Levels)', 
              fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('dt_tree_structure_top_levels.png', dpi=300, bbox_inches='tight')
    print("\n✓ Tree structure (top 4 levels) saved as 'dt_tree_structure_top_levels.png'")
    print(f"  (Full tree has {dt_optimized.get_depth()} levels)")

# ============================================================================
# 11. CROSS-VALIDATION ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("11. CROSS-VALIDATION ANALYSIS")
print("="*80)

# Perform cross-validation on optimized model
cv_scores = cross_val_score(dt_optimized, X_train, y_train, cv=5, scoring='f1')

print("\n--- 5-Fold Cross-Validation Results ---")
print(f"F1-Scores: {cv_scores}")
print(f"Mean F1-Score: {cv_scores.mean():.4f}")
print(f"Standard Deviation: {cv_scores.std():.4f}")
print(f"95% Confidence Interval: [{cv_scores.mean() - 1.96*cv_scores.std():.4f}, "
      f"{cv_scores.mean() + 1.96*cv_scores.std():.4f}]")

# ============================================================================
# 12. MODEL COMPARISON SUMMARY
# ============================================================================
print("\n" + "="*80)
print("12. MODEL COMPARISON SUMMARY")
print("="*80)

comparison_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'Tree Depth', 'Num Leaves'],
    'Baseline': [
        accuracy_score(y_test, y_pred_baseline),
        precision_score(y_test, y_pred_baseline),
        recall_score(y_test, y_pred_baseline),
        f1_score(y_test, y_pred_baseline),
        roc_auc_score(y_test, y_pred_proba_baseline),
        dt_baseline.get_depth(),
        dt_baseline.get_n_leaves()
    ],
    'Optimized': [
        accuracy_score(y_test, y_pred_opt),
        precision_score(y_test, y_pred_opt),
        recall_score(y_test, y_pred_opt),
        f1_score(y_test, y_pred_opt),
        roc_auc_score(y_test, y_pred_proba_opt),
        dt_optimized.get_depth(),
        dt_optimized.get_n_leaves()
    ]
})

# Calculate improvement
comparison_df['Improvement'] = comparison_df['Optimized'] - comparison_df['Baseline']
comparison_df['Improvement %'] = (comparison_df['Improvement'] / comparison_df['Baseline'] * 100).round(2)

print("\n--- Performance Comparison ---")
print(comparison_df.to_string(index=False))

# ============================================================================
# 13. KEY INSIGHTS
# ============================================================================
print("\n" + "="*80)
print("13. KEY INSIGHTS & RECOMMENDATIONS")
print("="*80)

print("\n1. FEATURE IMPORTANCE:")
top_3_features = feature_importance_df.head(3)
for idx, row in top_3_features.iterrows():
    print(f"   • {row['Feature_Clean']}: {row['Importance']:.4f}")

print("\n2. MODEL PERFORMANCE:")
print(f"   • The optimized model achieved {accuracy_score(y_test, y_pred_opt):.2%} accuracy")
print(f"   • F1-Score: {f1_score(y_test, y_pred_opt):.4f} (balance of precision and recall)")
print(f"   • ROC-AUC: {roc_auc_score(y_test, y_pred_proba_opt):.4f}")

print("\n3. CLINICAL IMPLICATIONS:")
print(f"   • Sensitivity (Recall): {recall_score(y_test, y_pred_opt):.2%}")
print(f"     → Ability to correctly identify children WITH cardiac pathology")
print(f"   • Specificity: {cm_optimized[0,0]/(cm_optimized[0,0]+cm_optimized[0,1]):.2%}")
print(f"     → Ability to correctly identify children WITHOUT cardiac pathology")

print("\n4. MODEL COMPLEXITY:")
print(f"   • Tree depth reduced from {dt_baseline.get_depth()} to {dt_optimized.get_depth()}")
print(f"   • This prevents overfitting while maintaining performance")

print("\n" + "="*80)
print("DECISION TREE ANALYSIS COMPLETE")
print("="*80)
print("\nGenerated files:")
print("  1. dt_feature_importance.png")
print("  2. dt_confusion_matrices.png")
print("  3. dt_roc_curves.png")
print("  4. dt_tree_structure.png (or dt_tree_structure_top_levels.png)")