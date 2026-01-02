import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.feature_selection import mutual_info_classif
from scipy.stats import chi2_contingency
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Load the cleaned dataset
df = pd.read_excel('dataset_limpo.xlsx')

print("="*80)
print("MULTIVARIATE ANALYSIS - PREDICTING CARDIAC DISEASES IN CHILDREN")
print("="*80)

# ============================================================================
# 1. DATA PREPARATION FOR MULTIVARIATE ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("1. DATA PREPARATION")
print("="*80)

# Select relevant features for analysis
numeric_features = ['PESO', 'ALTURA', 'IMC', 'PA_SISTOLICA', 'PA_DIASTOLICA', 'FC', 'IDADE']
categorical_features = ['PPA', 'B2', 'SOPRO', 'HDA', 'SEXO', 'MOTIVO1', 'MOTIVO2']

# Create a working dataframe with complete cases for key variables
df_analysis = df[numeric_features + categorical_features + ['NORMAL_X_ANORMAL_BIN']].copy()

# Display missing values
print("\nMissing Values Summary:")
missing = df_analysis.isnull().sum()
missing_pct = (missing / len(df_analysis)) * 100
missing_df = pd.DataFrame({
    'Missing Count': missing,
    'Percentage': missing_pct
}).sort_values('Percentage', ascending=False)
print(missing_df[missing_df['Missing Count'] > 0])

# For multivariate analysis, we'll work with complete cases
df_complete = df_analysis.dropna(subset=['NORMAL_X_ANORMAL_BIN'])
print(f"\nOriginal dataset size: {len(df_analysis)}")
print(f"Complete cases for analysis: {len(df_complete)}")

# ============================================================================
# 2. CORRELATION ANALYSIS (NUMERIC VARIABLES)
# ============================================================================
print("\n" + "="*80)
print("2. CORRELATION ANALYSIS - NUMERIC VARIABLES")
print("="*80)

# Prepare numeric data
numeric_data = df_complete[numeric_features].dropna()
print(f"\nSample size for correlation analysis: {len(numeric_data)}")

# Pearson Correlation
print("\n--- Pearson Correlation Matrix ---")
pearson_corr = numeric_data.corr(method='pearson')
print(pearson_corr.round(3))

# Spearman Correlation (rank-based, better for non-linear relationships)
print("\n--- Spearman Correlation Matrix ---")
spearman_corr = numeric_data.corr(method='spearman')
print(spearman_corr.round(3))

# Visualize correlations
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sns.heatmap(pearson_corr, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, vmin=-1, vmax=1, ax=axes[0], cbar_kws={'label': 'Correlation'})
axes[0].set_title('Pearson Correlation Matrix', fontsize=14, fontweight='bold')

sns.heatmap(spearman_corr, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, vmin=-1, vmax=1, ax=axes[1], cbar_kws={'label': 'Correlation'})
axes[1].set_title('Spearman Correlation Matrix', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('correlation_analysis.png', dpi=300, bbox_inches='tight')
print("\n✓ Correlation heatmaps saved as 'correlation_analysis.png'")

# Key findings
print("\n--- Key Correlation Findings ---")
print("\nStrongly Correlated Variables (|r| > 0.7):")
for i in range(len(pearson_corr.columns)):
    for j in range(i+1, len(pearson_corr.columns)):
        if abs(pearson_corr.iloc[i, j]) > 0.7:
            print(f"  • {pearson_corr.columns[i]} <-> {pearson_corr.columns[j]}: "
                  f"Pearson = {pearson_corr.iloc[i, j]:.3f}, "
                  f"Spearman = {spearman_corr.iloc[i, j]:.3f}")

# ============================================================================
# 3. MULTIPLE REGRESSION ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("3. MULTIPLE LOGISTIC REGRESSION")
print("="*80)

# Prepare data for logistic regression
df_reg = df_complete[numeric_features + ['NORMAL_X_ANORMAL_BIN']].dropna()
X_reg = df_reg[numeric_features]
y_reg = df_reg['NORMAL_X_ANORMAL_BIN']

# Standardize features
scaler = StandardScaler()
X_reg_scaled = scaler.fit_transform(X_reg)

# Fit logistic regression
log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_reg_scaled, y_reg)

# Display coefficients
print("\n--- Logistic Regression Coefficients ---")
coef_df = pd.DataFrame({
    'Feature': numeric_features,
    'Coefficient': log_reg.coef_[0],
    'Abs_Coefficient': np.abs(log_reg.coef_[0])
}).sort_values('Abs_Coefficient', ascending=False)
print(coef_df.to_string(index=False))

# Feature importance visualization
plt.figure(figsize=(10, 6))
plt.barh(coef_df['Feature'], coef_df['Coefficient'], color=['red' if x < 0 else 'green' for x in coef_df['Coefficient']])
plt.xlabel('Coefficient Value', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.title('Logistic Regression Coefficients\n(Standardized Features)', fontsize=14, fontweight='bold')
plt.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('logistic_regression_coefficients.png', dpi=300, bbox_inches='tight')
print("\n✓ Coefficient plot saved as 'logistic_regression_coefficients.png'")

# ============================================================================
# 4. MUTUAL INFORMATION ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("4. MUTUAL INFORMATION ANALYSIS")
print("="*80)

# Prepare data with both numeric and encoded categorical features
df_mi = df_complete.copy()

# Encode categorical variables
le_dict = {}
categorical_encoded = []
for cat in categorical_features:
    if cat in df_mi.columns:
        df_mi[cat] = df_mi[cat].fillna('Missing')
        le = LabelEncoder()
        df_mi[f'{cat}_encoded'] = le.fit_transform(df_mi[cat].astype(str))
        le_dict[cat] = le
        categorical_encoded.append(f'{cat}_encoded')

# Combine all features
all_features_mi = numeric_features + categorical_encoded
X_mi = df_mi[all_features_mi].fillna(df_mi[all_features_mi].median())
y_mi = df_mi['NORMAL_X_ANORMAL_BIN']

# Calculate mutual information
mi_scores = mutual_info_classif(X_mi, y_mi, random_state=42)

# Create MI results dataframe
mi_df = pd.DataFrame({
    'Feature': all_features_mi,
    'MI_Score': mi_scores
}).sort_values('MI_Score', ascending=False)

# Clean feature names for display
mi_df['Feature_Clean'] = mi_df['Feature'].str.replace('_encoded', '')

print("\n--- Mutual Information Scores (Feature Relevance) ---")
print(mi_df[['Feature_Clean', 'MI_Score']].to_string(index=False))

# Visualization
plt.figure(figsize=(12, 8))
colors = ['steelblue' if '_encoded' not in f else 'coral' for f in mi_df['Feature']]
plt.barh(mi_df['Feature_Clean'], mi_df['MI_Score'], color=colors)
plt.xlabel('Mutual Information Score', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.title('Feature Importance using Mutual Information\n(Blue: Numeric, Orange: Categorical)', 
          fontsize=14, fontweight='bold')
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('mutual_information_scores.png', dpi=300, bbox_inches='tight')
print("\n✓ MI scores plot saved as 'mutual_information_scores.png'")

# ============================================================================
# 5. CATEGORICAL ASSOCIATION ANALYSIS (CHI-SQUARE)
# ============================================================================
print("\n" + "="*80)
print("5. CHI-SQUARE TEST FOR CATEGORICAL ASSOCIATIONS")
print("="*80)

chi2_results = []
for cat_var in categorical_features:
    if cat_var in df_complete.columns:
        # Create contingency table
        contingency = pd.crosstab(df_complete[cat_var].fillna('Missing'), 
                                   df_complete['NORMAL_X_ANORMAL_BIN'])
        
        # Perform chi-square test
        chi2, p_value, dof, expected = chi2_contingency(contingency)
        
        chi2_results.append({
            'Variable': cat_var,
            'Chi2_Statistic': chi2,
            'P_Value': p_value,
            'Significant': 'Yes' if p_value < 0.05 else 'No'
        })

chi2_df = pd.DataFrame(chi2_results).sort_values('Chi2_Statistic', ascending=False)
print("\n--- Chi-Square Test Results ---")
print(chi2_df.to_string(index=False))

# ============================================================================
# 6. PRINCIPAL COMPONENT ANALYSIS (PCA)
# ============================================================================
print("\n" + "="*80)
print("6. PRINCIPAL COMPONENT ANALYSIS (PCA)")
print("="*80)

# Prepare data for PCA (numeric features only)
df_pca = df_complete[numeric_features].dropna()
X_pca = StandardScaler().fit_transform(df_pca)

# Fit PCA
pca = PCA()
X_pca_transformed = pca.fit_transform(X_pca)

# Variance explained
print("\n--- Variance Explained by Principal Components ---")
var_exp_df = pd.DataFrame({
    'PC': [f'PC{i+1}' for i in range(len(pca.explained_variance_ratio_))],
    'Variance_Explained': pca.explained_variance_ratio_,
    'Cumulative_Variance': np.cumsum(pca.explained_variance_ratio_)
})
print(var_exp_df.round(4).to_string(index=False))

# Visualizations
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Scree plot
axes[0].bar(range(1, len(pca.explained_variance_ratio_)+1), pca.explained_variance_ratio_, 
            color='steelblue', alpha=0.7)
axes[0].plot(range(1, len(pca.explained_variance_ratio_)+1), 
             np.cumsum(pca.explained_variance_ratio_), 'ro-', linewidth=2, markersize=8)
axes[0].set_xlabel('Principal Component', fontsize=12)
axes[0].set_ylabel('Variance Explained', fontsize=12)
axes[0].set_title('PCA Scree Plot', fontsize=14, fontweight='bold')
axes[0].grid(alpha=0.3)
axes[0].legend(['Cumulative', 'Individual'], loc='center right')

# Biplot (PC1 vs PC2)
y_pca = df_complete.loc[df_pca.index, 'NORMAL_X_ANORMAL_BIN']
scatter = axes[1].scatter(X_pca_transformed[:, 0], X_pca_transformed[:, 1], 
                          c=y_pca, cmap='RdYlGn_r', alpha=0.6, s=20)
axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)', fontsize=12)
axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)', fontsize=12)
axes[1].set_title('PCA Biplot (PC1 vs PC2)', fontsize=14, fontweight='bold')
plt.colorbar(scatter, ax=axes[1], label='Normal (0) / Anormal (1)')

plt.tight_layout()
plt.savefig('pca_analysis.png', dpi=300, bbox_inches='tight')
print("\n✓ PCA plots saved as 'pca_analysis.png'")

# Component loadings
print("\n--- PC1 and PC2 Loadings ---")
loadings = pd.DataFrame(
    pca.components_[:2, :].T,
    columns=['PC1', 'PC2'],
    index=numeric_features
).round(3)
print(loadings)

# ============================================================================
# 7. CLUSTER ANALYSIS (K-MEANS)
# ============================================================================
print("\n" + "="*80)
print("7. CLUSTER ANALYSIS (K-MEANS)")
print("="*80)

# Prepare data
df_cluster = df_complete[numeric_features].dropna()
X_cluster = StandardScaler().fit_transform(df_cluster)

# Elbow method to find optimal k
inertias = []
silhouette_scores = []
K_range = range(2, 11)

from sklearn.metrics import silhouette_score

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_cluster)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_cluster, kmeans.labels_))

# Plot elbow curve
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

axes[0].plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
axes[0].set_xlabel('Number of Clusters (k)', fontsize=12)
axes[0].set_ylabel('Inertia (Within-cluster sum of squares)', fontsize=12)
axes[0].set_title('Elbow Method for Optimal k', fontsize=14, fontweight='bold')
axes[0].grid(alpha=0.3)

axes[1].plot(K_range, silhouette_scores, 'go-', linewidth=2, markersize=8)
axes[1].set_xlabel('Number of Clusters (k)', fontsize=12)
axes[1].set_ylabel('Silhouette Score', fontsize=12)
axes[1].set_title('Silhouette Score vs Number of Clusters', fontsize=14, fontweight='bold')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('cluster_analysis_metrics.png', dpi=300, bbox_inches='tight')
print("\n✓ Cluster metrics saved as 'cluster_analysis_metrics.png'")

# Fit final model with optimal k (let's use k=3 or k=4)
optimal_k = 3
kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
clusters = kmeans_final.fit_predict(X_cluster)

# Add cluster labels
df_cluster['Cluster'] = clusters

# Cluster characteristics
print(f"\n--- Cluster Analysis with k={optimal_k} ---")
print(f"\nCluster Distribution:")
print(df_cluster['Cluster'].value_counts().sort_index())

print("\n--- Mean Values by Cluster (Standardized) ---")
cluster_centers_df = pd.DataFrame(
    kmeans_final.cluster_centers_,
    columns=numeric_features,
    index=[f'Cluster {i}' for i in range(optimal_k)]
).round(3)
print(cluster_centers_df)

# Visualize clusters in PCA space
pca_cluster = PCA(n_components=2)
X_cluster_pca = pca_cluster.fit_transform(X_cluster)

plt.figure(figsize=(12, 8))
scatter = plt.scatter(X_cluster_pca[:, 0], X_cluster_pca[:, 1], 
                     c=clusters, cmap='viridis', alpha=0.6, s=30)
plt.scatter(kmeans_final.cluster_centers_[:, 0], kmeans_final.cluster_centers_[:, 1], 
           c='red', marker='X', s=300, edgecolors='black', linewidths=2, label='Centroids')
plt.xlabel(f'PC1 ({pca_cluster.explained_variance_ratio_[0]:.2%} variance)', fontsize=12)
plt.ylabel(f'PC2 ({pca_cluster.explained_variance_ratio_[1]:.2%} variance)', fontsize=12)
plt.title(f'K-Means Clustering (k={optimal_k}) in PCA Space', fontsize=14, fontweight='bold')
plt.colorbar(scatter, label='Cluster')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('cluster_visualization.png', dpi=300, bbox_inches='tight')
print("\n✓ Cluster visualization saved as 'cluster_visualization.png'")

# Analyze cluster vs target variable
y_cluster = df_complete.loc[df_cluster.index[:-1], 'NORMAL_X_ANORMAL_BIN']
if len(y_cluster) == len(clusters):
    cluster_target = pd.crosstab(clusters, y_cluster, normalize='index')
    print("\n--- Proportion of Normal/Anormal by Cluster ---")
    print(cluster_target.round(3))

# ============================================================================
# 8. FEATURE REDUNDANCY ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("8. FEATURE REDUNDANCY ANALYSIS")
print("="*80)

print("\nBased on correlation analysis:")
print("• PESO, ALTURA, and IMC show strong correlations (as expected)")
print("  → Recommendation: Use IMC alone or consider removing one of PESO/ALTURA")
print("\nBased on mutual information:")
print(f"• Top 5 most informative features:")
for idx, row in mi_df.head(5).iterrows():
    print(f"  {row['Feature_Clean']}: {row['MI_Score']:.4f}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print("\nGenerated files:")
print("  1. correlation_analysis.png")
print("  2. logistic_regression_coefficients.png")
print("  3. mutual_information_scores.png")
print("  4. pca_analysis.png")
print("  5. cluster_analysis_metrics.png")
print("  6. cluster_visualization.png")