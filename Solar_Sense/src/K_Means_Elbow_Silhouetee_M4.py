# ============================================================
# K-MEANS AND K-MEANS++
# SOLAR GENERATION PREDICTION - PREPROCESSED DATASET
# ============================================================

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    accuracy_score,
)
warnings.filterwarnings("ignore")

# ============================================================
# Paths – adjust to project structure
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "final_preprocess_M2.csv")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "outputs", "K_Means_Elbow_Silhouette_M4_Outputs")

# Create output subfolders
KMEANS_FOLDER = os.path.join(OUTPUT_FOLDER, "KMeans")
KMEANS_PP_FOLDER = os.path.join(OUTPUT_FOLDER, "KMeansPlusPlus")
ELBOW_FOLDER = os.path.join(OUTPUT_FOLDER, "Elbow")
SILHOUETTE_FOLDER = os.path.join(OUTPUT_FOLDER, "Silhouette")
ACCURACY_FOLDER = os.path.join(OUTPUT_FOLDER, "Accuracy")
for folder in [KMEANS_FOLDER, KMEANS_PP_FOLDER, ELBOW_FOLDER, SILHOUETTE_FOLDER, ACCURACY_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# ============================================================
print("=" * 75)
print("K-MEANS AND K-MEANS++")
print("SOLAR GENERATION PREDICTION - PREPROCESSED DATASET")
print("=" * 75)
if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"Preprocessed dataset not found at {DATASET_PATH}")

data = pd.read_csv(DATASET_PATH)
print("\nPreprocessed dataset loaded successfully.")
print("Rows   :", data.shape[0])
print("Columns:", data.shape[1])

# ============================================================
print("\nAvailable columns:")
print("-" * 75)
for i, column in enumerate(data.columns, 1):
    print(i, ".", column)

# ------------------------------------------------------------
def normalize_column_name(column):
    return (
        str(column)
        .strip()
        .lower()
        .replace("_", "")
        .replace(" ", "")
        .replace("-", "")
    )

normalized_columns = {normalize_column_name(col): col for col in data.columns}

# ------------------------------------------------------------
required_features = ["AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION", "DAILY_YIELD"]
feature_columns = []
for feature in required_features:
    norm_feat = normalize_column_name(feature)
    if norm_feat in normalized_columns:
        feature_columns.append(normalized_columns[norm_feat])
    else:
        if feature == "DAILY_YIELD":
            alternatives = ["Daily_Yield", "DailyYield", "DC_POWER", "DCPower", "Total_Yield", "TotalYield"]
            found = None
            for alt in alternatives:
                norm_alt = normalize_column_name(alt)
                if norm_alt in normalized_columns:
                    found = normalized_columns[norm_alt]
                    break
            if found:
                feature_columns.append(found)
            else:
                raise ValueError(f"Daily yield / power column not found. Available columns: {list(data.columns)}")
        elif feature == "AMBIENT_TEMPERATURE":
            alternatives = ["Ambient_Temperature", "AmbientTemperature", "Temp_Ambient", "AmbientTemp"]
            found = None
            for alt in alternatives:
                norm_alt = normalize_column_name(alt)
                if norm_alt in normalized_columns:
                    found = normalized_columns[norm_alt]
                    break
            if found:
                feature_columns.append(found)
            else:
                raise ValueError(f"Required column '{feature}' was not found.")
        elif feature == "MODULE_TEMPERATURE":
            alternatives = ["Module_Temperature", "ModuleTemperature", "Temp_Module", "ModuleTemp"]
            found = None
            for alt in alternatives:
                norm_alt = normalize_column_name(alt)
                if norm_alt in normalized_columns:
                    found = normalized_columns[norm_alt]
                    break
            if found:
                feature_columns.append(found)
            else:
                raise ValueError(f"Required column '{feature}' was not found.")
        elif feature == "IRRADIATION":
            alternatives = ["Irradiation", "Solar_Irradiation", "SolarIrradiation", "Radiation"]
            found = None
            for alt in alternatives:
                norm_alt = normalize_column_name(alt)
                if norm_alt in normalized_columns:
                    found = normalized_columns[norm_alt]
                    break
            if found:
                feature_columns.append(found)
            else:
                raise ValueError(f"Required column '{feature}' was not found.")
        else:
            raise ValueError(f"Required column '{feature}' was not found.")

print("\n" + "=" * 75)
print("FEATURES USED FOR CLUSTERING")
print("=" * 75)
for f in feature_columns:
    print("-", f)

# ============================================================
X = data[feature_columns].copy()
# Ensure numeric
for col in feature_columns:
    X[col] = pd.to_numeric(X[col], errors="coerce")
# Replace infinities
X = X.replace([np.inf, -np.inf], np.nan)
# Impute missing
print("\nMissing values:")
print(X.isnull().sum())
for col in feature_columns:
    X[col] = X[col].fillna(X[col].median())
print("\nMissing values after preprocessing:")
print(X.isnull().sum())

# Standardise
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("\nFeature standardisation completed.")

# ------------------------------------------------------------
K_VALUES = range(2, 11)
print("\n" + "=" * 75)
print("K-MEANS ANALYSIS")
print("=" * 75)

sample_n = min(10000, len(X_scaled))

kmeans_inertia, kmeans_silhouette = [], []
for k in K_VALUES:
    model = KMeans(n_clusters=k, init="random", n_init=10, random_state=42)
    labels = model.fit_predict(X_scaled)
    kmeans_inertia.append(model.inertia_)
    sil = silhouette_score(X_scaled, labels, sample_size=sample_n, random_state=42)
    kmeans_silhouette.append(sil)
    print(f"K = {k:2d} | Inertia = {model.inertia_:.4f} | Silhouette = {kmeans_silhouette[-1]:.4f}")

print("\n" + "=" * 75)
print("K-MEANS++ ANALYSIS")
print("=" * 75)

kmeans_pp_inertia, kmeans_pp_silhouette = [], []
for k in K_VALUES:
    model = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
    labels = model.fit_predict(X_scaled)
    kmeans_pp_inertia.append(model.inertia_)
    sil = silhouette_score(X_scaled, labels, sample_size=sample_n, random_state=42)
    kmeans_pp_silhouette.append(sil)
    print(f"K = {k:2d} | Inertia = {model.inertia_:.4f} | Silhouette = {kmeans_pp_silhouette[-1]:.4f}")

best_k_kmeans = K_VALUES[np.argmax(kmeans_silhouette)]
best_k_kmeans_pp = K_VALUES[np.argmax(kmeans_pp_silhouette)]
print("\n" + "=" * 75)
print("SELECTED NUMBER OF CLUSTERS")
print("=" * 75)
print("K-Means best K       :", best_k_kmeans)
print("K-Means++ best K    :", best_k_kmeans_pp)

# ------------------------------------------------------------
# Plot Elbow
plt.figure(figsize=(10, 6))
plt.plot(list(K_VALUES), kmeans_inertia, marker="o")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method - K-Means")
plt.xticks(list(K_VALUES))
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(ELBOW_FOLDER, "elbow_kmeans.png"), dpi=300, bbox_inches="tight")
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(list(K_VALUES), kmeans_pp_inertia, marker="o")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method - K-Means++")
plt.xticks(list(K_VALUES))
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(ELBOW_FOLDER, "elbow_kmeans_plus_plus.png"), dpi=300, bbox_inches="tight")
plt.close()

# ------------------------------------------------------------
# Plot Silhouette
plt.figure(figsize=(10, 6))
plt.plot(list(K_VALUES), kmeans_silhouette, marker="o")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score - K-Means")
plt.xticks(list(K_VALUES))
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(SILHOUETTE_FOLDER, "silhouette_kmeans.png"), dpi=300, bbox_inches="tight")
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(list(K_VALUES), kmeans_pp_silhouette, marker="o")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score - K-Means++")
plt.xticks(list(K_VALUES))
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(SILHOUETTE_FOLDER, "silhouette_kmeans_plus_plus.png"), dpi=300, bbox_inches="tight")
plt.close()

# ------------------------------------------------------------
# Final clustering
kmeans_model = KMeans(n_clusters=best_k_kmeans, init="random", n_init=10, random_state=42)
kmeans_labels = kmeans_model.fit_predict(X_scaled)

kmeans_pp_model = KMeans(n_clusters=best_k_kmeans_pp, init="k-means++", n_init=10, random_state=42)
kmeans_pp_labels = kmeans_pp_model.fit_predict(X_scaled)

# Save clustered data
kmeans_result = X.copy()
kmeans_result["KMeans_Cluster"] = kmeans_labels + 1
kmeans_result.to_csv(os.path.join(KMEANS_FOLDER, "kmeans_clustered_data.csv"), index=False)

kmeans_pp_result = X.copy()
kmeans_pp_result["KMeansPlusPlus_Cluster"] = kmeans_pp_labels + 1
kmeans_pp_result.to_csv(os.path.join(KMEANS_PP_FOLDER, "kmeans_plus_plus_clustered_data.csv"), index=False)

# Metrics
km_silhouette = silhouette_score(X_scaled, kmeans_labels, sample_size=sample_n, random_state=42)
km_calinski = calinski_harabasz_score(X_scaled, kmeans_labels)
km_davies = davies_bouldin_score(X_scaled, kmeans_labels)

pp_silhouette = silhouette_score(X_scaled, kmeans_pp_labels, sample_size=sample_n, random_state=42)
pp_calinski = calinski_harabasz_score(X_scaled, kmeans_pp_labels)
pp_davies = davies_bouldin_score(X_scaled, kmeans_pp_labels)

kmeans_metrics = pd.DataFrame({
    "Method": ["K-Means"],
    "Number_of_Clusters": [best_k_kmeans],
    "Inertia": [kmeans_model.inertia_],
    "Silhouette_Score": [km_silhouette],
    "Calinski_Harabasz_Score": [km_calinski],
    "Davies_Bouldin_Score": [km_davies],
})
kmeans_metrics.to_csv(os.path.join(KMEANS_FOLDER, "kmeans_metrics.csv"), index=False)

kmeans_pp_metrics = pd.DataFrame({
    "Method": ["K-Means++"],
    "Number_of_Clusters": [best_k_kmeans_pp],
    "Inertia": [kmeans_pp_model.inertia_],
    "Silhouette_Score": [pp_silhouette],
    "Calinski_Harabasz_Score": [pp_calinski],
    "Davies_Bouldin_Score": [pp_davies],
})
kmeans_pp_metrics.to_csv(os.path.join(KMEANS_PP_FOLDER, "kmeans_plus_plus_metrics.csv"), index=False)

# 3D visualisations
fig = plt.figure(figsize=(11, 8))
ax = fig.add_subplot(111, projection="3d")
scatter = ax.scatter(X[feature_columns[0]], X[feature_columns[1]], X[feature_columns[2]], c=kmeans_labels, s=20, alpha=0.7)
ax.set_xlabel(feature_columns[0])
ax.set_ylabel(feature_columns[1])
ax.set_zlabel(feature_columns[2])
ax.set_title("K-Means Clustering")
fig.colorbar(scatter, ax=ax, label="Cluster")
plt.tight_layout()
plt.savefig(os.path.join(KMEANS_FOLDER, "kmeans_3D_clustering.png"), dpi=300, bbox_inches="tight")
plt.close()

fig = plt.figure(figsize=(11, 8))
ax = fig.add_subplot(111, projection="3d")
scatter = ax.scatter(X[feature_columns[0]], X[feature_columns[1]], X[feature_columns[2]], c=kmeans_pp_labels, s=20, alpha=0.7)
ax.set_xlabel(feature_columns[0])
ax.set_ylabel(feature_columns[1])
ax.set_zlabel(feature_columns[2])
ax.set_title("K-Means++ Clustering")
fig.colorbar(scatter, ax=ax, label="Cluster")
plt.tight_layout()
plt.savefig(os.path.join(KMEANS_PP_FOLDER, "kmeans_plus_plus_3D_clustering.png"), dpi=300, bbox_inches="tight")
plt.close()

# PCA visualisations
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
plt.figure(figsize=(10, 7))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_labels, s=20, alpha=0.7)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("K-Means Clusters - PCA Visualization")
plt.colorbar(scatter, label="Cluster")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(KMEANS_FOLDER, "kmeans_PCA_clustering.png"), dpi=300, bbox_inches="tight")
plt.close()

plt.figure(figsize=(10, 7))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_pp_labels, s=20, alpha=0.7)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("K-Means++ Clusters - PCA Visualization")
plt.colorbar(scatter, label="Cluster")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(KMEANS_PP_FOLDER, "kmeans_plus_plus_PCA_clustering.png"), dpi=300, bbox_inches="tight")
plt.close()

# Cluster counts
print("\n" + "=" * 75)
print("K-MEANS CLUSTER COUNTS")
print("=" * 75)
print(kmeans_result["KMeans_Cluster"].value_counts().sort_index())
print("\n" + "=" * 75)
print("K-MEANS++ CLUSTER COUNTS")
print("=" * 75)
print(kmeans_pp_result["KMeansPlusPlus_Cluster"].value_counts().sort_index())

# ------------------------------------------------------------
# Optional: if a target column exists, calculate cluster‑matching accuracy
target_candidates = ["Is_Daytime", "DC_POWER_Tier", "Generation_Status", "Status", "Target"]
target_column = None
for cand in target_candidates:
    norm = normalize_column_name(cand)
    if norm in normalized_columns:
        target_column = normalized_columns[norm]
        break

# If not directly in preprocessed CSV, check if daytime can be inferred for evaluation
if target_column is None and "IRRADIATION" in data.columns:
    data["Is_Daytime"] = (data["IRRADIATION"] > data["IRRADIATION"].min()).astype(int)
    target_column = "Is_Daytime"

def cluster_matching_accuracy(true_vals, cluster_vals):
    true_vals = pd.Series(true_vals).reset_index(drop=True)
    cluster_vals = pd.Series(cluster_vals).reset_index(drop=True)
    pred = np.empty(len(cluster_vals), dtype=true_vals.dtype)
    for cl in np.unique(cluster_vals):
        idx = np.where(cluster_vals == cl)[0]
        majority = true_vals.iloc[idx].mode()
        if len(majority) > 0:
            pred[idx] = majority.iloc[0]
    return accuracy_score(true_vals, pred)

if target_column is not None:
    print("\n" + "=" * 75)
    print("SOLAR CLUSTER MATCHING ACCURACY")
    print("=" * 75)
    target = data[target_column].copy()
    valid = target.notna()
    km_acc = cluster_matching_accuracy(target[valid], kmeans_labels[valid])
    pp_acc = cluster_matching_accuracy(target[valid], kmeans_pp_labels[valid])
    print("Target column:", target_column)
    print(f"K-Means   : {km_acc * 100:.2f}%")
    print(f"K-Means++ : {pp_acc * 100:.2f}%")
    pd.DataFrame({
        "Method": ["K-Means", "K-Means++"],
        "Cluster_Matching_Accuracy": [km_acc, pp_acc],
        "Accuracy_Percentage": [km_acc * 100, pp_acc * 100],
    }).to_csv(os.path.join(ACCURACY_FOLDER, "solar_cluster_matching_accuracy.csv"), index=False)
else:
    print("\nNo Target column was found.")
    print("Accuracy is not calculated because K-Means is an unsupervised algorithm.")

# ------------------------------------------------------------
print("\n" + "=" * 75)
print("FINAL CLUSTERING RESULTS")
print("=" * 75)
print("\nK-MEANS")
print("-" * 40)
print("Best K:", best_k_kmeans)
print("Inertia:", round(kmeans_model.inertia_, 4))
print("Silhouette:", round(km_silhouette, 4))
print("Calinski-Harabasz:", round(km_calinski, 4))
print("Davies-Bouldin:", round(km_davies, 4))
print("\nK-MEANS++")
print("-" * 40)
print("Best K:", best_k_kmeans_pp)
print("Inertia:", round(kmeans_pp_model.inertia_, 4))
print("Silhouette:", round(pp_silhouette, 4))
print("Calinski-Harabasz:", round(pp_calinski, 4))
print("Davies-Bouldin:", round(pp_davies, 4))

print("\n" + "=" * 75)
print("OUTPUTS SAVED")
print("=" * 75)
print("\nMain folder:")
print(OUTPUT_FOLDER)
print("\nK-Means:")
print(KMEANS_FOLDER)
print("\nK-Means++:")
print(KMEANS_PP_FOLDER)
print("\nElbow:")
print(ELBOW_FOLDER)
print("\nSilhouette:")
print(SILHOUETTE_FOLDER)
print("\nAccuracy:")
print(ACCURACY_FOLDER)
print("\nProgram completed successfully.")

# ============================================================
