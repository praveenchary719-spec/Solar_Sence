from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

from module4_utils import ROOT, scale, selected_features


OUTPUT = ROOT / "outputs" / "Hierarchical_Clust_Agglome_Divisive_M4_Outputs"
SAMPLE_SIZE = 500
N_CLUSTERS = 3


def main() -> None:
    data = pd.read_csv(ROOT / "dataset" / "final_preprocess_M2.csv")
    features = selected_features(data).sample(min(SAMPLE_SIZE, len(data)), random_state=42)
    values = scale(features)
    agglomerative = AgglomerativeClustering(n_clusters=N_CLUSTERS, linkage="ward")
    labels = agglomerative.fit_predict(values)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"Hierarchical_Cluster": labels + 1}, index=features.index).to_csv(
        OUTPUT / "agglomerative_cluster_labels.csv", index=False
    )
    metrics = pd.DataFrame({
        "Metric": ["Silhouette", "Calinski_Harabasz", "Davies_Bouldin"],
        "Value": [silhouette_score(values, labels),
                  calinski_harabasz_score(values, labels),
                  davies_bouldin_score(values, labels)],
    })
    metrics.to_csv(OUTPUT / "hierarchical_metrics.csv", index=False)
    tree = linkage(values, method="ward")
    plt.figure(figsize=(12, 7))
    dendrogram(tree, truncate_mode="lastp", p=30)
    plt.title("Agglomerative Hierarchical Clustering Dendrogram")
    plt.tight_layout()
    plt.savefig(OUTPUT / "agglomerative_dendrogram.png", dpi=300)
    plt.close()
    print(f"Hierarchical clustering complete: {len(features)} sampled rows")


if __name__ == "__main__":
    main()
