from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

from module4_utils import ROOT, scale, selected_features


OUTPUT = ROOT / "outputs" / "DBSCAN_Raw_dataset_M4_outputs"


SAMPLE_SIZE = 5000


def main() -> None:
    data = pd.read_csv(ROOT / "dataset" / "Plant_1_Merged_Data.csv")
    if len(data) > SAMPLE_SIZE:
        data = data.sample(SAMPLE_SIZE, random_state=42).reset_index(drop=True)
    features = selected_features(data)
    labels = DBSCAN(eps=0.8, min_samples=10, n_jobs=-1).fit_predict(scale(features))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    result = features.copy()
    result["DBSCAN_Cluster"] = labels
    result.to_csv(OUTPUT / "dbscan_clustered_data.csv", index=False)
    projection = PCA(n_components=2, random_state=42).fit_transform(scale(features))
    plt.figure(figsize=(9, 7))
    plt.scatter(projection[:, 0], projection[:, 1], c=labels, s=7, cmap="tab20")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("DBSCAN Clusters — Raw Solar Dataset")
    plt.tight_layout()
    plt.savefig(OUTPUT / "dbscan_pca_clusters.png", dpi=300)
    plt.close()
    pd.DataFrame({"Cluster": sorted(set(labels)),
                  "Count": [int((labels == label).sum()) for label in sorted(set(labels))]}
                 ).to_csv(OUTPUT / "cluster_counts.csv", index=False)
    print(f"DBSCAN raw complete: {len(set(labels))} labels")


if __name__ == "__main__":
    main()
