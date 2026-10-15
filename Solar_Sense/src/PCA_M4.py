from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from module4_utils import ROOT, load_numeric_frame, scale


OUTPUT = ROOT / "outputs" / "PCA_M4_Outputs"


def main() -> None:
    data, numeric = load_numeric_frame()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    scaled = scale(numeric)
    full = PCA().fit(scaled)
    components = PCA(n_components=2, random_state=42).fit_transform(scaled)
    pd.DataFrame(components, columns=["PC1", "PC2"]).to_csv(
        OUTPUT / "pca_2_components.csv", index=False
    )
    variance = pd.DataFrame({
        "Component": np.arange(1, len(full.explained_variance_ratio_) + 1),
        "ExplainedVarianceRatio": full.explained_variance_ratio_,
        "CumulativeExplainedVariance": np.cumsum(full.explained_variance_ratio_),
    })
    variance.to_csv(OUTPUT / "explained_variance.csv", index=False)
    plt.figure(figsize=(10, 6))
    plt.plot(variance["Component"], variance["CumulativeExplainedVariance"], marker="o")
    plt.axhline(0.95, color="red", linestyle="--", label="95% variance")
    plt.xlabel("Number of components")
    plt.ylabel("Cumulative explained variance")
    plt.title("PCA Explained Variance — Solar Dataset")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT / "explained_variance_plot.png", dpi=300)
    plt.close()
    plt.figure(figsize=(9, 7))
    plt.scatter(components[:, 0], components[:, 1], s=7, alpha=0.45)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA Projection — Solar Dataset")
    plt.tight_layout()
    plt.savefig(OUTPUT / "pca_2d_plot.png", dpi=300)
    plt.close()
    print(f"PCA complete: {data.shape[0]} rows")


if __name__ == "__main__":
    main()
