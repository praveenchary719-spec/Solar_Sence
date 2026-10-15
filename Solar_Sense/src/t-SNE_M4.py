from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE

from module4_utils import ROOT, load_numeric_frame, scale


OUTPUT = ROOT / "outputs" / "t-SNE_M4_outputs"
SAMPLE_SIZE = 3000


def main() -> None:
    data, numeric = load_numeric_frame()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    sample_idx = numeric.sample(min(SAMPLE_SIZE, len(numeric)), random_state=42).index
    sample = numeric.loc[sample_idx]
    sample_raw = data.loc[sample_idx]

    embedding = TSNE(
        n_components=2, perplexity=30, init="pca", learning_rate="auto",
        random_state=42, max_iter=1000
    ).fit_transform(scale(sample))

    result = pd.DataFrame({"TSNE_1": embedding[:, 0], "TSNE_2": embedding[:, 1]},
                          index=sample.index)
    result.to_csv(OUTPUT / "tsne_coordinates.csv", index=False)

    c_vals = sample_raw["DC_POWER"] if "DC_POWER" in sample_raw.columns else result["TSNE_1"]

    plt.figure(figsize=(11, 7.5), facecolor="#ffffff")
    scatter = plt.scatter(
        result["TSNE_1"], result["TSNE_2"],
        c=c_vals, cmap="plasma", s=16, alpha=0.75, edgecolors="none"
    )
    cbar = plt.colorbar(scatter, pad=0.02)
    cbar.set_label("DC Power Output (kW)", fontsize=11, fontweight="bold", labelpad=10)

    plt.title("t-SNE Projection — Solar Dataset", fontsize=15, fontweight="bold", pad=14)
    plt.xlabel("t-SNE Dimension 1", fontsize=11, fontweight="bold", labelpad=8)
    plt.ylabel("t-SNE Dimension 2", fontsize=11, fontweight="bold", labelpad=8)
    plt.grid(True, linestyle="--", alpha=0.3, color="#94a3b8")

    plt.tight_layout()
    plt.savefig(OUTPUT / "tsne_plot.png", dpi=200, bbox_inches="tight", facecolor="#ffffff")
    plt.close()
    print(f"t-SNE complete: {len(result)} sampled rows, saved to {OUTPUT / 'tsne_plot.png'}")


if __name__ == "__main__":
    main()
