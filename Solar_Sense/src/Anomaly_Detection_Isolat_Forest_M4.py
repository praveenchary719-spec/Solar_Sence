from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import IsolationForest

from module4_utils import ROOT, load_numeric_frame, scale


OUTPUT = ROOT / "outputs" / "Anomoly_Detec_Isolation_Forest_M4_outputs"


def main() -> None:
    data, numeric = load_numeric_frame()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    model = IsolationForest(
        n_estimators=200, contamination="auto", random_state=42, n_jobs=-1
    )
    labels = model.fit_predict(scale(numeric))
    result = data.copy()
    result["AnomalyScore"] = model.decision_function(scale(numeric))
    result["AnomalyLabel"] = labels
    result["IsAnomaly"] = labels == -1
    result.to_csv(OUTPUT / "isolation_forest_results.csv", index=False)
    pd.DataFrame(
        {"Metric": ["Rows", "Anomalies", "Anomaly_Percentage"],
         "Value": [len(result), int((labels == -1).sum()),
                   float((labels == -1).mean() * 100)]}
    ).to_csv(OUTPUT / "anomaly_summary.csv", index=False)
    plt.figure(figsize=(9, 5))
    plt.hist(result["AnomalyScore"], bins=50, color="steelblue")
    plt.xlabel("Isolation Forest decision score")
    plt.ylabel("Observations")
    plt.title("Isolation Forest Anomaly Scores")
    plt.tight_layout()
    plt.savefig(OUTPUT / "anomaly_score_distribution.png", dpi=300)
    plt.close()
    print(f"Isolation Forest complete: {int((labels == -1).sum())} anomalies")


if __name__ == "__main__":
    main()
