"""Decision-tree classification on the preprocessed solar dataset.

The source CSV is never modified.  All predictions and reports are written to
``outputs/CV_DT_All_Classifi_Regres_Metrics_M5_outputs``.
"""

from pathlib import Path
import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = PROJECT_ROOT / "dataset" / "final_preprocess_M2.csv"
OUTPUT_FOLDER = (
    PROJECT_ROOT / "outputs" / "CV_DT_All_Classifi_Regres_Metrics_M5_outputs"
)
RANDOM_STATE = 42
REQUESTED_SPLITS = 5


def detect_target(columns: pd.Index) -> str:
    candidates = [
        "DC_POWER",
        "AC_POWER",
        "DAILY_YIELD",
        "Placement",
        "Placed",
        "placement",
        "placed",
        "Status",
        "status",
        "Target",
        "target",
    ]
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return str(columns[-1])


def save_curve(path: Path, x: np.ndarray, y: np.ndarray, label: str,
               xlabel: str, ylabel: str, title: str) -> None:
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def main() -> None:
    print("=" * 70)
    print("DECISION TREE CLASSIFIER — SOLAR GENERATION PREDICTION")
    print("=" * 70)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {INPUT_FILE}. Please check the input path."
        )

    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(INPUT_FILE)
    working_data = data.drop_duplicates().reset_index(drop=True)
    duplicate_count = len(data) - len(working_data)
    target_column = detect_target(working_data.columns)

    if target_column not in working_data:
        raise ValueError(f"Target column does not exist: {target_column}")

    # For solar data, bin DC_POWER into classes if it's numeric
    if target_column == "DC_POWER" and working_data[target_column].dtype in [np.float64, np.int64]:
        bins = [-np.inf, 0.01, 0.3, 0.7, np.inf]
        labels_str = ["Zero", "Low", "Medium", "High"]
        working_data[target_column] = pd.cut(
            working_data[target_column], bins=bins, labels=labels_str
        ).astype(str)

    working_data = working_data.dropna(subset=[target_column]).copy()
    X = working_data.drop(columns=[target_column]).copy()
    y = working_data[target_column].copy()

    categorical_columns = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()
    if categorical_columns:
        X = pd.get_dummies(X, columns=categorical_columns, drop_first=False)
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True)).fillna(0)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y.astype(str))
    number_of_classes = len(label_encoder.classes_)
    if number_of_classes < 2:
        raise ValueError("The target column contains fewer than two classes.")

    class_counts = np.bincount(y_encoded)
    n_splits = min(REQUESTED_SPLITS, int(class_counts.min()))
    if n_splits < 2:
        raise ValueError("Each target class needs at least two rows for CV.")

    dataset_info = pd.DataFrame(
        {
            "Parameter": [
                "Original Rows",
                "Original Columns",
                "Working Rows",
                "Working Columns",
                "Target Column",
                "Number of Classes",
                "Number of Features",
                "Duplicate Rows Removed",
                "CV Folds",
            ],
            "Value": [
                data.shape[0],
                data.shape[1],
                working_data.shape[0],
                working_data.shape[1],
                target_column,
                number_of_classes,
                X.shape[1],
                duplicate_count,
                n_splits,
            ],
        }
    )
    dataset_info.to_csv(OUTPUT_FOLDER / "dataset_information.csv", index=False)

    model = DecisionTreeClassifier(
        criterion="entropy",
        random_state=RANDOM_STATE,
    )
    cv = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE
    )
    y_pred = cross_val_predict(model, X, y_encoded, cv=cv, method="predict")
    y_probability = cross_val_predict(
        model, X, y_encoded, cv=cv, method="predict_proba"
    )

    metrics = {
        "Accuracy": accuracy_score(y_encoded, y_pred),
        "Precision": precision_score(
            y_encoded, y_pred, average="weighted", zero_division=0
        ),
        "Recall": recall_score(
            y_encoded, y_pred, average="weighted", zero_division=0
        ),
        "F1": f1_score(y_encoded, y_pred, average="weighted", zero_division=0),
        "RMSE": np.sqrt(mean_squared_error(y_encoded, y_pred)),
        "MAE": mean_absolute_error(y_encoded, y_pred),
        "R2": r2_score(y_encoded, y_pred),
    }
    non_zero = y_encoded != 0
    metrics["MAPE"] = (
        np.mean(
            np.abs((y_encoded[non_zero] - y_pred[non_zero]) / y_encoded[non_zero])
        )
        * 100
        if np.any(non_zero)
        else np.nan
    )

    try:
        if number_of_classes == 2:
            positive_probability = y_probability[:, 1]
            metrics["ROC_AUC"] = roc_auc_score(y_encoded, positive_probability)
            fpr, tpr, _ = roc_curve(y_encoded, positive_probability)
            save_curve(
                OUTPUT_FOLDER / "ROC_AUC_Curve.png",
                fpr,
                tpr,
                f"Decision Tree ROC-AUC = {metrics['ROC_AUC']:.4f}",
                "False Positive Rate",
                "True Positive Rate",
                "Decision Tree ROC Curve — Cross Validation",
            )
            metrics["PR_AUC"] = average_precision_score(
                y_encoded, positive_probability
            )
            precision_curve, recall_curve, _ = precision_recall_curve(
                y_encoded, positive_probability
            )
            save_curve(
                OUTPUT_FOLDER / "PR_AUC_Curve.png",
                recall_curve,
                precision_curve,
                f"PR-AUC = {metrics['PR_AUC']:.4f}",
                "Recall",
                "Precision",
                "Decision Tree Precision-Recall Curve",
            )
        else:
            metrics["ROC_AUC"] = roc_auc_score(
                y_encoded, y_probability, multi_class="ovr", average="weighted"
            )
            metrics["PR_AUC"] = average_precision_score(
                pd.get_dummies(y_encoded), y_probability, average="weighted"
            )
    except ValueError:
        metrics["ROC_AUC"] = np.nan
        metrics["PR_AUC"] = np.nan

    pd.DataFrame(
        {"Metric": list(metrics), "Value": list(metrics.values())}
    ).to_csv(OUTPUT_FOLDER / "all_metrics.csv", index=False)

    labels = np.arange(number_of_classes)
    cm = confusion_matrix(y_encoded, y_pred, labels=labels)
    pd.DataFrame(
        cm,
        index=[f"Actual_{c}" for c in label_encoder.classes_],
        columns=[f"Predicted_{c}" for c in label_encoder.classes_],
    ).to_csv(OUTPUT_FOLDER / "confusion_matrix.csv")
    report = classification_report(
        y_encoded,
        y_pred,
        labels=labels,
        target_names=[str(c) for c in label_encoder.classes_],
        zero_division=0,
    )
    (OUTPUT_FOLDER / "classification_report.txt").write_text(report, encoding="utf-8")

    pd.DataFrame(
        {
            "actual_encoded": y_encoded,
            "actual_label": label_encoder.inverse_transform(y_encoded),
            "predicted_encoded": y_pred,
            "predicted_label": label_encoder.inverse_transform(y_pred),
            "correct": y_encoded == y_pred,
        }
    ).to_csv(OUTPUT_FOLDER / "cross_validated_predictions.csv", index=False)

    outlier_mask = np.zeros(len(X), dtype=bool)
    for column in X.select_dtypes(include=np.number).columns:
        q1, q3 = X[column].quantile([0.25, 0.75])
        iqr = q3 - q1
        outlier_mask |= (
            (X[column] < q1 - 1.5 * iqr) | (X[column] > q3 + 1.5 * iqr)
        ).to_numpy()

    groups = {"outliers": outlier_mask, "normal": ~outlier_mask}
    sensitivity_rows = []
    for group, mask in groups.items():
        sensitivity_rows.append(
            {
                "Group": group,
                "Count": int(mask.sum()),
                "Precision": precision_score(
                    y_encoded[mask], y_pred[mask], average="weighted", zero_division=0
                ) if mask.any() else np.nan,
                "Recall": recall_score(
                    y_encoded[mask], y_pred[mask], average="weighted", zero_division=0
                ) if mask.any() else np.nan,
                "F1": f1_score(
                    y_encoded[mask], y_pred[mask], average="weighted", zero_division=0
                ) if mask.any() else np.nan,
            }
        )
    sensitivity_df = pd.DataFrame(sensitivity_rows)
    sensitivity_df.to_csv(OUTPUT_FOLDER / "outlier_sensitivity.csv", index=False)

    outlier_row = sensitivity_df.loc[sensitivity_df["Group"] == "outliers"].iloc[0]
    normal_row = sensitivity_df.loc[sensitivity_df["Group"] == "normal"].iloc[0]
    sensitivity_difference = (
        float(normal_row["F1"] - outlier_row["F1"])
        if pd.notna(outlier_row["F1"]) and pd.notna(normal_row["F1"])
        else np.nan
    )

    model.fit(X, y_encoded)
    permutation = permutation_importance(
        model, X, y_encoded, scoring="f1_weighted", n_repeats=5,
        random_state=RANDOM_STATE, n_jobs=-1
    )
    importance = pd.DataFrame(
        {
            "Feature": X.columns,
            "Permutation Importance Mean": permutation.importances_mean,
            "Permutation Importance Std": permutation.importances_std,
            "Tree Importance": model.feature_importances_,
        }
    ).sort_values("Permutation Importance Mean", ascending=False)
    importance.to_csv(OUTPUT_FOLDER / "permutation_feature_importance.csv", index=False)
    tree_importance = importance.sort_values("Tree Importance", ascending=False)
    tree_importance.to_csv(OUTPUT_FOLDER / "feature_importance.csv", index=False)

    top_features = tree_importance.head(min(20, len(tree_importance)))
    plt.figure(figsize=(10, 7))
    plt.barh(top_features["Feature"][::-1], top_features["Tree Importance"][::-1])
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Decision Tree Feature Importance — Solar Dataset")
    plt.tight_layout()
    plt.savefig(OUTPUT_FOLDER / "feature_importance.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("Decision Tree Confusion Matrix — Solar Dataset")
    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Class")
    plt.xticks(labels, label_encoder.classes_, rotation=45)
    plt.yticks(labels, label_encoder.classes_)
    for row_index in range(cm.shape[0]):
        for column_index in range(cm.shape[1]):
            plt.text(
                column_index,
                row_index,
                cm[row_index, column_index],
                ha="center",
                va="center",
            )
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(OUTPUT_FOLDER / "confusion_matrix.png", dpi=300)
    plt.close()

    if pd.notna(outlier_row["F1"]):
        plt.figure(figsize=(7, 5))
        plt.bar(["Normal F1", "Outlier F1"], [normal_row["F1"], outlier_row["F1"]])
        plt.ylabel("F1 Score")
        plt.title("Decision Tree Outlier Sensitivity — Solar Dataset")
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(OUTPUT_FOLDER / "outlier_sensitivity.png", dpi=300)
        plt.close()

    performance_results = pd.DataFrame(
        {
            "Metric": [
                "Accuracy",
                "Precision",
                "Recall",
                "F1 Score",
                "ROC-AUC",
                "PR-AUC",
                "RMSE",
                "MAE",
                "MAPE (%)",
                "P-Square (R2)",
                "Outlier Precision",
                "Outlier Recall",
                "Outlier F1",
                "Normal Precision",
                "Normal Recall",
                "Normal F1",
                "Outlier Sensitivity Difference",
            ],
            "Score": [
                metrics["Accuracy"],
                metrics["Precision"],
                metrics["Recall"],
                metrics["F1"],
                metrics["ROC_AUC"],
                metrics["PR_AUC"],
                metrics["RMSE"],
                metrics["MAE"],
                metrics["MAPE"],
                metrics["R2"],
                outlier_row["Precision"],
                outlier_row["Recall"],
                outlier_row["F1"],
                normal_row["Precision"],
                normal_row["Recall"],
                normal_row["F1"],
                sensitivity_difference,
            ],
        }
    )
    performance_results.to_csv(OUTPUT_FOLDER / "performance_results.csv", index=False)

    pd.DataFrame(
        {
            "Parameter": [
                "Model",
                "Criterion",
                "Cross Validation",
                "Number of Folds",
                "Shuffle",
                "Random State",
                "Number of Features",
                "Number of Classes",
            ],
            "Value": [
                "Decision Tree Classifier",
                "Entropy",
                "Stratified K-Fold",
                n_splits,
                True,
                RANDOM_STATE,
                X.shape[1],
                number_of_classes,
            ],
        }
    ).to_csv(OUTPUT_FOLDER / "model_details.csv", index=False)

    print(f"Dataset: {data.shape[0]} rows, {X.shape[1]} features")
    print(f"Target: {target_column}; classes: {list(label_encoder.classes_)}")
    print(f"{n_splits}-fold CV F1: {metrics['F1']:.4f}")
    print(f"Outputs: {OUTPUT_FOLDER}")


if __name__ == "__main__":
    main()
