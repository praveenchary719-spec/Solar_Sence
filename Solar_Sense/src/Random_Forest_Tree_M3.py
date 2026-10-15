# ================================================================
# RANDOM FOREST CLASSIFIER
# SOLAR GENERATION PREDICTION USING RAW DATASET
# ================================================================
#
# IMPORTANT:
# The original RAW dataset is NEVER modified.
#
# All preprocessing is performed on copies / inside a pipeline.
#
# OUTPUTS:
#   1. Accuracy
#   2. Precision
#   3. Recall
#   4. F1 Score
#   5. Confusion Matrix
#   6. Performance Graph
#   7. Actual vs Predicted Chart
#   8. Class Distribution Chart
#   9. Feature Importance Chart
#  10. Random Forest Tree Visualization
#  11. Classification Report
#  12. Test Predictions
#  13. Trained Model
#  14. Random Forest Parameters
#
# Output folders are created ONLY if they do not already exist.
# ================================================================

import os
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# ================================================================
DATASET_PATH = os.path.join(PROJECT_ROOT, "dataset", "Plant_1_Merged_Data.csv")

# ================================================================
OUTPUT_FOLDER = os.path.join(PROJECT_ROOT, "outputs", "Random_Forest_Tree_M3_Outputs")

# ================================================================
METRICS_FOLDER = os.path.join(OUTPUT_FOLDER, "metrics")
PREDICTIONS_FOLDER = os.path.join(OUTPUT_FOLDER, "predictions")
CONFUSION_FOLDER = os.path.join(OUTPUT_FOLDER, "confusion_matrix")
CHARTS_FOLDER = os.path.join(OUTPUT_FOLDER, "charts")
TREE_FOLDER = os.path.join(OUTPUT_FOLDER, "random_forest_tree")
FEATURE_FOLDER = os.path.join(OUTPUT_FOLDER, "feature_importance")
MODEL_FOLDER = os.path.join(OUTPUT_FOLDER, "model")

# ================================================================
folders = [
    METRICS_FOLDER,
    PREDICTIONS_FOLDER,
    CONFUSION_FOLDER,
    CHARTS_FOLDER,
    TREE_FOLDER,
    FEATURE_FOLDER,
    MODEL_FOLDER,
]
for folder in folders:
    os.makedirs(folder, exist_ok=True)

# ================================================================
print("\n")
print("=" * 80)
print("          RANDOM FOREST SOLAR GENERATION PREDICTION")
print("=" * 80)

# ================================================================
if not os.path.exists(DATASET_PATH):
    print("\nERROR: Dataset not found.")
    print("\nCheck dataset path:")
    print(DATASET_PATH)
    raise SystemExit
print("\nDataset found successfully.")

# ================================================================
df = pd.read_csv(DATASET_PATH)
print("\nDataset loaded successfully.")
print("Number of rows    :", df.shape[0])
print("Number of columns :", df.shape[1])

# ================================================================
data = df.copy()

# ================================================================
print("\nDataset columns:")
print(list(data.columns))
print("\nFirst five records:")
print(data.head())

# ================================================================
possible_targets = [
    "Is_Daytime",
    "DC_POWER_Tier",
    "Generation_Status"
]
target_column = None
for column in possible_targets:
    if column in data.columns:
        target_column = column
        break

# If target is not present in raw dataset, create Is_Daytime (standard binary target in Solar_Sense)
if target_column is None:
    min_irrad = data["IRRADIATION"].min() if "IRRADIATION" in data.columns else 0
    data["Is_Daytime"] = (data["IRRADIATION"] > min_irrad).astype(int)
    target_column = "Is_Daytime"
    print("\nEngineered target 'Is_Daytime' created (1 if IRRADIATION > min else 0).")

# ================================================================
if target_column is None:
    print("\nTarget column was not automatically detected.")
    print("\nAvailable columns:")
    print(list(data.columns))
    print("\nSet target_column manually in the program.")
    raise SystemExit
print("\nTarget column:", target_column)

# ================================================================
data_model = data.dropna(subset=[target_column]).copy()
print("\nRecords used for modeling:", len(data_model))

# ================================================================
# Extract HOUR from DATE_TIME if available to enrich features
if "DATE_TIME" in data_model.columns:
    try:
        data_model["HOUR"] = pd.to_datetime(data_model["DATE_TIME"]).dt.hour
    except Exception:
        pass

# Determine columns to drop to prevent direct data leakage
drop_columns = [target_column]
if "DATE_TIME" in data_model.columns:
    drop_columns.append("DATE_TIME")

if target_column == "Is_Daytime":
    # Drop IRRADIATION (which defines Is_Daytime) and direct power generation
    for leak_col in ["IRRADIATION", "DC_POWER", "AC_POWER"]:
        if leak_col in data_model.columns:
            drop_columns.append(leak_col)
elif target_column == "DC_POWER_Tier":
    for leak_col in ["DC_POWER", "AC_POWER"]:
        if leak_col in data_model.columns:
            drop_columns.append(leak_col)

X = data_model.drop(columns=drop_columns).copy()
y = data_model[target_column].copy()

# ================================================================
print("\nTarget class distribution:")
print(y.value_counts())

# ================================================================
empty_columns = X.columns[X.isnull().all()].tolist()
if len(empty_columns) > 0:
    print("\nCompletely empty columns found:")
    print(empty_columns)
    X = X.drop(columns=empty_columns)

# ================================================================
numeric_features = X.select_dtypes(include=np.number).columns.tolist()

# ================================================================
categorical_features = X.select_dtypes(include=["object", "str", "category", "bool"]).columns.tolist()
print("\nNumeric features:")
print(numeric_features)
print("\nCategorical features:")
print(categorical_features)

# ================================================================
numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])

# ================================================================
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

# ================================================================
transformers_list = []
if len(numeric_features) > 0:
    transformers_list.append(("numeric", numeric_transformer, numeric_features))
if len(categorical_features) > 0:
    transformers_list.append(("categorical", categorical_transformer, categorical_features))

preprocessor = ColumnTransformer(
    transformers=transformers_list,
    remainder="drop",
)

# ================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print("\nTraining samples:", len(X_train))
print("Testing samples :", len(X_test))

# ================================================================
random_forest = RandomForestClassifier(
    n_estimators=100,
    criterion="gini",
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=5,
    max_features="sqrt",
    bootstrap=True,
    random_state=42,
    n_jobs=-1,
)

# ================================================================
model = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", random_forest)])

# ================================================================
print("\nTraining Random Forest...")
model.fit(X_train, y_train)
print("Random Forest training completed.")

# ================================================================
print("\nGenerating predictions...")
y_pred = model.predict(X_test)
print("Prediction completed.")

# ================================================================
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

# ================================================================
print("\n")
print("=" * 80)
print("           RANDOM FOREST PERFORMANCE")
print("=" * 80)
print(f"\nAccuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print("\nPercentage:")
print(f"Accuracy  : {accuracy * 100:.2f}%")
print(f"Precision : {precision * 100:.2f}%")
print(f"Recall    : {recall * 100:.2f}%")
print(f"F1 Score  : {f1 * 100:.2f}%")

# ================================================================
metrics_df = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
    "Score": [accuracy, precision, recall, f1],
    "Percentage": [accuracy * 100, precision * 100, recall * 100, f1 * 100],
})
metrics_path = os.path.join(METRICS_FOLDER, "random_forest_metrics.csv")
metrics_df.to_csv(metrics_path, index=False)

# ================================================================
classification_report_result = classification_report(
    y_test, y_pred, output_dict=True, zero_division=0
)
classification_report_df = pd.DataFrame(classification_report_result).transpose()
classification_report_path = os.path.join(METRICS_FOLDER, "classification_report.csv")
classification_report_df.to_csv(classification_report_path)

# ================================================================
cm = confusion_matrix(y_test, y_pred)
random_forest_classifier = model.named_steps["classifier"]
class_labels = random_forest_classifier.classes_
print("\n")
print("=" * 80)
print("               CONFUSION MATRIX")
print("=" * 80)
print(cm)

# ================================================================
cm_df = pd.DataFrame(
    cm,
    index=["Actual_" + str(label) for label in class_labels],
    columns=["Predicted_" + str(label) for label in class_labels],
)
cm_csv_path = os.path.join(CONFUSION_FOLDER, "confusion_matrix.csv")
cm_df.to_csv(cm_csv_path)

# ================================================================
plt.figure(figsize=(8, 6))
plt.imshow(cm, cmap="Blues")
plt.title("Random Forest - Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.xticks(range(len(class_labels)), class_labels, rotation=45)
plt.yticks(range(len(class_labels)), class_labels)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, str(cm[i, j]), ha="center", va="center")
plt.colorbar()
plt.tight_layout()
confusion_image_path = os.path.join(CONFUSION_FOLDER, "confusion_matrix.png")
plt.savefig(confusion_image_path, dpi=300, bbox_inches="tight")
plt.close()

# ================================================================
metric_names = ["Accuracy", "Precision", "Recall", "F1 Score"]
metric_values = [accuracy * 100, precision * 100, recall * 100, f1 * 100]
plt.figure(figsize=(10, 6))
bars = plt.bar(metric_names, metric_values)
plt.title("Random Forest Performance")
plt.xlabel("Evaluation Metric")
plt.ylabel("Score (%)")
plt.ylim(0, 100)
for bar, value in zip(bars, metric_values):
    plt.text(bar.get_x() + bar.get_width() / 2, value + 1, f"{value:.2f}%", ha="center")
plt.tight_layout()
performance_path = os.path.join(CHARTS_FOLDER, "performance_graph.png")
plt.savefig(performance_path, dpi=300, bbox_inches="tight")
plt.close()

# ================================================================
actual_counts = y_test.value_counts()
predicted_counts = pd.Series(y_pred).value_counts()
comparison_df = pd.DataFrame({"Actual": actual_counts, "Predicted": predicted_counts}).fillna(0)
comparison_df = comparison_df.reindex(class_labels)
plt.figure(figsize=(9, 6))
x = np.arange(len(class_labels))
width = 0.35
plt.bar(x - width / 2, comparison_df["Actual"], width, label="Actual")
plt.bar(x + width / 2, comparison_df["Predicted"], width, label="Predicted")
plt.xlabel("Solar Generation Class")
plt.ylabel("Number of Observations")
plt.title("Actual vs Predicted Solar Generation")
plt.xticks(x, class_labels)
plt.legend()
plt.tight_layout()
actual_predicted_path = os.path.join(CHARTS_FOLDER, "actual_vs_predicted.png")
plt.savefig(actual_predicted_path, dpi=300, bbox_inches="tight")
plt.close()

# ================================================================
class_counts = y.value_counts()
plt.figure(figsize=(8, 6))
plt.bar(class_counts.index.astype(str), class_counts.values)
plt.xlabel("Solar Generation Class")
plt.ylabel("Number of Observations")
plt.title("Solar Generation Class Distribution")
plt.xticks(rotation=45)
plt.tight_layout()
class_distribution_path = os.path.join(CHARTS_FOLDER, "class_distribution.png")
plt.savefig(class_distribution_path, dpi=300, bbox_inches="tight")
plt.close()

# ================================================================
feature_names = model.named_steps["preprocessor"].get_feature_names_out()

# ================================================================
feature_importances = random_forest.feature_importances_
feature_importance_df = pd.DataFrame({"Feature": feature_names, "Importance": feature_importances})
feature_importance_df = feature_importance_df.sort_values(by="Importance", ascending=False)

# ================================================================
feature_importance_path = os.path.join(FEATURE_FOLDER, "feature_importance.csv")
feature_importance_df.to_csv(feature_importance_path, index=False)

# ================================================================
top_features = feature_importance_df.head(15).sort_values(by="Importance")
plt.figure(figsize=(10, 7))
plt.barh(top_features["Feature"], top_features["Importance"])
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Top 15 Random Forest Feature Importances")
plt.tight_layout()
feature_chart_path = os.path.join(FEATURE_FOLDER, "feature_importance.png")
plt.savefig(feature_chart_path, dpi=300, bbox_inches="tight")
plt.close()

# ================================================================
print("\n")
print("=" * 80)
print("             RANDOM FOREST TREE VISUALIZATION")
print("=" * 80)

# Get the trained Random Forest
random_forest_classifier = model.named_steps["classifier"]

# Select first tree
first_tree = random_forest_classifier.estimators_[0]
print("\nDisplaying Random Forest Tree 1...")
plt.figure(figsize=(24, 14))

# ================================================================
plot_tree(
    first_tree,
    feature_names=feature_names,
    class_names=[str(label) for label in class_labels],
    filled=True,
    rounded=True,
    proportion=True,
    impurity=False,
    precision=2,
    max_depth=3,
    fontsize=10,
)
plt.title("Random Forest - Tree 1 (Top 3 Levels)", fontsize=18, pad=24)
plt.tight_layout(pad=3.0)

tree_image_path = os.path.join(TREE_FOLDER, "random_forest_tree_1.png")
plt.savefig(tree_image_path, dpi=300, bbox_inches="tight")
plt.close()
print("\nRandom Forest Tree 1 displayed successfully.")
print("Tree image saved at:")
print(tree_image_path)

# ================================================================
test_predictions = X_test.copy()
test_predictions["Actual"] = y_test.values
test_predictions["Predicted"] = y_pred
prediction_path = os.path.join(PREDICTIONS_FOLDER, "test_predictions.csv")
test_predictions.to_csv(prediction_path, index=False)

# ================================================================
model_path = os.path.join(MODEL_FOLDER, "random_forest_model.pkl")
with open(model_path, "wb") as file:
    pickle.dump(model, file)

# ================================================================
parameters_df = pd.DataFrame({
    "Parameter": [
        "Algorithm",
        "Number of Trees",
        "Criterion",
        "Maximum Depth",
        "Minimum Samples Split",
        "Minimum Samples Leaf",
        "Maximum Features",
        "Bootstrap",
        "Random State",
    ],
    "Value": [
        "Random Forest Classifier",
        random_forest.n_estimators,
        random_forest.criterion,
        random_forest.max_depth,
        random_forest.min_samples_split,
        random_forest.min_samples_leaf,
        random_forest.max_features,
        random_forest.bootstrap,
        random_forest.random_state,
    ],
})
parameters_path = os.path.join(METRICS_FOLDER, "random_forest_parameters.csv")
parameters_df.to_csv(parameters_path, index=False)

# ================================================================
print("\n")
print("=" * 80)
print("          RANDOM FOREST COMPLETED SUCCESSFULLY")
print("=" * 80)
print("\nOriginal RAW dataset was NOT modified.")
print("\nDataset:")
print(DATASET_PATH)
print("\nAll outputs stored in:")
print(OUTPUT_FOLDER)
print("\n")
print("FINAL PERFORMANCE")
print("-" * 50)
print(f"Accuracy  : {accuracy * 100:.2f}%")
print(f"Precision : {precision * 100:.2f}%")
print(f"Recall    : {recall * 100:.2f}%")
print(f"F1 Score  : {f1 * 100:.2f}%")
print("\n")
print("OUTPUT FOLDERS")
print("-" * 50)
print("1. metrics")
print("2. predictions")
print("3. confusion_matrix")
print("4. charts")
print("5. random_forest_tree")
print("6. feature_importance")
print("7. model")
print("\n")
print("=" * 80)
print("                PROGRAM FINISHED")
print("=" * 80)
