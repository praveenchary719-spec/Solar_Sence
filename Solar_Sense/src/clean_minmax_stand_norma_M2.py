# ---------------------------------------------------------
# Numeric column pre-process Techniques
# Mean, Median, mode
# Feature Scaling, Standardization, and Normalization
# Save all results in ONE CSV file (clean_minmax_stand_norma_M2.csv)
# ------------------------------------------------------------
# check the scikit-learn library is installed or not
# if not install with the command "pip install scikit-learn"


import matplotlib.pyplot as plt
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, Normalizer, StandardScaler


# ------------------------------------------------------------
# Load Dataset
# ------------------------------------------------------------
DATASET_DIR = Path(__file__).resolve().parent.parent / "dataset"
INPUT_FILE = DATASET_DIR / "Plant_1_Merged_Data.csv"
OUTPUT_FILE = DATASET_DIR / "clean_minmax_stand_norma_M2.csv"


def main() -> None:
    df = pd.read_csv(INPUT_FILE)

    print("Original Dataset")
    print("------------------------")
    print(df.head())

    print("Dataset Shape:", df.shape)
    print("\nData Types:")
    print("------------------------")
    print(df.dtypes)

    print("\nMissing Values:")
    print("------------------------")
    print(df.isnull().sum())

    print("\nDuplicate Records:", df.duplicated().sum())

    # ---------------------------------------------------
    # Step 2: Remove Duplicate Records
    # ---------------------------------------------------
    df = df.drop_duplicates()

    # ---------------------------------------------------
    # Step 3: Handle Missing Values
    # ---------------------------------------------------
    numerical_columns = df.select_dtypes(include=["number"]).columns
    for column in numerical_columns:
        df[column] = df[column].fillna(df[column].mean())

    categorical_columns = df.select_dtypes(include=["object", "str"]).columns
    for column in categorical_columns:
        df[column] = df[column].str.strip()
        mode_values = df[column].dropna().mode()
        if mode_values.empty:
            raise ValueError(
                f"Column '{column}' does not have a mode to use for missing values."
            )
        df[column] = df[column].fillna(mode_values.iloc[0])

    # ------------------------------------------------------------
    # Select Numeric Columns
    # ------------------------------------------------------------
    numeric_columns = df.select_dtypes(include=["number"]).columns
    if len(numeric_columns) == 0:
        raise ValueError("No numeric columns were found for scaling.")

    print("\nNumeric Columns:")
    print(list(numeric_columns))

    # ------------------------------------------------------------
    # Standardization (Z-score)
    # Mean = 0, Standard Deviation = 1
    # ------------------------------------------------------------
    standard_scaler = StandardScaler()
    standardized = standard_scaler.fit_transform(df[numeric_columns])
    for i, col in enumerate(numeric_columns):
        df[f"{col}_Standardized"] = standardized[:, i]

    # ------------------------------------------------------------
    # Feature Scaling (Min-Max Scaling)
    # Values between 0 and 1
    # ------------------------------------------------------------
    minmax_scaler = MinMaxScaler()
    scaled = minmax_scaler.fit_transform(df[numeric_columns])
    for i, col in enumerate(numeric_columns):
        df[f"{col}_Scaled"] = scaled[:, i]

    # ------------------------------------------------------------
    # Normalization (L2 Normalization)
    # Each row becomes a unit vector
    # ------------------------------------------------------------
    normalizer = Normalizer(norm="l2")
    normalized = normalizer.fit_transform(df[numeric_columns])
    for i, col in enumerate(numeric_columns):
        df[f"{col}_Normalized"] = normalized[:, i]

    # ------------------------------------------------------------
    # Display Results after pre-processing (Verify Dataset)
    # ------------------------------------------------------------
    print("\n Display Results after Preprocessed Dataset")
    print(df.head())

    print("\nDataset Shape:", df.shape)

    print("\nDataset Information")
    df.info()

    print("\nColumns in Dataset:")
    print(df.columns)

    print("\nMissing Values After Preprocessing")
    print(df.isnull().sum())

    print("\nDuplicate Records After Preprocessing")
    print(df.duplicated().sum())

    # ---------------------------------------------------
    # Step 8: Save Preprocessed Dataset
    # ---------------------------------------------------
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print("\nPreprocessed dataset saved successfully.")

    def plot_histogram_grid(columns, title, color):
        cols_per_row = 4
        rows = (len(columns) + cols_per_row - 1) // cols_per_row
        fig, axes = plt.subplots(rows, cols_per_row, figsize=(20, max(6, rows * 3.5)))
        axes = axes.flatten()

        for index, column in enumerate(columns):
            axes[index].hist(df[column].dropna(), bins=20, color=color, edgecolor="black")
            axes[index].set_title(column, fontsize=10)
            axes[index].tick_params(axis="both", labelsize=8)
            axes[index].grid(alpha=0.2)

        for index in range(len(columns), len(axes)):
            axes[index].axis("off")

        fig.suptitle(title, fontsize=16)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        safe_title = title.replace(" ", "_").replace("/", "_")
        out_path = OUTPUT_FILE.parent.parent / "outputs" / "Preprocessing_Scaling" / f"{safe_title}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()

    original_columns = list(numeric_columns)
    standardized_columns = [f"{col}_Standardized" for col in numeric_columns]
    scaled_columns = [f"{col}_Scaled" for col in numeric_columns]
    normalized_columns = [f"{col}_Normalized" for col in numeric_columns]

    plot_histogram_grid(original_columns, "Histograms of Original Numeric Columns", "#4C72B0")
    plot_histogram_grid(standardized_columns, "Histograms of Standardized Columns", "#55A868")
    plot_histogram_grid(scaled_columns, "Histograms of Min-Max Scaled Columns", "#C44E52")
    plot_histogram_grid(normalized_columns, "Histograms of Normalized Columns", "#8172B3")


if __name__ == "__main__":
    main()
