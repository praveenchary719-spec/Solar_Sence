from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


DATASET_PATH = Path(__file__).resolve().parent.parent / "dataset" / "Plant_1_Merged_Data.csv"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "Dataset_Missing_Values"

# 1. Load the dataset
# Tip: Use raw string syntax (r"...") or forward slashes to prevent Windows path escaping issues
df = pd.read_csv(DATASET_PATH)

# 2. Retrieve data in different ways
print("--- First 5 Rows ---")
print(df.head())

print("\n---- First 6 Columns ----")
subset = df.iloc[:, 0:6]
print(subset)

# 3. Identify missing values per column
missing_counts = df.isnull().sum()
print("\n----- Missing Values Per Column: ----------")
print(missing_counts)
print("-" * 40)

# 4. Detect duplicate rows
duplicate_rows = df[df.duplicated()]
print(f"Total duplicate rows detected: {len(duplicate_rows)}")
print(duplicate_rows)
print("-" * 40)

# 5. Produce a missingness heatmap
fig = plt.figure()
fig.set_figwidth(10)
fig.set_figheight(6)
# Fixed comment syntax and trailing colon below
sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap="viridis")
plt.title("Missing Values Heatmap")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
plt.savefig(OUTPUT_DIR / "Missing_Values_Heatmap.png", bbox_inches="tight")
plt.close()
print(f"Missing Values Heatmap saved to: {OUTPUT_DIR / 'Missing_Values_Heatmap.png'}")
categorical_columns = [column for column in df.columns if df[column].dtype == "object"]
categorical_df = df[categorical_columns]
print("-----------------------------------")
print("12. Display Categorical (Object) Attributes:")
print("-----------------------------------")
print(categorical_df)
print("Missing Values in Categorical Attributes")
print("=" * 50)
print(categorical_df.isnull().sum())
print("\n13. Total Missing Categorical Values:", categorical_df.isnull().sum().sum())
print("-----------------------------------")
print("14. Missing Values in Each Column")
print("-" * 40)
print(df.isnull().sum())
total_missing = df.isnull().sum().sum()
print("-----------------------------------")
print("15. Total Missing Values:", total_missing)
print("----------------------------------")
duplicate_count = df.duplicated().sum()
print("16. Number of Duplicate Records:", duplicate_count)
print("-----------------------------------")
print("17. Statistical Overview")
print("-----------------------------------")
print(df.describe())