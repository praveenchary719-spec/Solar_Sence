import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
# Ensure the output directory exists
output_dir = Path(__file__).resolve().parent.parent / "outputs" / "Boxplots_correlation"
os.makedirs(output_dir, exist_ok=True)


# -------------------------------------------------------------
# 1. Load Solar Sense Dataset
# -------------------------------------------------------------

DATASET_PATH = Path(__file__).resolve().parent.parent / "dataset" / "Plant_1_Merged_Data.csv"
df = pd.read_csv(DATASET_PATH)


print("Dataset Loaded Successfully. Shape:", df.shape)


# -------------------------------------------------------------
# 2. Compute Correlation Matrix & Generate Heatmap
# -------------------------------------------------------------
# Select only numerical columns for correlation
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
corr_matrix = df[numerical_cols].corr()


# Print matrix to terminal
print("\n--- Correlation Matrix ---")
print(corr_matrix)


# Create Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(
   corr_matrix,
   annot=True,
   cmap="coolwarm",# from blue (cool) through white red (warm)
   fmt=".2f",
   vmin=-1,
   vmax=1,
   square=True,
   linewidths=0.5
)
plt.title("Correlation Heatmap of Numerical Features", fontsize=14, fontweight="bold")
plt.tight_layout()


# Export Heatmap
heatmap_path = os.path.join(output_dir, "correlation_heatmap.png")
plt.savefig(heatmap_path, dpi=300)
plt.close()
print(f"Exported heatmap to: {heatmap_path}")


# -------------------------------------------------------------
# 3. Produce Boxplots: Numerical Features vs DC_POWER
# -------------------------------------------------------------
target_col = "DC_POWER"


if target_col in df.columns:
   for col in numerical_cols:
       plt.figure(figsize=(6, 5))
       sns.boxplot(
           x=target_col,
           y=col,
           data=df,
           palette="Set2",
           hue=target_col,  # Prevents future deprecation warnings
           legend=False
       )
       plt.title(f"{col} vs {target_col}", fontsize=12, fontweight="bold")
       plt.xlabel(target_col)
       plt.ylabel(col)
       plt.tight_layout()


       # Export individual boxplot
       boxplot_filename = f"boxplot_{col}_vs_{target_col}.png"
       boxplot_path = os.path.join(output_dir, boxplot_filename)
       plt.savefig(boxplot_path, dpi=300)
       plt.close()
       print(f"Exported boxplot to: {boxplot_path}")
else:
   print(f"\nTarget column '{target_col}' not found in dataset. Skipping boxplots.")


print("\nAll EDA tasks completed successfully!")
