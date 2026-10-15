import pandas as pd
import numpy as np
from pathlib import Path




# ==========================================================
# Load Solar Sense Dataset
# Original dataset is NOT modified
# ==========================================================


DATASET_DIR = Path(__file__).resolve().parent.parent / "dataset"

# Prefer the target-encoded input if available; otherwise fall back to the raw dataset.
target_file = DATASET_DIR / "clean_target_encode_M2.csv"
raw_file = DATASET_DIR / "Plant_1_Merged_Data.csv"

if target_file.exists():
    print(f"Using input file: {target_file}")
    df = pd.read_csv(target_file)
elif raw_file.exists():
    print(f"Warning: {target_file.name} not found. Falling back to raw dataset: {raw_file.name}")
    df = pd.read_csv(raw_file)
else:
    raise FileNotFoundError(
        f"Input file not found. Place either '{target_file.name}' or '{raw_file.name}' in {DATASET_DIR}"
    )


# Create copy
data = df.copy()




# ==========================================================
# 1. Remove Leading and Trailing Spaces
# ==========================================================


for col in data.select_dtypes(include=["object", "str"]).columns:
   data[col] = data[col].str.strip()




# ==========================================================
# 2. Identify Missing Values
# ==========================================================


print("Missing Values Before Cleaning:")
print(data.isnull().sum())




# ==========================================================
# 3. Remove Duplicate Records
# ==========================================================


duplicate_count = data.duplicated().sum()


data = data.drop_duplicates()


print("\nDuplicate Records Removed:",
     duplicate_count)




# ==========================================================
# 4. Identify Numerical and Categorical Columns
# ==========================================================


num_cols = data.select_dtypes(
   include=np.number
).columns.tolist()


cat_cols = data.select_dtypes(
   exclude=np.number
).columns.tolist()


cat_cols = [col for col in cat_cols if col != "DATE_TIME"]


print("\nNumerical Columns:")
print(num_cols)


print("\nCategorical Columns:")
print(cat_cols)




# ==========================================================
# 5. Fill Missing Numerical Values with Mean
# ==========================================================


for col in num_cols:


   mean_value = data[col].mean()


   data[col] = data[col].fillna(mean_value)




# ==========================================================
# 6. Fill Missing Categorical Values with Mode
# ==========================================================


for col in cat_cols:


   mode_values = data[col].dropna().mode()
   if mode_values.empty:
       raise ValueError(f"Column '{col}' does not have a mode to use for missing values.")
   mode_value = mode_values.iloc[0]


   data[col] = data[col].fillna(mode_value)




# ==========================================================
# 7. Pandas-Based Embedding Encoding
# ==========================================================


embedding_output = pd.DataFrame()




embedding_size = 3     # Number of embedding dimensions




for col in cat_cols:


   # Get unique categories
   categories = data[col].unique()




   # Create embedding values
   embedding_matrix = {}


   for index, category in enumerate(categories):


       vector = np.zeros(embedding_size)


       vector[index % embedding_size] = 1


       embedding_matrix[category] = vector




   # Convert category to embedding vector


   embeddings = data[col].map(
       embedding_matrix
   )




   embedding_df = pd.DataFrame(
       embeddings.tolist(),
       columns=[
           f"Embedding_{col}_1",
           f"Embedding_{col}_2",
           f"Embedding_{col}_3"
       ]
   )




   embedding_output = pd.concat(
       [
           embedding_output,
           embedding_df
       ],
       axis=1
   )




# ==========================================================
# 8. Merge Numerical Columns + Embeddings
# ==========================================================


final_output = pd.concat(
   [
       data[num_cols].reset_index(drop=True),
       embedding_output.reset_index(drop=True)
   ],
   axis=1
)




# ==========================================================
# 9. Check Missing Values After Processing
# ==========================================================


print("\nMissing Values After Cleaning:")
print(final_output.isnull().sum())




# ==========================================================
# 10. Save Result
# ==========================================================


DATASET_DIR.mkdir(parents=True, exist_ok=True)
final_output.to_csv(
   DATASET_DIR / "clean_embedded_encode_M2.csv",
   index=False
)




print("\n======================================")
print("Embedding Encoding Completed")
print("Original dataset is NOT modified")
print("Output File:")
print("clean_embedded_encode_M2.csv")
print("======================================")
