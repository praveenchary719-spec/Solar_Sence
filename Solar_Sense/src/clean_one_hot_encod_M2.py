##—-----------------------------------------------------------------------------------------------
import pandas as pd
import numpy as np
from pathlib import Path


from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer




# ==========================================================
# Load Solar Sense Dataset
# Original dataset will NOT be modified
# ==========================================================


DATASET_DIR = Path(__file__).resolve().parent.parent / "dataset"
DATASET_PATH = DATASET_DIR / "Plant_1_Merged_Data.csv"
df = pd.read_csv(DATASET_PATH)


# Create a copy for processing
data = df.copy()


print("Original Dataset")
print("------------------------")
print(data.head())


print("Dataset Shape:", df.shape)
print("\nData Types:")
print("------------------------")
print(data.dtypes)




print("\nDuplicate Records:", df.duplicated().sum())




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


# print(data.shape)     # Outputs: rows and columns
# print(data.shape[0])  # Outputs: (Number of rows)
#print(data.shape[1])  # Outputs: (Number of columns)


before_duplicates = data.shape[0]


data = data.drop_duplicates()


after_duplicates = data.shape[0]


print("\nDuplicate Records Removed:",
     before_duplicates - after_duplicates)




# ==========================================================
# Separate Numerical and Categorical Columns
# ==========================================================


num_cols = data.select_dtypes(
   include=np.number
).columns.tolist()


cat_cols = data.select_dtypes(
   exclude=np.number
).columns.tolist()

# Remove DATE_TIME from categorical encoding — it is a timestamp, not a category
date_cols = []
if "DATE_TIME" in cat_cols:
   cat_cols.remove("DATE_TIME")
   date_cols = ["DATE_TIME"]




# ==========================================================
# 4. Fill Missing Numerical Values with Mean
# ==========================================================


if len(num_cols) > 0:


   num_imputer = SimpleImputer(
       strategy="mean"
   )


   data[num_cols] = num_imputer.fit_transform(
       data[num_cols]
   )




# ==========================================================
# 5. Fill Missing Categorical Values with Mode
# ==========================================================


if len(cat_cols) > 0:


   cat_imputer = SimpleImputer(
       strategy="most_frequent"
   )


   data[cat_cols] = cat_imputer.fit_transform(
       data[cat_cols]
   )




# ==========================================================
# 6. One-Hot Encoding
# ==========================================================


if len(cat_cols) > 0:


   encoder = OneHotEncoder(
       sparse_output=False,
       handle_unknown="ignore"
   )


   encoded_values = encoder.fit_transform(
       data[cat_cols]
   )


   ohe_col_names = encoder.get_feature_names_out(cat_cols)


   encoded_df = pd.DataFrame(
       encoded_values,
       columns=ohe_col_names
   )


   # Reset index for merging
   encoded_df.reset_index(
       drop=True,
       inplace=True
   )


   # Keep numerical columns (reset index to align with encoded_df)
   numeric_df = data[num_cols].reset_index(
       drop=True
   )


   # Preserve DATE_TIME column if present
   date_df = data[date_cols].reset_index(drop=True) if date_cols else None


   # Build final output:
   # Column order: DATE_TIME (if present) | numeric columns | one-hot encoded columns
   # This mirrors the original dataset structure — timestamp first, then values, then encoding
   dfs_to_concat = []
   if date_df is not None:
       dfs_to_concat.append(date_df)
   dfs_to_concat.append(numeric_df)
   dfs_to_concat.append(encoded_df)

   final_output = pd.concat(
       dfs_to_concat,
       axis=1
   )


else:


   final_output = data.copy()




# ==========================================================
# 7. Check Missing Values After Cleaning
# ==========================================================


print("\nMissing Values After Cleaning:")
print(final_output.isnull().sum())


# ==========================================================
# 8. Print One-Hot Encoded Columns Summary
# ==========================================================


print("\n======================================")
print("One-Hot Encoded Columns Created:")
print("======================================")
if len(cat_cols) > 0:
   for col_name in ohe_col_names:
       print(" -", col_name)
   print(f"\nTotal OHE columns created: {len(ohe_col_names)}")
   print(f"Final dataset shape: {final_output.shape}")
else:
   print("No categorical columns found for encoding.")


# ==========================================================
# Save Final Result
# ==========================================================


DATASET_DIR.mkdir(parents=True, exist_ok=True)
output_path = DATASET_DIR / "clean_one_hot_encoding_M2.csv"
final_output.to_csv(
    output_path,
    index=False
)




print("\n======================================")
print("Original dataset is NOT modified.")
print("Cleaning and One-Hot Encoding completed.")
print("Output file:")
print("clean_one_hot_encoding_M2.csv")
print("======================================")
