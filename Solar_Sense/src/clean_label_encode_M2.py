import pandas as pd
import numpy as np
from pathlib import Path


from sklearn.preprocessing import LabelEncoder
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


before = data.shape[0]


data = data.drop_duplicates()


after = data.shape[0]


print("\nDuplicate Records Removed:",
     before - after)




# ==========================================================
# Separate Numerical and Categorical Columns
# ==========================================================


num_cols = data.select_dtypes(
   include=np.number
).columns.tolist()


cat_cols = data.select_dtypes(
   exclude=np.number
).columns.tolist()




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
# 6. Label Encoding
# ==========================================================


label_encoders = {}


for col in cat_cols:


   encoder = LabelEncoder()


   data[col] = encoder.fit_transform(
       data[col]
   )


   label_encoders[col] = encoder




# ==========================================================
# 7. Check Missing Values After Cleaning
# ==========================================================


print("\nMissing Values After Cleaning:")
print(data.isnull().sum())




# ==========================================================
# Save Result
# ==========================================================


DATASET_DIR.mkdir(parents=True, exist_ok=True)
data.to_csv(
   DATASET_DIR / "clean_label_encode_M2.csv",
   index=False
)




print("\n======================================")
print("Original dataset is NOT modified.")
print("Label Encoding completed successfully.")
print("Output file:")
print("clean_label_encode_M2.csv")
print("======================================")
