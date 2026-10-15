# --------------------------------------------
# Solar Sense Dataset Preprocessing
# --------------------------------------------

from pathlib import Path

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


# Read original dataset
base_dir = Path(__file__).resolve().parent.parent
input_file = base_dir / "dataset" / "Plant_1_Merged_Data.csv"
output_file = base_dir / "dataset" / "final_preprocess_M2.csv"


df = pd.read_csv(input_file)


# Create a copy so original dataset remains unchanged
processed_df = df.copy()

print("Original Dataset Shape:", processed_df.shape)


# --------------------------------------------
# Remove Duplicate Records
# --------------------------------------------
processed_df = processed_df.drop_duplicates()


# --------------------------------------------
# Handle Missing Values
# --------------------------------------------

# Numeric Columns
numeric_cols = processed_df.select_dtypes(include=['number']).columns

for col in numeric_cols:
   processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
   processed_df[col] = processed_df[col].fillna(processed_df[col].median())


# Categorical Columns
categorical_cols = processed_df.select_dtypes(include=['object', 'string', 'category']).columns

for col in categorical_cols:
   mode_value = processed_df[col].mode()
   if not mode_value.empty:
       processed_df[col] = processed_df[col].fillna(mode_value.iloc[0])
   else:
       processed_df[col] = processed_df[col].fillna('unknown')


# --------------------------------------------
# Clean Text Data
# --------------------------------------------
for col in categorical_cols:
   processed_df[col] = processed_df[col].astype(str).str.strip().str.lower()


# --------------------------------------------
# Label Encoding
# --------------------------------------------
for col in categorical_cols:
   encoder = LabelEncoder()
   processed_df[col] = encoder.fit_transform(processed_df[col])


# --------------------------------------------
# Feature Scaling
# --------------------------------------------
scaler = StandardScaler()
processed_df[numeric_cols] = scaler.fit_transform(processed_df[numeric_cols])


# --------------------------------------------
# Save Preprocessed Dataset
# --------------------------------------------
output_file.parent.mkdir(parents=True, exist_ok=True)
processed_df.to_csv(output_file, index=False)


print("\nPreprocessing Completed Successfully!")
print("Original Dataset Shape :", df.shape)
print("Processed Dataset Shape:", processed_df.shape)
print("Saved File :", output_file)
