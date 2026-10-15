import pandas as pd
import os

path = 'e:/Apps/pythonProject/Solar_Sense/dataset'
files = [f for f in os.listdir(path) if f.endswith('.csv') and 'Plant' in f]

for f in files:
    try:
        df = pd.read_csv(os.path.join(path, f))
        print(f"\n=== {f} ===")
        print(f"Shape (Rows, Columns): {df.shape}")
        print(f"Total Missing Values: {df.isnull().sum().sum()}")
        print(f"Duplicate Rows: {df.duplicated().sum()}")
        
        # Check basic stats of numeric columns
        numeric_cols = df.select_dtypes(include='number').columns
        for col in numeric_cols:
            negatives = (df[col] < 0).sum()
            if negatives > 0:
                print(f"  [!] Column '{col}' has {negatives} negative values (Anomalous).")
                
    except Exception as e:
        print(f"Error reading {f}: {e}")
