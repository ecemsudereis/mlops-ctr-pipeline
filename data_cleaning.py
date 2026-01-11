import pandas as pd
from sklearn.feature_extraction import FeatureHasher

# 1. Data Ingestion & Rebalancing (Mandatory Pattern)
# Loading raw data from Kaggle source
print("Step 1: Ingesting raw data and applying REBALANCING pattern...")
df = pd.read_csv('train.gz', compression='gzip', nrows=500000) 

# Addressing highly skewed data via Downsampling [cite: 38]
df_1 = df[df['click'] == 1]
df_0 = df[df['click'] == 0].sample(len(df_1), random_state=42)
df = pd.concat([df_1, df_0]).sample(frac=1).reset_index(drop=True)
print(f"Rebalancing complete. Current dataset size: {df.shape}")

# 2. Feature Engineering: Feature Cross (Mandatory Pattern)
# Concatenating device type and hour to capture interaction effects [cite: 32]
print("Step 2: Generating FEATURE CROSS (device_type + hour)...")
df['device_hour'] = df['device_type'].astype(str) + "_" + df['hour'].astype(str)

# 3. Handling High Cardinality: Hashed Feature Pattern (Mandatory Pattern)
# Mapping high-cardinality categorical data into a fixed number of buckets [cite: 31]
print("Step 3: Applying HASHED FEATURE pattern to high-cardinality columns...")
high_card_cols = ['site_id', 'app_id', 'device_id', 'device_hour']

for col in high_card_cols:
    hasher = FeatureHasher(n_features=1024, input_type='string')
    # Converting categorical input into fixed buckets to accept trade-offs of collision [cite: 31]
    hashed = hasher.transform(df[col].astype(str).apply(lambda x: [x]))
    df[col] = hashed.toarray().sum(axis=1)

# 4. Data Export (CI/CD Pipeline Output)
# Saving cleaned data as .parquet for efficient storage and downstream training [cite: 25]
print("Step 4: Exporting processed data to .parquet format...")
df.to_parquet('processed_data.parquet', index=False)

print("--------------------------------------------------")
print("SUCCESS: Data processing pipeline complete.")
print("Output File: 'processed_data.parquet' is ready for training.")
print("--------------------------------------------------")






df_raw = pd.read_csv('train.gz', compression='gzip', nrows=500000)
raw_rows = len(df_raw)
initial_features = len(df_raw.columns)
print(f"REPORT - Raw Rows: {raw_rows}")
print(f"REPORT - Initial Feature Count: {initial_features}")


df_1 = df_raw[df_raw['click'] == 1]
df_0 = df_raw[df_raw['click'] == 0].sample(len(df_1), random_state=42)
df = pd.concat([df_1, df_0]).sample(frac=1).reset_index(drop=True)
balanced_rows = len(df)
print(f"REPORT - Balanced Rows: {balanced_rows}")


df['device_hour'] = df['device_type'].astype(str) + "_" + df['hour'].astype(str)


final_features = len(df.columns)
print(f"REPORT - Final Feature Count: {final_features}")