import pandas as pd

# 1. Load cleaned datasets
df_valid = pd.read_csv('data/Crypto_cleaned.csv')
df_scam = pd.read_csv('data/urls_cleaned.csv')

# 2. Rename 'crypturl' to 'url' in the valid tokens dataset
df_valid = df_valid.rename(columns={'crypturl': 'url'})

# 3. Assign labels
df_valid['label'], df_scam['label'] = 0, 1

# 4. Drop any extra columns to keep only one 'url' column in both
#    (valid dataset had no 'url' originally, scams dataset has 'url' already)
#    Ensure both dataframes have identical column sets
all_columns = set(df_valid.columns).union(df_scam.columns)
for col in all_columns:
    if col not in df_valid.columns:
        df_valid[col] = pd.NA  # Add missing columns with NA
    if col not in df_scam.columns:
        df_scam[col] = pd.NA  # Add missing columns with NA

# all_columns = ['url', 'label']  # Define the columns we want to keep
# df_valid = df_valid[all_columns]
# df_scam = df_scam[all_columns]

# 5. Concatenate the dataframes


# 6. Drop any duplicate columns (if needed)
#    e.g., if there was an old 'crypturl' left over, ensure it's removed


# 7. Save the final labeled dataset with unified 'url' column


# 8. Display summary
