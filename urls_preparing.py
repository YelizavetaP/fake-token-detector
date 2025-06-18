import pandas as pd
import ast

# 1. Load the raw URLs dataset
df = pd.read_csv('data/urls.csv')

# 2. Parse the 'addresses' column from string to Python list
#  # Example: "{'ETH': ['0xD0cC2B24980CBCCA47EF755Da88B220a82291407']}" -> ['0xD0cC2B24980CBCCA47EF755Da88B220a82291407']
def parse_addresses(addresses_str):
    try:
        addr_dict = ast.literal_eval(addresses_str) #eval dict from str
        all_addr = []
        for addr in addr_dict.values(): #add all addresses to a list
            all_addr.extend(addr)

        return all_addr
    
    except (ValueError, SyntaxError):
        return []
    
df['parsed_addresses'] = df['addresses'].apply(parse_addresses)

# 3. Explode the DataFrame so each row has one address
df_exploded = df.explode('parsed_addresses').rename(columns={'parsed_addresses': 'address'})

# 4. Drop rows where address is null or empty
df_exploded = df_exploded[df_exploded['address'].notna() & (df_exploded['address'] != '')]
df.dropna()

# 5. Remove duplicate addresses, keeping the first occurrence
df_exploded = df_exploded.drop_duplicates(subset='address')

# 6. (Optional) Reset index after cleaning
df_exploded = df_exploded.reset_index(drop=True)

# 7. Save the cleaned URLs dataset
cleaned_file_path = 'data/urls_cleaned.csv'
pd.DataFrame.to_csv(df_exploded, cleaned_file_path, index=False)

# 8. Display basic info
print("Cleaned URLs DataFrame Info:")
print(df_exploded.info())
print('--' * 40)
print(df_exploded.dtypes)
print('--' * 40)
print(df_exploded.head())

# Provide path to the cleaned file for downstream merging
print(f"Cleaned URLs dataset saved to: {cleaned_file_path}")