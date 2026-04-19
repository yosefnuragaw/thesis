import pandas as pd

df = pd.read_csv('/home/yosef/ws/thesis/data/corrigible-less/test_infer.csv')

# Drop rows where 'question' is null or empty string
df_clean = df[df['question'].notna() & (df['question'].str.strip() != '')]

df_clean.to_csv('/home/yosef/ws/thesis/data/corrigible-less/test_infer.csv', index=False)

print(f"Original rows: {len(df)}")
print(f"After cleaning: {len(df_clean)}")
print(f"Removed: {len(df) - len(df_clean)}")