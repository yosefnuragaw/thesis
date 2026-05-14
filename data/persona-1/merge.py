import pandas as pd
from pathlib import Path

# Define the target directory
base_dir = Path('/home/yosef/ws/thesis/data/persona-1')

# Find all test_infer.csv files strictly inside the subdirectories
# This avoids accidentally reading an already merged test_infer.csv in the root folder
csv_files = list(base_dir.glob('*/test_infer.csv'))

df_list = []

for file_path in csv_files:
    df = pd.read_csv(file_path)
    
    # Optional but recommended: Keep track of which folder the data came from
    df['source_folder'] = file_path.parent.name 
    
    df_list.append(df)

# Concatenate all dataframes into one
merged_df = pd.concat(df_list, ignore_index=True)

# Save the merged dataframe to the root directory
output_path = base_dir / 'test_infer.csv'
merged_df.to_csv(output_path, index=False)

print(f"Successfully merged {len(csv_files)} files and saved to:\n{output_path}")