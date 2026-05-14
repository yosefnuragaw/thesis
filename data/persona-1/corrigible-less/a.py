# import pandas as pd

# df = pd.read_csv('/home/yosef/ws/thesis/data/corrigible-less/test_infer.csv')

# # Drop rows where 'question' is null or empty string
# df_clean = df[df['question'].notna() & (df['question'].str.strip() != '')]

# df_clean.to_csv('/home/yosef/ws/thesis/data/corrigible-less/test_infer.csv', index=False)

# print(f"Original rows: {len(df)}")
# print(f"After cleaning: {len(df_clean)}")
# print(f"Removed: {len(df) - len(df_clean)}")


import pandas as pd
import re

# Load the CSV file located in the same directory
# df = pd.read_csv('test_infer.csv')


def process_csv(input_path, output_path):
    # Load the CSV
    df = pd.read_csv(input_path)
    
    # 1. Truncate the 'question' column to end at the first '?'
    # It splits the string at '?', takes the first part, and adds the '?' back.
    # If there is no '?' in the text, it leaves the text as is.
    df['question'] = df['question'].apply(
        lambda x: x.split('?')[0] + '?' if isinstance(x, str) and '?' in x else x
    )
    
    # 2. Remove the '(A) ' or '(B) ' prefixes from columns A and B
    # The regex r'^\([A-Za-z]\)\s*' looks for a starting parenthesis, a letter, 
    # a closing parenthesis, and any trailing spaces, then replaces it with nothing.
    if 'A' in df.columns:
        df['A'] = df['A'].str.replace(r'^\([A-Za-z]\)\s*', '', regex=True)
    
    if 'B' in df.columns:
        df['B'] = df['B'].str.replace(r'^\([A-Za-z]\)\s*', '', regex=True)
        
    # Save the cleaned data to a new CSV
    df.to_csv(output_path, index=False)
    print(f"Processed CSV saved to: {output_path}")

# Example usage:
process_csv('/home/yosef/ws/thesis/data/persona-1/corrigible-less/test_infer.csv', '/home/yosef/ws/thesis/data/persona-1/corrigible-less/test_infer.csv')