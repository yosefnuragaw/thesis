import pandas as pd


# def remove_empty_questions(input_path: str, output_path: str) -> int:
#     """Remove rows with empty question from a CSV and save to output."""
#     df = pd.read_csv(input_path)
#     n_before = len(df)
#     df = df[df["question"].notna() & df["question"].str.strip().ne("")]
#     n_removed = n_before - len(df)
#     df.to_csv(output_path, index=False)
#     return n_removed


# if __name__ == "__main__":
#     base = "/home/yosef/ws/thesis/data/coordinate-other"

#     n1 = remove_empty_questions(f"{base}/test.csv", f"{base}/test.csv")
#     print(f"test.csv: removed {n1} rows with empty question")

#     n2 = remove_empty_questions(f"{base}/test_infer.csv", f"{base}/test_infer.csv")
#     print(f"test_infer.csv: removed {n2} rows with empty question")

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