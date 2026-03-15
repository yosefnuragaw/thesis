import os
import re
import pandas as pd

# Define the base directory
base_dir = "reasoning/power-seeking"

# Regex pattern to extract the type and multiplier
file_pattern = re.compile(r"reasoning_(behavior|utility)_mul_([-\d\.]+)\.csv")

def process_directories(base_path):
    if not os.path.exists(base_path):
        print(f"Error: Directory '{base_path}' not found.")
        return

    # Iterate through all items in the base directory
    for subdir_name in os.listdir(base_path):
        subdir_path = os.path.join(base_path, subdir_name)
        
        if not os.path.isdir(subdir_path):
            continue
            
        behavior_dfs = []
        utility_dfs = []
        
        # Iterate through all files to gather them
        for filename in os.listdir(subdir_path):
            match = file_pattern.match(filename)
            if match:
                csv_type = match.group(1)
                multiplier = float(match.group(2))
                
                file_path = os.path.join(subdir_path, filename)
                
                try:
                    df = pd.read_csv(file_path)
                    
                    # Rename columns to include the multiplier so they don't overlap when merged horizontally
                    df = df.rename(columns={
                        'Score': f'Score_{multiplier}',
                        'Reasoning': f'Reasoning_{multiplier}'
                    })
                    
                    if csv_type == 'behavior':
                        behavior_dfs.append((multiplier, df))
                    elif csv_type == 'utility':
                        utility_dfs.append((multiplier, df))
                        
                except Exception as e:
                    print(f"  Error reading {filename}: {e}")

        # CONDITION: Skip the baseline "mul 0" directory entirely
        # If the only multiplier we found is 0.0, we skip this directory
        behavior_mults = [m for m, _ in behavior_dfs]
        if not behavior_mults or (len(behavior_mults) == 1 and behavior_mults[0] == 0.0):
            print(f"Skipping baseline/mul_0 directory: {subdir_name}")
            continue
            
        print(f"\nProcessing directory: {subdir_name}")

        # Sort by multiplier so columns are ordered logically from lowest to highest multiplier
        behavior_dfs.sort(key=lambda x: x[0])
        utility_dfs.sort(key=lambda x: x[0])

        # Process Behavior
        if behavior_dfs:
            sorted_b_dfs = [item[1] for item in behavior_dfs]
            
            # Merge horizontally side-by-side
            behavior_merged = pd.concat(sorted_b_dfs, axis=1)
            
            # Find all the new 'Score_*' columns
            score_cols = [col for col in behavior_merged.columns if col.startswith('Score_')]
            
            # Create a mask: True if ANY score column in the row equals 0
            mask = (behavior_merged[score_cols] == 0).any(axis=1)
            
            # Filter out the rows where the mask is True (keep rows where mask is False)
            behavior_filtered = behavior_merged[~mask]
            
            output_path = os.path.join(subdir_path, f"{subdir_name}_behavior_merged_horizontal.csv")
            behavior_filtered.to_csv(output_path, index=False)
            print(f"  Saved Behavior: Dropped {mask.sum()} rows with a Score of 0. Kept {len(behavior_filtered)} rows.")

        # Process Utility
        if utility_dfs:
            sorted_u_dfs = [item[1] for item in utility_dfs]
            
            # Merge horizontally side-by-side
            utility_merged = pd.concat(sorted_u_dfs, axis=1)
            
            score_cols = [col for col in utility_merged.columns if col.startswith('Score_')]
            mask = (utility_merged[score_cols] == 0).any(axis=1)
            utility_filtered = utility_merged[~mask]
            
            output_path = os.path.join(subdir_path, f"{subdir_name}_utility_merged_horizontal.csv")
            utility_filtered.to_csv(output_path, index=False)
            print(f"  Saved Utility: Dropped {mask.sum()} rows with a Score of 0. Kept {len(utility_filtered)} rows.")

if __name__ == "__main__":
    process_directories(base_dir)