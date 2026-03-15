import os
import re
import pandas as pd

# The column that uniquely identifies the row across all files
ID_COLUMN = "question" 

# Define the base directories
base_dirs = [
    "reasoning/power-seeking/gemma", 
    "reasoning/power-seeking/mistral", 
    "reasoning/power-seeking/llama"
]

# Paths to your baseline "0" multiplier CSVs
# (Reordered to match the gemma -> mistral -> llama order of base_dirs)
BASELINE_0_BEHAVIOR = [
    "reasoning/power-seeking/gemma/google_gemma-3-1b-it/reasoning_behavior_mul_0.csv",
    "reasoning/power-seeking/mistral/mistralai_Mistral-7B-Instruct-v0.3/reasoning_behavior_mul_0.csv",
    "reasoning/power-seeking/llama/meta-llama_Llama-3.1-8B-Instruct/reasoning_behavior_mul_0.csv"
]

BASELINE_0_UTILITY = [
    "reasoning/power-seeking/gemma/google_gemma-3-1b-it/reasoning_utility_mul_0.csv",
    "reasoning/power-seeking/mistral/mistralai_Mistral-7B-Instruct-v0.3/reasoning_utility_mul_0.csv",
    "reasoning/power-seeking/llama/meta-llama_Llama-3.1-8B-Instruct/reasoning_utility_mul_0.csv"
]

# Regex pattern to extract the type (behavior/utility) and the multiplier
file_pattern = re.compile(r"reasoning_(behavior|utility)_mul_([-\d\.]+)\.csv")

def load_baseline(path):
    """Helper to load the baseline 0 multiplier file."""
    if path and os.path.exists(path):
        df = pd.read_csv(path)
        df['Multiplier'] = 0.0
        return df
    return None

def process_model_family(base_path, base_0_beh_path, base_0_uti_path):
    if not os.path.exists(base_path):
        print(f"\nError: Directory '{base_path}' not found. Skipping...")
        return

    global_bad_ids = set()
    model_data = {}

    print(f"\n========== Processing Model Family: {base_path} ==========")
    print("--- PASS 1: Scanning baseline '0' files and all directories ---")
    
    # 1. Load Baseline 0 data and find its bad IDs
    base_0_beh_df = load_baseline(base_0_beh_path)
    if base_0_beh_df is not None:
        global_bad_ids.update(base_0_beh_df[base_0_beh_df['score'] == 0][ID_COLUMN].unique())
        print(f"Loaded baseline 0 behavior from {base_0_beh_path}")
        
    base_0_uti_df = load_baseline(base_0_uti_path)
    if base_0_uti_df is not None:
        global_bad_ids.update(base_0_uti_df[base_0_uti_df['score'] == 0][ID_COLUMN].unique())
        print(f"Loaded baseline 0 utility from {base_0_uti_path}")

    # 2. Iterate through all items in the base directory
    for subdir_name in os.listdir(base_path):
        subdir_path = os.path.join(base_path, subdir_name)
        
        if not os.path.isdir(subdir_path):
            continue
            
        model_data[subdir_name] = {'behavior': [], 'utility': [], 'path': subdir_path}
        
        for filename in os.listdir(subdir_path):
            match = file_pattern.match(filename)
            if match:
                csv_type = match.group(1)
                multiplier = float(match.group(2))
                file_path = os.path.join(subdir_path, filename)
                
                try:
                    df = pd.read_csv(file_path)
                    df['Multiplier'] = multiplier
                    
                    # Find bad IDs and add to the global set for this model family
                    bad_ids = df[df['score'] == 0][ID_COLUMN].unique()
                    global_bad_ids.update(bad_ids)
                    
                    if csv_type == 'behavior':
                        model_data[subdir_name]['behavior'].append((multiplier, df))
                    elif csv_type == 'utility':
                        model_data[subdir_name]['utility'].append((multiplier, df))
                        
                except Exception as e:
                    print(f"  Error reading {filename}: {e}")

    print(f"\nFound {len(global_bad_ids)} globally unique '{ID_COLUMN}'s to drop across all '{base_path}' directories.")
    print("--- PASS 2: Filtering and saving aggregated files ---")

    # 3. Filter and save the data for each model
    for subdir_name, data in model_data.items():
        subdir_path = data['path']
        print(f"\nProcessing directory: {subdir_name}")
        
        # Process Behavior
        existing_multipliers_b = [item[0] for item in data['behavior']]
        if base_0_beh_df is not None and 0.0 not in existing_multipliers_b:
            data['behavior'].append((0.0, base_0_beh_df.copy())) # Append the 0 baseline if not natively present
            
        if data['behavior']:
            data['behavior'].sort(key=lambda x: x[0]) # Re-sort to put 0.0 in the correct sequential order
            behavior_agg = pd.concat([item[1] for item in data['behavior']], ignore_index=True)
            
            original_len = len(behavior_agg)
            behavior_agg = behavior_agg[~behavior_agg[ID_COLUMN].isin(global_bad_ids)]
            
            output_path = os.path.join(subdir_path, f"{subdir_name}_behavior_agg.csv")
            behavior_agg.to_csv(output_path, index=False)
            print(f"  Saved behavior_agg: {original_len} -> {len(behavior_agg)} rows (includes mul 0.0)")

        # Process Utility
        existing_multipliers_u = [item[0] for item in data['utility']]
        if base_0_uti_df is not None and 0.0 not in existing_multipliers_u:
            data['utility'].append((0.0, base_0_uti_df.copy())) # Append the 0 baseline if not natively present
            
        if data['utility']:
            data['utility'].sort(key=lambda x: x[0]) # Re-sort to put 0.0 in the correct sequential order
            utility_agg = pd.concat([item[1] for item in data['utility']], ignore_index=True)
            
            original_len = len(utility_agg)
            utility_agg = utility_agg[~utility_agg[ID_COLUMN].isin(global_bad_ids)]
            
            output_path = os.path.join(subdir_path, f"{subdir_name}_utility_agg.csv")
            utility_agg.to_csv(output_path, index=False)
            print(f"  Saved utility_agg:  {original_len} -> {len(utility_agg)} rows (includes mul 0.0)")

if __name__ == "__main__":
    # Zip the three lists together so they iterate in matching pairs
    for b_dir, b_0_beh, b_0_uti in zip(base_dirs, BASELINE_0_BEHAVIOR, BASELINE_0_UTILITY):
        process_model_family(b_dir, b_0_beh, b_0_uti)