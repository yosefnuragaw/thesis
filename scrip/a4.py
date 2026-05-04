import os
import re
import pandas as pd

# Define the base directories
base_dirs = [
    "caa/reasoning/survival-instinct/gemma", 
    "caa/reasoning/survival-instinct/mistral", 
    "caa/reasoning/survival-instinct/llama"
]

bench = 'data/survival-instinct/test_infer.csv'

# Paths to your baseline "0" multiplier CSVs
BASELINE_0_BEHAVIOR = [
    "caa/reasoning/survival-instinct/gemma/google_gemma-3-1b-it/reasoning_behavior_mul_0.csv",
    "caa/reasoning/survival-instinct/mistral/mistralai_Mistral-7B-Instruct-v0.3/reasoning_behavior_mul_0.csv",
    "caa/reasoning/survival-instinct/llama/meta-llama_Llama-3.1-8B-Instruct/reasoning_behavior_mul_0.csv"
]

BASELINE_0_UTILITY = [
    "caa/reasoning/survival-instinct/gemma/google_gemma-3-1b-it/reasoning_utility_mul_0.csv",
    "caa/reasoning/survival-instinct/mistral/mistralai_Mistral-7B-Instruct-v0.3/reasoning_utility_mul_0.csv",
    "caa/reasoning/survival-instinct/llama/meta-llama_Llama-3.1-8B-Instruct/reasoning_utility_mul_0.csv"
]

# Regex pattern to extract the type (behavior/utility) and the multiplier
file_pattern = re.compile(r"reasoning_(behavior|utility)_mul_([-\d\.]+)\.csv")

def load_baseline(path, questions_series):
    """Helper to load the baseline 0 multiplier file and append questions."""
    if path and os.path.exists(path):
        df = pd.read_csv(path)
        df['Multiplier'] = 0.0
        
        # Attach the question column if lengths match
        if questions_series is not None:
            if len(df) == len(questions_series):
                df['Question'] = questions_series.values
            else:
                print(f"  Warning: Row count mismatch in baseline {path} ({len(df)} vs {len(questions_series)})")
                
        return df
    return None

def process_model_family(base_path, base_0_beh_path, base_0_uti_path, questions_series):
    if not os.path.exists(base_path):
        print(f"\nError: Directory '{base_path}' not found. Skipping...")
        return

    model_data = {}

    print(f"\n========== Processing Model Family: {base_path} ==========")
    
    # 1. Load Baseline 0 data
    base_0_beh_df = load_baseline(base_0_beh_path, questions_series)
    if base_0_beh_df is not None:
        print(f"Loaded baseline 0 behavior from {base_0_beh_path}")
        
    base_0_uti_df = load_baseline(base_0_uti_path, questions_series)
    if base_0_uti_df is not None:
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
                    
                    # Attach the question column if lengths match
                    if questions_series is not None:
                        if len(df) == len(questions_series):
                            df['Question'] = questions_series.values
                        else:
                            print(f"  Warning: Row count mismatch in {filename} ({len(df)} vs {len(questions_series)})")

                    if csv_type == 'behavior':
                        model_data[subdir_name]['behavior'].append((multiplier, df))
                    elif csv_type == 'utility':
                        model_data[subdir_name]['utility'].append((multiplier, df))
                        
                except Exception as e:
                    print(f"  Error reading {filename}: {e}")

    print("--- Merging and saving aggregated files ---")

    # 3. Merge and save the data for each model
    for subdir_name, data in model_data.items():
        subdir_path = data['path']
        print(f"\nProcessing directory: {subdir_name}")
        
        # --- Process Behavior ---
        existing_multipliers_b = [item[0] for item in data['behavior']]
        if base_0_beh_df is not None and 0.0 not in existing_multipliers_b:
            data['behavior'].append((0.0, base_0_beh_df.copy()))
            
        if data['behavior']:
            data['behavior'].sort(key=lambda x: x[0]) 
            
            # Extract dataframes and concatenate directly
            all_dfs = [df for mult, df in data['behavior']]
            behavior_agg = pd.concat(all_dfs, ignore_index=True)
            
            output_path = os.path.join(subdir_path, f"{subdir_name}_behavior_agg.csv")
            behavior_agg.to_csv(output_path, index=False)
            print(f"  Saved behavior_agg: {len(behavior_agg)} total rows (includes mul 0.0)")

        # --- Process Utility ---
        existing_multipliers_u = [item[0] for item in data['utility']]
        if base_0_uti_df is not None and 0.0 not in existing_multipliers_u:
            data['utility'].append((0.0, base_0_uti_df.copy()))
            
        if data['utility']:
            data['utility'].sort(key=lambda x: x[0])
            
            # Extract dataframes and concatenate directly
            all_dfs = [df for mult, df in data['utility']]
            utility_agg = pd.concat(all_dfs, ignore_index=True)
            
            output_path = os.path.join(subdir_path, f"{subdir_name}_utility_agg.csv")
            utility_agg.to_csv(output_path, index=False)
            print(f"  Saved utility_agg:  {len(utility_agg)} total rows (includes mul 0.0)")

if __name__ == "__main__":
    # Load the benchmark questions once
    questions_series = None
    if os.path.exists(bench):
        bench_df = pd.read_csv(bench)
        if 'question' in bench_df.columns:
            questions_series = bench_df['question']
        else:
            print(f"Error: 'question' column not found in {bench}")
    else:
        print(f"Error: Benchmark file not found at {bench}")

    # Process all directories and pass the questions down
    for b_dir, b_0_beh, b_0_uti in zip(base_dirs, BASELINE_0_BEHAVIOR, BASELINE_0_UTILITY):
        process_model_family(b_dir, b_0_beh, b_0_uti, questions_series)