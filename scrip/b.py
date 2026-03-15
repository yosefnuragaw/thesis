import os

# Define the base directories where the generated files are located
base_dirs = [
    "reasoning/power-seeking/gemma", 
    "reasoning/power-seeking/mistral", 
    "reasoning/power-seeking/llama"
]

def delete_generated_files(base_path):
    if not os.path.exists(base_path):
        print(f"Directory '{base_path}' not found. Skipping...")
        return 0

    deleted_count = 0
    
    # Walk through all subdirectories and files
    for root, dirs, files in os.walk(base_path):
        for filename in files:
            # Check if the file is one of the generated aggregate files
            if filename.endswith("_behavior_agg.csv") or filename.endswith("_utility_agg.csv"):
                file_path = os.path.join(root, filename)
                try:
                    os.remove(file_path)
                    print(f"  Deleted: {file_path}")
                    deleted_count += 1
                except Exception as e:
                    print(f"  Error deleting {file_path}: {e}")
                    
    return deleted_count

if __name__ == "__main__":
    total_deleted = 0
    
    for b_dir in base_dirs:
        print(f"\nScanning '{b_dir}' for generated files...")
        total_deleted += delete_generated_files(b_dir)
        
    print(f"\nCleanup complete. Total aggregated files deleted: {total_deleted}")