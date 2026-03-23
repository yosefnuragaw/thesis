import wandb
import os

# Your API Key and login
WANDB_API_KEY = "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"
wandb.login(key=WANDB_API_KEY)
os.environ["WANDB_API_KEY"] = WANDB_API_KEY


def download_best_bipo_step():
    api = wandb.Api()
    
    entity = "yosefnuragaw"
    project = "Final-Power-Seeking"
    run_id = "k1wjyqco" 
    
    print("Fetching run history (Max 10,000 rows)...")
    run = api.run(f"{entity}/{project}/{run_id}")
    
    # 1. Fetch exactly the columns we need, up to 10,000 rows maximum
    history = run.history(
        samples=10000, 
        keys=["_step", "eval/test_dataset_sub_loss", "eval/test_dataset_add_loss"]
    )
    
    # 2. Clean the data: Drop missing values and duplicates
    df = history.dropna(subset=["eval/test_dataset_sub_loss", "eval/test_dataset_add_loss"]).copy()
    df = df.drop_duplicates(subset=["_step"], keep="last")
    df = df[df["_step"] > 0].sort_values(by="_step").reset_index(drop=True)
    
    print(df[['_step', 'eval/test_dataset_sub_loss', 'eval/test_dataset_add_loss']])
    if df.empty:
        print("[ERROR] DataFrame is empty. Check if eval metrics were logged in these 10k rows.")
        return

    # 3. Find the sweet spot using Minimax
    df['worst_loss'] = df[['eval/test_dataset_sub_loss', 'eval/test_dataset_add_loss']].max(axis=1)
    best_idx = df['worst_loss'].idxmin()
    best_row = df.loc[best_idx]
    best_step = int(best_row['_step'])
    
    # 4. Calculate the true evaluation interval (delta_step)
    if len(df) >= 2:
        delta_step = int(df["_step"].iloc[1] - df["_step"].iloc[0])
    else:
        delta_step = int(df["_step"].iloc[0])
    
    # 5. Mathematically determine the version
    calculated_version = (best_step // delta_step)
    best_version_str = f"v{calculated_version}"
    
    print(f"\n--- Sweet Spot Found! ---")
    print(f"Global Step: {best_step}")
    print(f"Eval Interval: Every {delta_step} steps")
    print(f"Target Artifact: {best_version_str}")
    print(f"Worst Loss:  {best_row['worst_loss']:.4f}")
    print(f"Sub Loss:    {best_row['eval/test_dataset_sub_loss']:.4f}")
    print(f"Add Loss:    {best_row['eval/test_dataset_add_loss']:.4f}")
    print(f"-------------------------\n")

    # 6. Download the artifacts
    base_name = f"power-seeking-llama3-1-8b-all-2-{run_id}_steering-vec-layer"
    layers = range(32)
    
    print(f"Starting download of {len(layers)} artifacts...")
    for layer in layers:
        artifact_identifier = f"{entity}/{project}/{base_name}{layer}:{best_version_str}"
        try:
            artifact = api.artifact(artifact_identifier)
            save_path = f"./"
            artifact.download(root=save_path)
            print(f"Successfully downloaded Layer {layer} ({best_version_str})")
        except Exception as e:
            # We pass silently because not all layers might be saved
            print(f"Error downloading layer {layer}: {e}")
            pass

if __name__ == "__main__":
    download_best_bipo_step()