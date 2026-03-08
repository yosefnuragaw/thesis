import wandb
import os

# Your provided API key
WANDB_API_KEY = "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"
wandb.login(key=WANDB_API_KEY)
os.environ["WANDB_API_KEY"] = WANDB_API_KEY

def download_full_bipo_history():
    api = wandb.Api()
    
    entity = "yosefnuragaw"
    project = "BiPO-Mistral-7b-Power-Seeking"

    # The specific base path identified from your terminal success
    base_name = "power-seeking-power-seeking-all-exp-6-kr78ckt8_steering-vec-layer"
    
    layers = range(32)  # 0 to 25
    versions = [19] # v0 to v19

    print(f"Starting download of {len(layers) * len(versions)} artifacts...")

    for v in versions:
        for layer in layers:
            artifact_identifier = f"{entity}/{project}/{base_name}{layer}:v{v}"
            
            try:
                artifact = api.artifact(artifact_identifier)
                # Download to current directory
                artifact.download(root=".")
                
                print(f"Successfully downloaded Layer {layer} Version v{v}")
                        
            except Exception as e:
                # Silently skip if a specific version doesn't exist for a layer
                continue

if __name__ == "__main__":
    download_full_bipo_history()