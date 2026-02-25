import torch
import os

vec_path = "/home/yosef/ws/thesis/pretrained_vector/power-seeking_llama-2/vec_ep20_layer15.pt"

if os.path.exists(vec_path):
    # Load the vector
    vec = torch.load(vec_path, map_location='cpu')
    
    print("-" * 30)
    print(f"FILE: {os.path.basename(vec_path)}")
    print(f"DTYPE:  {vec.dtype}")
    print(f"SHAPE:  {vec.shape}")
    print(f"DEVICE: {vec.device}")
    print("-" * 30)

    # Explain the mismatch
    if vec.dtype == torch.float32:
        print("⚠️  CRITICAL: This vector is FLOAT32.")
        print("Llama 2 (Half/float16) will crash when adding this vector")
        print("unless you cast it with .to(torch.float16)")
    elif vec.dtype == torch.float16:
        print("✅ SUCCESS: Vector is FLOAT16 (Half).")
    elif vec.dtype == torch.bfloat16:
        print("💡 INFO: Vector is BFLOAT16.")
else:
    print(f"❌ Error: File not found at {vec_path}")