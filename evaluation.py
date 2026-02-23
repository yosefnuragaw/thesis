from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    pipeline,
    HfArgumentParser
)
import torch
import os

from models import BlockWrapper
from utils import set_seed

def init_model(
        model_name: str, vec_dir: str, epoch: int, layer: int, multiplier: int
    )->tuple[AutoModelForCausalLM, AutoTokenizer]:

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    model.warnings_issued = {}
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    vec_path = f"{vec_dir}/vec_ep{epoch}_layer{layer}.pt"

    if os.path.exists(vec_path):
        layer_device = next(model.model.layers[layer].parameters()).device
        steering_vector = torch.load(vec_path, map_location=layer_device)
        
        model.model.layers[layer] = BlockWrapper(
            model.model.layers[layer], 
            hidden_dim=model.config.hidden_size, 
            vec=steering_vector
        )
        model.model.layers[layer].set_multiplier(multiplier)

    else:
        raise ValueError(f"Vector not found at {vec_path}")
        
    model.config.use_cache = False
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    return model,tokenizer

