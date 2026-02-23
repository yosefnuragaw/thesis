from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    pipeline,
    HfArgumentParser
)
from dataclasses import dataclass, field
import argparse
import torch
import os
from typing import List, Optional
import gc
from tqdm import tqdm
from torch.utils.data import DataLoader


from models import BlockWrapper
from utils import set_seed, get_eval_data, batch_logps

@dataclass
class ScriptArguments:
    """
    The arguments for the DPO eval script, matching the training config structure.
    """
    model_name_or_path: Optional[str] = field(
        default="google/gemma-3-1b-it",
        metadata={"help": "The model checkpoint for weights initialization."}
    )
    behavior: Optional[str] = field(default="power-seeking", metadata={"help": "the behavior"})
    layer: Optional[List[int]] = field(
        default_factory=lambda: list(range(26)), 
        metadata={"help": "the layer the steering vector extracted from"}
    )

    vec_dir: Optional[str] = field(
        default="/kaggle/working/BiPO/vector/power-seeking_gemma-3",
        metadata={"help": "Directory where .pt vectors are saved"}
    )
    eval_epoch: Optional[int] = field(default=18, metadata={"help": "Which epoch's vector to load"})
    prompt: Optional[str] = field(default="", metadata={"help": "What prompts for generation eval"})

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

def eval_accuracy(
        model, loader: DataLoader, multiplier: float, layers: List[int], epoch: int|None, verbose: bool = False
    ):
    OPT = ['A', 'B']
    correct = [0,0]
    total = [0,0]
    
    if verbose:
        pbar = tqdm(loader, desc="Evaluating", ncols=100)
    else:
        pbar = loader
    
    for batch in pbar:
        label = batch["label"][0]
        q_len = batch["question_length"]

        for layer in layers:
            if isinstance(model.model.layers[layer], BlockWrapper):
                if label != 'A':
                    model.model.layers[layer].set_multiplier(-multiplier)
                else:
                    model.model.layers[layer].set_multiplier(multiplier)
        
        indx = None
        if label != 'A':
            indx = 0
        else:
            indx = 1

        avg_logp = []
        for input_ids, attention_mask in zip(batch["input_ids"], batch["attention_mask"]):
            input_ids = input_ids.to(model.device)
            attention_mask = attention_mask.to(model.device)
    
            with torch.no_grad():
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                logps, _ = batch_logps(logits, input_ids)
                
                sliced = logps[0, q_len - 1:]
                avg_logp.append(sliced.mean().item())

        pred = OPT[avg_logp.index(max(avg_logp))]
        
        total[indx] += 1
        if pred == label:
            correct[indx] += 1
    
        
        curr_positive = correct[0] / total[0] if total[0] > 0 else 0.0
        curr_negative = correct[1]/ total[1] if total[1] > 0 else 0.0

        if verbose:
            if epoch is not None:
                pbar.set_description(f"[Epoch:] {epoch} [Multiplier:] {multiplier}  [Positive Accuracy:] {curr_positive:.4f} [Negative Accuracy:] {curr_negative:.4f}")
            else:
                pbar.set_description(f"Baseline {multiplier}  [Positive Accuracy:] {curr_positive:.4f} [Negative Accuracy:] {curr_negative:.4f}")

    positive_acc = correct[0] / total[0] if total[0] > 0 else 0.0
    negative_acc = correct[1]/ total[1] if total[1] > 0 else 0.0

    return positive_acc, negative_acc
    


if __name__ == "__main__":
    set_seed(seed=11)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, required=True, help="Path to your YAML config file")
    parser.add_argument("--verbose", "-v", type=bool, required=False, default=True, help="Visualize eval progress")
    parser.add_argument("--task", "-t", type=str, required=False, default="both", help="Visualize eval progress")
    args, remaining = parser.parse_known_args()

    hf_parser = HfArgumentParser(ScriptArguments)
    if args.config.endswith(".yaml"):
        script_args = hf_parser.parse_yaml_file(yaml_file=args.config, allow_extra_keys=True)[0]
    elif args.config.endswith(".json"):
        script_args = hf_parser.parse_json_file(json_file=args.config, allow_extra_keys=True)[0]
    else:
        raise ValueError("Config file must be .yaml or .json")
    

    data = get_eval_data(script_args.behavior)
    model, tokenizer = init_model(
        model_name=script_args.model_name_or_path,
        vec_dir=script_args.vec_dir,
        epoch=script_args.eval_epoch,
        layer=script_args.layer,
        multiplier= 0
    )

