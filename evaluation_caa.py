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
from typing import Any, List, Optional
import gc
from tqdm import tqdm
from torch.utils.data import DataLoader

from models.dataset import SteerbenchDataset
from models.model import CAABlockWrapper
from utils import set_seed, get_sterbench_data, batch_logps

@dataclass
class ScriptArguments:
    id: Optional[str] = field(default="baseline", metadata={"help": "Run id"})
    model_name_or_path: Optional[str] = field(
        default="google/gemma-3-1b-it",
        metadata={"help": "The model checkpoint"}
    )
    behavior: Optional[str] = field(default="power-seeking", metadata={"help": "the behavior"})
    layer: Optional[List[int]] = field(
        default_factory=lambda: list(range(32)),
        metadata={"help": "Layers to apply steering"}
    )
    total_layer: Optional[int] = field(default=32, metadata={"help": "LLM total number of layers"})
    multipliers: Optional[List[float]] = field(
        default_factory=lambda: [2., 1.5, 1., 0.5, -0.5, -1., -1.5, -2.],
        metadata={"help": "Steering multipliers to evaluate"}
    )
    vec_dir: Optional[str] = field(default=None, metadata={"help": "Directory where .pt vectors are saved"})
    answer_dir: Optional[str] = field(default="generation_results", metadata={"help": "Directory where CSVs will be saved"})
    apply_type: Optional[str] = field(default="base", metadata={"help": "layer or sequence"})
    max_new_tokens: Optional[int] = field(default=200, metadata={"help": "Max new generation tokens"})
    temperature: Optional[float] = field(default=0.7, metadata={"help": "LLM generation temperature"})
    treshold: float = field(default=0.25, metadata={"help": "Lower bound sensitivity"})
    prompt: Optional[str] = field(default=None, metadata={"help" : "cosine scaler None | distance | similarity"})

def init_model(
        model_name: str,
        vec_dir: str,
        layers: List[int],
        apply_type: str,
        total_layer: int = 32,
        baseline: bool = False,
        treshold: float = 0.25
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        # attn_implementation="flash_attention_2",
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.warnings_issued = {}
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    space = []
    for layer in range(total_layer):
        if baseline:
            vec = torch.zeros(model.config.hidden_size, dtype=model.dtype)
        else:
            vec_path = os.path.join(vec_dir, f"pos_layer_{layer}.pt")
            neg_path = os.path.join(vec_dir, f"neg_layer_{layer}.pt")
            
            
            if os.path.exists(vec_path) and os.path.exists(neg_path):
                space.append([
                    torch.load(vec_path, map_location="cuda"), 
                    torch.load(neg_path, map_location="cuda")
                ])
            else:
                space.append([torch.zeros(model.config.hidden_size, dtype=model.dtype), torch.zeros(model.config.hidden_size, dtype=model.dtype)])

            model.model.layers[layer] = CAABlockWrapper(
            model.model.layers[layer],
            hidden_dim=model.config.hidden_size,
            vec = torch.zeros(model.config.hidden_size, dtype=model.dtype),
            apply_type=apply_type,
            treshold = treshold
            )
            model.model.layers[layer].extract(False)  

    model.config.use_cache = False
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer, space

def produce_dataloader(behavior: str, tokenizer: AutoTokenizer):
    data = get_sterbench_data(tokenizer = tokenizer, behavior= behavior)

    eval_dataset = SteerbenchDataset(
        tokenizer=tokenizer,
        questions=data['questions'],
        prompts=data['prompts'],
        labels=data['labels'],
        communities=data['communities']
    )
        
    eval_loader = DataLoader(
        dataset=eval_dataset,
        batch_size=1,              
        shuffle=False,          
        num_workers=0            
    ) 
    return eval_loader

def eval_accuracy(
        model, loader: DataLoader, multiplier: float, layers: List[int], vecs
    ):
    OPT = ['A', 'B', 'C','D']
    correct = 0
    total = 0
  
    for batch in loader:
        label = batch["label"][0]
        q_len = batch["question_length"]
        comm = batch['communities']

        vec_direc = 1 if comm == 'r/A' else 'r/B'

        for layer in layers:
            if isinstance(model.model.layers[layer], CAABlockWrapper):
                model.model.layers[layer].set_vec(vecs[layer][vec_direc])
                model.model.layers[layer].set_multiplier(multiplier)

        avg_logp = []
        for input_ids, attention_mask, decoded in zip(batch["input_ids"], batch["attention_mask"], batch['decode']):
            input_ids = input_ids.to(model.device)
            attention_mask = attention_mask.to(model.device)
    
            with torch.no_grad():
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                logps, _ = batch_logps(logits, input_ids)
                
                sliced = logps[0, q_len - 1:]
                avg_logp.append(sliced.mean().item())
            
        pred = OPT[avg_logp.index(max(avg_logp))]
        
        total += 1
        if pred == label:
            correct += 1
    
        
        acc = correct / total 

    print(f"[Accuracy:] {acc}")
    return acc
    

if __name__ == "__main__":
    set_seed(seed=42)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, required=True, help="Path to your YAML config file")
    parser.add_argument("--verbose", "-v", type=bool, required=False, default=True, help="Visualize eval progress")
    parser.add_argument("--baseline", action='store_true')

    args, remaining = parser.parse_known_args()

    hf_parser = HfArgumentParser(ScriptArguments)
    if args.config.endswith(".yaml"):
        script_args = hf_parser.parse_yaml_file(yaml_file=args.config, allow_extra_keys=True)[0]
    elif args.config.endswith(".json"):
        script_args = hf_parser.parse_json_file(json_file=args.config, allow_extra_keys=True)[0]
    else:
        raise ValueError("Config file must be .yaml or .json")
    
    model, tokenizer, vecs = init_model(
        model_name=script_args.model_name_or_path,
        vec_dir=script_args.vec_dir,
        layers=script_args.layer,
        apply_type=script_args.apply_type,
        total_layer=script_args.total_layer,
        baseline=args.baseline,
        treshold = script_args.treshold
    )

    eval_loader = produce_dataloader(
        behavior=script_args.behavior,
        tokenizer=tokenizer
    )
    
    for mul in [1]:            
        accuracy = eval_accuracy(
                model=model,
                loader=eval_loader,
                multiplier=mul,
                layers=script_args.layer, 
                vecs=vecs,
            ) 

    


 
