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

from models.dataset import MultipleOptionDataset
from models.model import CAABlockWrapper
from models.prompts import SYSTEM_PROMPT
from utils import set_seed, get_eval_data, batch_logps

@dataclass
class ScriptArguments:
    """
    The arguments for the CAA script, matching the training config structure.
    """
    id: Optional[str] = field(default="baseline", metadata={"help": "Run id"})
    model_name_or_path: Optional[str] = field(
        default="google/gemma-3-1b-it",
        metadata={"help": "The model checkpoint for weights initialization."}
    )
    total_layer: Optional[int] = field(default=200, metadata={"help": "LLM total number of layers"})
    behavior: Optional[str] = field(default="power-seeking", metadata={"help": "the behavior"})
    layer: Optional[List[int]] = field(
        default_factory=lambda: list(range(26)), 
        metadata={"help": "the layer the steering vector extracted from"}
    )

    vec_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Directory where .pt vectors are saved"}
    )

def produce_dataloader(behavior: str, tokenizer: AutoTokenizer):
    data = get_eval_data(tokenizer = tokenizer, behavior= behavior)

    eval_dataset = MultipleOptionDataset(
        tokenizer=tokenizer,
        questions=data['questions'],
        prompts=data['prompts'],
        labels=data['labels'],
    )
        
    eval_loader = DataLoader(
        dataset=eval_dataset,
        batch_size=1,              
        shuffle=False,          
        num_workers=0            
    ) 
    return eval_loader

def init_model(
        model_name: str, apply_type: str, total_layer: int = 26
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

    for layer in range(total_layer):
        model.model.layers[layer] = CAABlockWrapper(
            model.model.layers[layer],
            hidden_dim=model.config.hidden_size,
            vec=torch.zeros(model.config.hidden_size, dtype=model.dtype),
            apply_type=apply_type
        )
        model.model.layers[layer].extract(True)  # extract() takes a bool

    model.config.use_cache = False
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def extract_caa(
        model, loader: DataLoader, multiplier: float, layers: List[int], vec_dir: str, total_layer: int = 26, verbose: bool = False,
    ):
    directions = [1, -1]

    pbar = tqdm(directions, desc="Extracting", ncols=100) if verbose else directions

    for direction in pbar:
        for batch in loader:
            for layer in layers:
                if isinstance(model.model.layers[layer], CAABlockWrapper):
                    model.model.layers[layer].set_multiplier(direction * multiplier)

            for input_ids, attention_mask in zip(batch["input_ids"], batch["attention_mask"]):
                input_ids = input_ids.to(model.device)
                attention_mask = attention_mask.to(model.device)

                with torch.no_grad():
                    _ = model(input_ids=input_ids, attention_mask=attention_mask).logits

    # After both directions are collected, extract and save per layer
    os.makedirs(vec_dir, exist_ok=True)

    for layer in range(total_layer):
        if isinstance(model.model.layers[layer], CAABlockWrapper):
            vec_pos, vec_neg = model.model.layers[layer].extract_vec(clear=True)

            torch.save(vec_pos, os.path.join(vec_dir, f"pos_layer_{layer}.pt"))
            torch.save(vec_neg, os.path.join(vec_dir, f"neg_layer_{layer}.pt"))


if __name__ == "__main__":
    set_seed(seed=42)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, required=True)
    parser.add_argument("--apply_type", type=str, default="layer")
    args, remaining = parser.parse_known_args()

    hf_parser = HfArgumentParser(ScriptArguments)
    if args.config.endswith(".yaml"):
        script_args = hf_parser.parse_yaml_file(yaml_file=args.config, allow_extra_keys=True)[0]
    elif args.config.endswith(".json"):
        script_args = hf_parser.parse_json_file(json_file=args.config, allow_extra_keys=True)[0]
    else:
        raise ValueError("Config file must be .yaml or .json")

    model, tokenizer = init_model(
        model_name=script_args.model_name_or_path,
        apply_type=args.apply_type,
        total_layer=script_args.total_layer,
    )

    eval_loader = produce_dataloader(
        behavior=script_args.behavior,
        tokenizer=tokenizer,
    )

    extract_caa(
        model=model,
        loader=eval_loader,
        multiplier=1.0,
        layers=script_args.layer,
        vec_dir=script_args.vec_dir,
        total_layer=script_args.total_layer,
        verbose=True,
    )
        

