import argparse
import os
import random
import numpy as np
import copy
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass, field
from typing import Tuple, Dict, List, Optional
import torch
from transformers import HfArgumentParser


import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from utils import set_seed, get_eval_data
from evaluation import init_model, eval_accuracy
from models import BlockWrapper
from dataset import MultipleOptionDataset


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, required=True, help="Path to your YAML config file")
    parser.add_argument("--verbose", "-v", type=bool, required=False, default=True, help="Visualize eval progress")
    args, remaining = parser.parse_known_args()

    hf_parser = HfArgumentParser(ScriptArguments)
    if args.config.endswith(".yaml"):
        script_args = hf_parser.parse_yaml_file(yaml_file=args.config, allow_extra_keys=True)[0]
    elif args.config.endswith(".json"):
        script_args = hf_parser.parse_json_file(json_file=args.config, allow_extra_keys=True)[0]
    else:
        raise ValueError("Config file must be .yaml or .json")

    set_seed(seed=11)
    
    
    print("Loading model to GPU...")
    model, tokenizer = init_model(
        model_name=script_args.model_name_or_path,
        vec_dir=script_args.vec_dir,
        layers=script_args.layer,
        multiplier= 0
    )
    data = get_eval_data(tokenizer = tokenizer, behavior = script_args.behavior)

    eval_dataset = MultipleOptionDataset(
        tokenizer=tokenizer,
        questions=data.questions,
        prompts=data.prompts,
        labels=data.labels,
    )
        
    eval_loader = DataLoader(
        dataset=eval_dataset,
        batch_size=1,              
        shuffle=False,          
        num_workers=0            
    )

    model.eval()

    base_accuracy = eval_accuracy(
                    model=model,
                    loader=eval_loader,
                    multiplier=0,
                    layers=script_args.layer, 
                    epoch=None,
                    vec_dir=script_args.vec_dir, 
                    verbose=args.verbose
        )
    
    print(f"[Config:] {args.config} [Behavior:] {script_args.behavior} | [Positive Accuracy:] {base_accuracy[0]:.4f} [Negative Accuracy:] {base_accuracy[1]:.4f}")
    
    original_layers = torch.nn.ModuleList([copy.deepcopy(layer) for layer in model.model.layers])
    
    for epo in range(20):
        for layer in script_args.layer:
            vec_path = f"{script_args.vec_dir}/vec_ep{epo}_layer{layer}.pt"
            if os.path.exists(vec_path):
                layer_device = next(model.model.layers[layer].parameters()).device
                steering_vector = torch.load(vec_path, map_location=layer_device)
                
                model.model.layers[layer] = BlockWrapper(
                original_layers[layer], 
                hidden_dim=model.config.hidden_size, 
                vec=steering_vector
            )
                
                model.config.use_cache = False

     
        
        for mul in [1.,1.5,2]:
            accuracy = eval_accuracy(
                model=model,
                loader=eval_loader,
                multiplier=mul,
                layers=script_args.layer, 
                epoch=epo,
                vec_dir=script_args.vec_dir, 
                verbose=args.verbose
            )

            if mul == 1. and accuracy.positive < base_accuracy.positive or accuracy.negative < base_accuracy.negative:
                print(f"Epoch {epo} skipped")
                break