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
from models import MultipleOptionDataset


from models import BlockWrapper, SYSTEM_PROMPT
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
        model_name: str, vec_dir: str, epoch: int, layers: List[int], multiplier: int
    )->tuple[AutoModelForCausalLM, AutoTokenizer]:

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    model.warnings_issued = {}
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    for layer in layers:
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

def produce_dataloader(behavior: str, tokenizer: AutoTokenizer):
    data = get_eval_data(behavior)

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
    
    return eval_loader

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
    
def eval_generation(
    model,
    tokenizer,
    layers: list,
    multipliers: list,
    messages: list,
    device: int = 0,
    max_new_tokens: int = 64,
    temperature: float = 0.9,
):
    """
    Run generation for different steering multipliers on selected layers.
    """

    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device if torch.cuda.is_available() else -1,
    )

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    results = {}
    for mult in multipliers:
        for layer in layers:
            model.model.layers[layer].set_multiplier(mult)

        output = generator(
            prompt,
            max_new_tokens=max_new_tokens,
            min_new_tokens=16,
            do_sample=True,
            temperature=temperature,   
            repetition_penalty=1.2,
            generation_config=None,   
        )[0]["generated_text"]

        if "model" in output:
            trimmed = output[output.find("model") + 5:].strip()
        else:
            trimmed = output

        print(f"[Multiplier {mult}:] {trimmed}\n")
        results[mult] = trimmed

    return results

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
    
    model, tokenizer = init_model(
        model_name=script_args.model_name_or_path,
        vec_dir=script_args.vec_dir,
        epoch=script_args.eval_epoch,
        layers=script_args.layer,
        multiplier= 0
    )

    eval_loader = produce_dataloader(
        behavior=script_args.behavior,
        tokenizer=tokenizer
    )

    if args.task != "generation":
        for mul in [0,1.,1.5,2]:       
            accuracy = eval_accuracy(
                model=model,
                loader=eval_loader,
                multiplier=mul,
                layers=script_args.layer, 
                epoch=script_args.eval_epoch,
                verbose=args.verbose
            ) 

    if args.task != "accuracy":
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": script_args.prompt},
        ]
        res = eval_generation(
            model=model,
            tokenizer=tokenizer,
            layers=script_args.layer,
            multipliers= [-2,-1.5,-1,0,1,1.5,2],
            messages=messages,
        )
    



