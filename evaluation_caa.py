from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    pipeline,
    HfArgumentParser
)
from dataclasses import dataclass, field
import torch
import os
from typing import Any, List, Optional
import gc
from tqdm import tqdm
from torch.utils.data import DataLoader

from models.dataset import SteerbenchDataset
from models.model import CAABlockWrapper
from utils import set_seed, load_dataset,batch_logps
SYSTEM_PROMPT= "You are a helpful, honest and concise assistant."

def get_sterbench_data(tokenizer, behavior, system_prompt=SYSTEM_PROMPT, generation_prompt:bool = True):
    path = f"./data/{behavior}/test_infer.csv"
    if not os.path.exists(path):
         raise FileNotFoundError(f"Data file not found: {path}")
         
    dataset = load_dataset("csv", data_files=path, split='train')
    
    questions = [] 
    prompts = []  
    labels = []    
    communities=[]
    for row in dataset:
        if row['question'] is not None:  
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": row['question']},
            ]
            full_prompt = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=generation_prompt
            )
            
            questions.append(full_prompt)
            
            current_options = [row[col] for col in ['A','B','C','D'] if col in row and row[col] is not None]
            prompts.append(current_options)
            labels.append(row['label'])
            communities.append(row['communities'])

    return{
        'questions':questions,
        'prompts':prompts,
        'labels':labels,
        'communities':communities,
    }
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
        attn_implementation="sdpa",
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
                space.append([
                    torch.zeros(model.config.hidden_size, dtype=model.dtype), 
                    torch.zeros(model.config.hidden_size, dtype=model.dtype)
                ])

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
    data = get_sterbench_data(tokenizer=tokenizer, behavior=behavior)

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

def eval_accuracy(model, loader: DataLoader, multiplier: float, layers: List[int], vecs):
    OPT = ['A', 'B', 'C','D']
    correct = 0
    total = 0
  
    for batch in loader:
        label = batch["label"][0]
        q_len = batch["question_length"]
        
        comm = batch['communities'][0]
        vec_direc = 0 if comm == 'r/A' else 1

        for layer in layers:
            if isinstance(model.model.layers[layer], CAABlockWrapper):
                model.model.layers[layer].set_vec(vecs[layer][vec_direc])
                model.model.layers[layer].set_multiplier(multiplier)

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
        
        total += 1
        if pred == label:
            correct += 1
    
    acc = correct / total 
    print(f"[Accuracy:] {acc}")
    return acc
    
# === EXECUTION BLOCK FOR JUPYTER ===
CONFIG_PATH = ["/kaggle/input/notebooks/diansss/git-loader/thesis/configs/caa/environment-climateskeptics/llama3_1-8b/environment-climateskeptics-all-sequence.yaml",
              "/kaggle/input/notebooks/diansss/git-loader/thesis/configs/caa/environment-climateskeptics/llama3_1-8b/environment-climateskeptics-all.yaml",
              "/kaggle/input/notebooks/diansss/git-loader/thesis/configs/caa/simpleliving-unethical/llama3_1-8b/simpleliving-unethical-all-sequence.yaml",
              "/kaggle/input/notebooks/diansss/git-loader/thesis/configs/caa/simpleliving-unethical/llama3_1-8b/simpleliving-unethical-all.yaml",
              "/kaggle/input/notebooks/diansss/git-loader/thesis/configs/caa/liberal-conservatives/llama3_1-8b/liberal-conservatives-all-sequence.yaml",
              "/kaggle/input/notebooks/diansss/git-loader/thesis/configs/caa/liberal-conservatives/llama3_1-8b/liberal-conservatives-all.yaml"]
IS_BASELINE = False  # Set to True if you want to run the baseline evaluation

if __name__ == "__main__":
    set_seed(seed=42)

    for conf in CONFIG_PATH:
        # Parse the YAML configuration directly
        print(conf)
        hf_parser = HfArgumentParser(ScriptArguments)
        if conf.endswith(".yaml"):
            script_args = hf_parser.parse_yaml_file(yaml_file=conf, allow_extra_keys=True)[0]
        elif conf.endswith(".json"):
            script_args = hf_parser.parse_json_file(json_file=conf, allow_extra_keys=True)[0]
        else:
            raise ValueError("Config file must be .yaml or .json")
        
        print(f"Loading model: {script_args.model_name_or_path}...")
        model, tokenizer, vecs = init_model(
            model_name=script_args.model_name_or_path,
            vec_dir=script_args.vec_dir,
            layers=script_args.layer,
            apply_type=script_args.apply_type,
            total_layer=script_args.total_layer,
            baseline=IS_BASELINE,
            treshold=script_args.treshold
        )
    
        print("Generating dataloader...")
        eval_loader = produce_dataloader(
            behavior=script_args.behavior,
            tokenizer=tokenizer
        )
        
        print("Starting evaluation...")
        for mul in [1]:            
            accuracy = eval_accuracy(
                model=model,
                loader=eval_loader,
                multiplier=mul,
                layers=script_args.layer, 
                vecs=vecs,
            )