from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    pipeline,
    HfArgumentParser
)
from abc import abstractmethod
from typing import List, Optional
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import os
import numpy as np
from datasets import load_dataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap

from models.dataset import MultipleOptionDataset
from models.model import BlockWrapper
from models.prompts import SYSTEM_PROMPT
from utils import set_seed

class PatcherEngine:
    @abstractmethod
    def compute_matrix_sweep(self, multipliers: List[float]):
        pass
    
    @abstractmethod
    def compute_weight(self):
        pass

    @abstractmethod
    def __init_model(self)->tuple[AutoModelForCausalLM, AutoTokenizer]:
        pass

class RAPR(PatcherEngine):
    def __init__(self, model_name:str, vec_dir:str, layers:List[int], eval_epoch:int, loader: DataLoader, verbose:bool = False):
        self.model_name = model_name
        self.vec_dir = vec_dir
        self.layers = layers
        self.eval_epoch = eval_epoch
        self.loader = loader
        self.verbose = verbose

    def compute_matrix_sweep(self, multipliers: List[float]): 
        """
        Runs the exact compute_matrix logic, but loops over a list of multipliers 
        to generate a dictionary of matrices for the blowout calculation.
        """
        N = len(self.layers) 
        sweep_results = {1: {}, -1: {}}
        
        # Init model once to save massive amounts of time
        model = self.__init_model()
        base = sorted(self.layers, reverse=True)
        
        for m in multipliers:
            stat = {
                1: np.full((N, N), np.nan),
                -1: np.full((N, N), np.nan)
            }
            for direction in [1,-1]:
                if self.verbose:
                    pbar = tqdm(base, desc=f"[Dir: {direction} | Mul: {m}]", ncols=100)
                else:
                    pbar = base
                    
                for idx, _ in enumerate(pbar):
                    current_layers = base[:idx + 1]
                    for layer in self.layers:
                        if isinstance(model.model.layers[layer], BlockWrapper) and layer in current_layers:
                            vec_path = f"{self.vec_dir}/vec_ep{self.eval_epoch}_layer{layer}.pt"
                            if os.path.exists(vec_path):
                                layer_device = next(model.model.layers[layer].parameters()).device
                                steering_vector = torch.load(vec_path, map_location=layer_device)
                                model.model.layers[layer].set_vector(steering_vector)
                                
                                # INJECT THE DYNAMIC MULTIPLIER HERE
                                model.model.layers[layer].set_multiplier(direction * m)
                            else:
                                raise ValueError(f"Vector not found at {vec_path}")
                                
                    for batch in self.loader:
                        for input_ids, attention_mask, d in zip(batch["input_ids"], batch["attention_mask"], batch["d"]):
                            with torch.no_grad():
                                if d == direction:
                                    model(input_ids=input_ids.to(model.device), attention_mask=attention_mask.to(model.device))

                    for layer in range(N):
                        if isinstance(model.model.layers[layer], BlockWrapper) and layer in current_layers:
                            mean, std, max_val, min_val, rel_norm = model.model.layers[layer].get_cosine_statistics()
                            stat[direction][N-idx-1, layer] = mean
                            
            # Save the completed matrices for this multiplier
            sweep_results[1][m] = stat[1]
            sweep_results[-1][m] = stat[-1]

        return sweep_results

    def compute_weight(stat:dict):
        pass

    def __init_model(self):
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            attn_implementation="flash_attention_2", 
            use_cache=True,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        model.warnings_issued = {}
        model.to("cuda" if torch.cuda.is_available() else "cpu")
        for layer in self.layers:
            model.model.layers[layer] = BlockWrapper(
                        model.model.layers[layer], 
                        hidden_dim=model.config.hidden_size, 
                        vec= torch.zeros(model.config.hidden_size, dtype=model.dtype),
                    )
        model.eval()
        return model

def get_prompts(tokenizer, behavior, system_prompt=SYSTEM_PROMPT, generation_prompt:bool = True, k:int = 60, seed:int = 42):
    path = f"./data/{behavior}/train.csv"
    if not os.path.exists(path):
         raise FileNotFoundError(f"Data file not found: {path}")
         
    dataset = load_dataset("csv", data_files=path, split='train')
    
    if k is not None:
        sample_size = min(k, len(dataset))
        dataset = dataset.shuffle(seed=seed).select(range(sample_size))
    
    questions = [] 
    prompts = []  
    labels = []
    
    for row in dataset:
        if row['question'] is not None:  
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": row['question']},
            ]
            full_prompt = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=generation_prompt
            )
            
            questions.append(full_prompt)
            current_options = [row[col] for col in ['matching','not_matching'] if col in row and row[col] is not None]
            prompts.append(current_options)
            labels.append('')

    return {
        'questions': questions,
        'prompts': prompts,
        'labels': labels
    }

def produce_dataloader(behavior: str, tokenizer: AutoTokenizer):
    data = get_prompts(tokenizer = tokenizer, behavior= behavior)

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

def calculate_compounded_blowout_weights(matrices_dict, multipliers_tested):
    """
    Calculates the maximum safe multiplier before Off-Manifold decay
    by looking exclusively at Column 0 (the fully compounded state).
    """
    num_layers = 32
    max_safe_multipliers = np.zeros(num_layers)
    
    for layer in range(num_layers):
        best_distance = float('inf')
        best_m = 0.0
        
        for m in multipliers_tested:
            # Look only at Column 0 (the state where all layers are steered)
            compounded_distance = matrices_dict[m][layer, 0]
            
            if np.isnan(compounded_distance):
                continue
                
            # Did the distance improve (drop)?
            if compounded_distance < best_distance:
                best_distance = compounded_distance
                best_m = m
            else:
                # DECAY DETECTED! Spike in distance means blowout.
                break 
                
        # Fallback in case first multiplier was already blowing it out
        if best_m == 0.0 and len(multipliers_tested) > 0:
            best_m = multipliers_tested[0]
            
        max_safe_multipliers[layer] = best_m
        
    # Normalize into the [0, 1] Weight Profile
    W = max_safe_multipliers / np.max(max_safe_multipliers)
    return W, max_safe_multipliers

# ---------------------------------------------------------
# Execution Block
# ---------------------------------------------------------
if __name__ == '__main__':
    model_id = "meta-llama/Llama-3.1-8B-Instruct"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    # Assuming produce_dataloader is available in your environment
    loader = produce_dataloader(
        behavior='wealth-seeking',
        tokenizer=tokenizer
    )
    
    engine = RAPR(
        model_name=model_id,
        vec_dir="pretrained_vector/wealth-seeking/llama-3/all",
        layers=list(range(32)),
        eval_epoch=9,
        loader=loader,
        verbose=True
    )

    # 1. Define the multipliers to sweep
    # (Testing 1.0 to 4.0 in 0.5 increments for high precision)
    multipliers_to_test = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    
    print(f"\nStarting Calibration Sweep across Multipliers: {multipliers_to_test}")
    
    # 2. Run the sweep
    sweep_results = engine.compute_matrix_sweep(multipliers_to_test)
    
    # 3. Calculate and print optimal weights for +1 Direction
    print("\n" + "="*40)
    print("ANALYSIS FOR DIRECTION: +1 (UNSAFE)")
    print("="*40)
    matrices_pos = {m: sweep_results[1][m].T for m in multipliers_to_test}
    W_pos, max_m_pos = calculate_compounded_blowout_weights(matrices_pos, multipliers_to_test)
    print(f"Absolute Max Safe Multipliers:\n{max_m_pos}")
    print(f"Normalized Weight Profile (W_pos) [0 to 1]:\n{np.round(W_pos, 3).tolist()}")

    # 4. Calculate and print optimal weights for -1 Direction
    print("\n" + "="*40)
    print("ANALYSIS FOR DIRECTION: -1 (SAFE)")
    print("="*40)
    matrices_neg = {m: sweep_results[-1][m].T for m in multipliers_to_test}
    W_neg, max_m_neg = calculate_compounded_blowout_weights(matrices_neg, multipliers_to_test)
    print(f"Absolute Max Safe Multipliers:\n{max_m_neg}")
    print(f"Normalized Weight Profile (W_neg) [0 to 1]:\n{np.round(W_neg, 3).tolist()}")
    print("\nCalibration Complete. Use these arrays in your final inference script!")