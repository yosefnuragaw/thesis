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


from models.dataset import MultipleOptionDataset
from models.model import BlockWrapper
from models.prompts import SYSTEM_PROMPT
from utils import set_seed

class PatcherEngine:
    @abstractmethod
    def compute_matrix(self):
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

    def compute_matrix(self): 
        N = len(self.layers) 
        stat = {
            1: np.full((N, N), np.nan),
            -1: np.full((N, N), np.nan)
        }
        model = self.__init_model()
        base = sorted(self.layers, reverse=True)
        for direction in [1,-1]:
            if self.verbose:
                pbar = tqdm(base, desc=f"[Direction:]{direction}", ncols=100)
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
                            model.model.layers[layer].set_multiplier(direction)
                        else:
                            raise ValueError(f"Vector not found at {vec_path}")
                            
                for batch in self.loader:
                    for input_ids, attention_mask in zip(batch["input_ids"], batch["attention_mask"]):
                        with torch.no_grad():
                            model(input_ids=input_ids.to(model.device), attention_mask=attention_mask.to(model.device))

                for layer in range(N):
                    if isinstance(model.model.layers[layer], BlockWrapper) and layer in current_layers:
                        mean, std, max_val, min_val, rel_norm = model.model.layers[layer].get_cosine_statistics()
                        stat[direction][N-idx-1, layer] = mean
                
                print(current_layers)
                print(stat[direction][N-idx-1, :])

        return stat

    def compute_weight(stat:dict):
        pass

    def __init_model(self)->tuple[AutoModelForCausalLM, AutoTokenizer]:
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
            
            if self.eval_epoch != None and layer in self.layers:
                vec_path = f"{self.vec_dir}/vec_ep{self.eval_epoch}_layer{layer}.pt"
                if os.path.exists(vec_path):
                    layer_device = next(model.model.layers[layer].parameters()).device
                    steering_vector = torch.load(vec_path, map_location=layer_device)
                    model.model.layers[layer].set_vector(steering_vector)
                    model.model.layers[layer].set_multiplier(1)

                else:
                    raise ValueError(f"Vector not found at {vec_path}")
            
        model.eval()
        return model
    
def get_prompts(tokenizer, behavior, system_prompt=SYSTEM_PROMPT, generation_prompt:bool = True, k:int = 30, seed:int = 42):
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

if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    tokenizer.pad_token = tokenizer.eos_token

    loader = produce_dataloader(
        behavior= 'wealth-seeking',
        tokenizer=tokenizer
    )
    engine = RAPR(
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        vec_dir="pretrained_vector/wealth-seeking/llama-3/all",
        layers= list(range(32)),
        eval_epoch=9,
        loader= loader,
        verbose= True
    )

    result = engine.compute_matrix()
    for dir in result.keys():
        print(f'[Direction:] {dir}')
        print(result[dir])