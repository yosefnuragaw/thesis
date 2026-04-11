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
                    for input_ids, attention_mask,d in zip(batch["input_ids"], batch["attention_mask"],batch["d"]):
                        with torch.no_grad():
                            if d == direction:
                                model(input_ids=input_ids.to(model.device), attention_mask=attention_mask.to(model.device))

                for layer in range(N):
                    if isinstance(model.model.layers[layer], BlockWrapper) and layer in current_layers:
                        mean, std, max_val, min_val, rel_norm = model.model.layers[layer].get_cosine_statistics()
                        stat[direction][N-idx-1, layer] = mean

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
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import torch
from transformers import AutoTokenizer

# (Assuming you have these imported from your project context)
# from your_module import produce_dataloader, RAPR

def _draw_heatmap(ax, fig, rgba: np.ndarray, norm, cmap, N: int):
    """
    Display the heatmap with:
      x = Steered Layer (32 on left → 1 on right)
      y = Layer Index   (0 at bottom → 31 at top, displayed 31 top → 0 bottom)
    Mask (white) is top-right; data is bottom-left.
    """
    # Transpose so (x=steered, y=layer_index), then flip both axes for orientation
    display = rgba.transpose(1, 0, 2)[:, ::-1, :]
    ax.imshow(display, aspect="auto", origin="lower", interpolation="nearest")

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.04)
    cbar.set_label("Cosine Distance (Mean)", fontsize=10)
    cbar.ax.yaxis.set_tick_params(color="black", labelsize=8)
    cbar.outline.set_edgecolor("black")
    cbar.outline.set_linewidth(1.0)

    ticks = list(range(N))

    # X ticks: steered layer 32→1 (left to right)
    x_labels = [i for i in range(N)]
    ax.set_xticks(ticks)
    ax.set_xticklabels(x_labels, fontsize=7)

    # Y ticks: layer index 31→0 (top to bottom, origin=lower so reversed)
    y_labels = [str(N - 1 - i) for i in range(N)]
    ax.set_yticks(ticks)
    ax.set_yticklabels(y_labels, fontsize=7)

    ax.tick_params(axis="both", which="both", direction="in", length=3)
    ax.set_xlabel("Layer Index  (0 → 31)", fontsize=11, labelpad=6)
    ax.set_ylabel("Layer Index  (31 → 0)", fontsize=11, labelpad=6)
    ax.set_title("Cosine-Distance Heatmap", fontsize=13, pad=8, fontweight="bold")

    # Faint grid every 4
    for v in range(0, N, 4):
        ax.axvline(v - 0.5, color="black", lw=0.4, alpha=0.15)
        ax.axhline(v - 0.5, color="black", lw=0.4, alpha=0.15)

    ax.text(0.99, 0.99, "white = inactive",
            transform=ax.transAxes, fontsize=7.5, color="#555",
            ha="right", va="top")


def build_heatmap_rgba(
    heat: np.ndarray,
    active: np.ndarray,
    cmap, # Fixed variable name from 'stat' to 'cmap' to match usage
) -> tuple:
    """Return (rgba, norm) with inactive cells painted white."""
    vmin = float(np.nanmin(heat[active]))
    vmax = 1.0 # Or use np.nanmax(heat[active]) if you want dynamic max
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    rgba = cmap(norm(heat))
    rgba[~active] = [1.0, 1.0, 1.0, 1.0] # Paint inactive white
    return rgba, norm


def min_max_normalize(data):
    d_min, d_max = np.nanmin(data), np.nanmax(data)
    if d_max == d_min: return np.zeros_like(data)
    return (data - d_min) / (d_max - d_min)

def pareto(data):
    sorted_vals = np.sort(data)[::-1]
    return (np.cumsum(sorted_vals) / np.nansum(data))

def mask_to_strictly_decreasing(data):
    cleaned = np.array(data, dtype=float).copy()
    if len(cleaned) == 0: return cleaned
    current_min = cleaned[0]
    for i in range(1, len(cleaned)):
        if cleaned[i] >= current_min or np.isnan(cleaned[i]):
            cleaned[i] = 0
        else:
            current_min = cleaned[i]
            cleaned[i] = 1.

    cleaned[0] = 1.
    return cleaned


if __name__ == '__main__':
    model_id = "meta-llama/Llama-3.1-8B-Instruct"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    # Assuming produce_dataloader and RAPR are imported
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

    result = engine.compute_matrix()
    
    final = {
        1: {'pos': None, 'neg': None},
        2: {'pos': None, 'neg': None},
        3: {'pos': None, 'neg': None}
    }
    
    # Iterate over each direction and draw
    for dir in result.keys():
        matrix = result[dir].T 
        
        print(f"\n{'='*10} Direction: {dir} {'='*10}")
        print(f"\n--- Statistics for {dir} ---")
        
        # 1. Math / Stats
        raw_dist = np.nanmean(matrix, axis=0)
        strict_dist = mask_to_strictly_decreasing(raw_dist)
        norm_row_avg = min_max_normalize(raw_dist) 

        key = 'pos' if dir == 1 else 'neg'
        final[3][key] = np.where(strict_dist == 0., norm_row_avg ** 3, norm_row_avg)
        final[2][key] = np.where(strict_dist == 0., norm_row_avg ** 2, norm_row_avg)
        final[1][key] = np.where(strict_dist == 0., norm_row_avg ** 1, norm_row_avg)
        
        # 2. Draw Heatmap for this direction
        # Assume missing/masked values are NaNs
        active_mask = ~np.isnan(matrix) 
        
        # You can change the colormap here (e.g. 'viridis', 'plasma', 'coolwarm')
        cmap = plt.get_cmap("viridis") 
        rgba, norm = build_heatmap_rgba(matrix, active_mask, cmap)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        N = matrix.shape[0] # Should be 32 for Llama-3 8B
        
        _draw_heatmap(ax, fig, rgba, norm, cmap, N)
        
        # Optionally overwrite the title to include the specific direction
        ax.set_title(f"Cosine-Distance Heatmap (Direction: {dir})", fontsize=13, pad=8, fontweight="bold")
        
        plt.tight_layout()
        plt.savefig(f"heatmap_direction_{dir}.png", dpi=300)
        plt.show() # Remove or comment this out if you are running on a headless server without a GUI

    # Print final calculated weights
    print("\n--- Final Weights ---")
    for key in final.keys():
        pos = final[key]['pos']
        neg = final[key]['neg']
        print(f"Weight Level: {key}")
        if pos is not None:
            print(f'pos_weight: {pos.tolist()}')
        if neg is not None:
            print(f'neg_weight: {neg.tolist()}')