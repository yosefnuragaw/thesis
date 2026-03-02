from typing import List, Dict, Optional
import torch
from torch.utils.data import Dataset
from transformers import TrainerCallback
import math
try:
    import wandb
    has_wandb = True
except ImportError:
    has_wandb = False


MODEL_TEMPLATE_MAP: Dict[str, str]= {
    'meta-llama/Llama-2-7b-chat-hf': 'llama-2',
    'mistralai/Mistral-7B-Instruct-v0.2': 'mistral',
    'google/gemma-3-1b-it': 'gemma-3',
    'Qwen/Qwen3-8B': 'qwen-7b'
}

class BlockWrapper(torch.nn.Module):
    def __init__(self, block, hidden_dim, vec: Optional[torch.Tensor] = None, buffer: bool = False):
        super().__init__()
        self.multiplier = 1.0
        self.block = block

        try:
            ref_param = next(block.parameters())
            self.init_dtype = ref_param.dtype
        except StopIteration:
            self.init_dtype = torch.float32
            
        if vec is not None:
            self.vec = vec.to(self.init_dtype)
        else:
            self.vec = torch.nn.Parameter(torch.zeros(hidden_dim, dtype= self.init_dtype))

        self.buffer = buffer
        self.buffer_space = []

    def forward(self, *args, **kwargs):
        output = self.block(*args, **kwargs)

        if isinstance(output, tuple):
            self.buffer_space.append(output[0].detach().mean(dim=1).cpu())
            modified_hidden = output[0] + (self.multiplier * self.vec.to(output[0].device))
            output = (modified_hidden,) + output[1:]
            
        elif isinstance(output, torch.Tensor):
            self.buffer_space.append(output.detach().mean(dim=1).cpu())
            output = output + (self.multiplier * self.vec.to(output.device))
        
        return output

    def set_multiplier(self, multiplier):
        self.multiplier = multiplier

    def set_vector(self, vec):
        self.vec = vec.to(self.init_dtype)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.block, name)
    
    def save(self, filepath:str="output_buffer.pt"):
        if not self.buffer_space:
            print("Empty buffer")
            return
        try:
            torch.save(self.buffer_space, filepath)
            print(f"Success saving {len(self.buffer_space)} to {filepath}")
        except RuntimeError:
            print(f"Failed saving {len(self.buffer_space)} to {filepath}")

    def clear_buffer(self):
        self.buffer_space = []


class QuantileSchedulerCallback(TrainerCallback):
    def __init__(self, start_val, schedule_type="linear"):
        self.start_val = start_val
        self.end_val = 1-1e-8
        self.schedule_type = schedule_type

    def on_step_begin(self, args, state, control, model, **kwargs):
        if state.max_steps == 0:
            return
            
        progress = min(1.0, state.global_step / state.max_steps)
        
        if self.schedule_type == "linear":
            current_val = self.start_val + progress * (self.end_val - self.start_val)
        elif self.schedule_type == "cosine":
            current_val = self.end_val + 0.5 * (self.start_val - self.end_val) * (1 + math.cos(math.pi * progress))
            
        if hasattr(model, "quantile_threshold"):
            model.quantile_threshold = current_val
            
        # Log to wandb
        if has_wandb and wandb.run is not None:
            if state.global_step > 0 and state.global_step % args.logging_steps == 0:
                wandb.log(
                    {"custom/quantile_threshold": current_val}, 
                    step=state.global_step
                )