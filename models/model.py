from typing import List, Dict, Optional
import torch
from torch.utils.data import Dataset
import math


MODEL_TEMPLATE_MAP: Dict[str, str]= {
    'meta-llama/Llama-2-7b-chat-hf': 'llama-2',
    'mistralai/Mistral-7B-Instruct-v0.2': 'mistral',
    'google/gemma-3-1b-it': 'gemma-3',
    'Qwen/Qwen3-8B': 'qwen-7b'
}

class MaskGate(torch.nn.Module):
    def __init__(self, hidden_dim: int, dtype: torch.dtype = torch.float32, function: str = "sigmoid"):
        super().__init__()

        func_map = {
            "sigmoid": torch.nn.Sigmoid(),
            "tanh": torch.nn.Tanh(),
            "softplus": torch.nn.Softplus()
        }
        
        if function not in func_map:
            raise ValueError(f"Function {function} not supported. Choose from {list(func_map.keys())}")
        
        self.gate = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, 512, dtype=dtype),
            torch.nn.GELU(),
            torch.nn.Linear(512, hidden_dim, dtype=dtype),
            func_map[function]
        )

    def forward(self, x):
        return self.gate(x)


class BlockWrapper(torch.nn.Module):
    def __init__(self, block, hidden_dim, vec: Optional[torch.Tensor] = None, buffer: bool = False, gate_function:Optional[str] = None, skip:Optional[str] = None, k1:float = 0.):
        super().__init__()
        self.multiplier = 1.0
        self.block = block
        self.k1 = k1
        try:
            ref_param = next(block.parameters())
            self.init_dtype = ref_param.dtype
        except StopIteration:
            self.init_dtype = torch.float32
            
        if vec is not None:
            self.vec = vec.to(self.init_dtype)
        else:
            self.vec = torch.nn.Parameter(torch.zeros(hidden_dim, dtype= self.init_dtype))


        if gate_function is not None:
            self.gate_mask= MaskGate(hidden_dim = hidden_dim, dtype=self.init_dtype, function=gate_function)
            
        else:
            self.gate_mask = None

        self.skip =skip

        self.buffer = buffer
        self.buffer_space = []

    def forward(self, hidden_states, *args, **kwargs):
        output = self.block(hidden_states, *args, **kwargs)

        with torch.no_grad():
            out_tensor = output[0] if isinstance(output, tuple) else output
            avg_hidden = hidden_states.detach().mean(dim=1)
            avg_output = out_tensor.detach().mean(dim=1)
            cos_sim = torch.nn.functional.cosine_similarity(avg_hidden, avg_output, dim=-1)
            cos_sim = torch.clamp(cos_sim, min=-1.0, max=1.0) 
            cos_dis = 1 - cos_sim
            cos_dis = 1-cos_sim
            
            exp_sim = torch.exp(cos_sim)
            abs_exp_sim = torch.exp(cos_sim.abs())
            linear_abs_sim =1 + self.k1*(2*cos_sim.abs()-1)
            linear_abs_sim_sens =1 + self.k1*(1-cos_sim.abs()*2)
            lass = 1 + 0.5 (1-cos_sim.abs() *2)
            ldss = 1 + 0.5 (cos_dis - 0.05)
            linear_dis_sim_sens =1 + self.k1*((1-cos_sim)*2)
            print(f'{self.block.__class__.__name__} cosine_distance: {cos_dis.mean().item()} | Scale lass {lass.mean().item()} | Scale ldss {ldss.mean().item()}')
            
        mask = self.multiplier 
        if self.gate_mask:
            mask = mask * self.gate_mask(hidden_states)
        
        if self.skip:
            if self.skip == 'similarity': #redundant layer has higher scale
                mask = mask * cos_sim

            elif self.skip == 'distance': # sensitive layer has higher scale
                mask = mask * cos_dis
            
            elif self.skip == 'exp_similarity': #redundant layer has higher scale
                mask = mask * exp_sim

            elif self.skip == 'abs_exp_similarity':
                mask = mask * abs_exp_sim

            elif self.skip == 'linear_abs_similarity': #redundant layer has higher scale
                mask = mask * linear_abs_sim

            elif self.skip == 'linear_abs_similarity_sens':  # sensitive layer has higher scale
                mask = mask * linear_abs_sim_sens

            elif self.skip == 'linear_abs_similarity_sens_plus':  # sensitive layer has higher scale
                mask = mask * linear_abs_sim_sens_plus

        if isinstance(mask, torch.Tensor):
            while mask.dim() < hidden_states.dim():
                mask = mask.unsqueeze(-1)

        if isinstance(output, tuple):
            self.buffer_space.append(output[0].detach().mean(dim=1).cpu())
            modified_hidden = output[0] + (mask * self.vec.to(output[0].device))
            output = (modified_hidden,) + output[1:]
            
        elif isinstance(output, torch.Tensor):
            self.buffer_space.append(output.detach().mean(dim=1).cpu())
            output = output + (mask * self.vec.to(output.device))
        
        return output

    def set_multiplier(self, multiplier):
        self.multiplier = multiplier

    
    def set_gate(self, path):
        if self.gate:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            state_dict = torch.load(path, map_location=device, weights_only=True)
            
            self.gate.load_state_dict(state_dict)
            self.gate.to(device)
            self.gate.eval()
        else:
            raise ValueError("Gate not initialized. Please define the model architecture first.")


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

