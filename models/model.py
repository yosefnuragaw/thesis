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
        self.func = func_map[function]
        self.h = torch.nn.Parameter(torch.tensor([0.0], dtype=dtype))
    def forward(self):
        return self.func(self.h)


class MaskGate2(torch.nn.Module):
    def __init__(self, hidden_dim: int, dtype: torch.dtype = torch.float32, function: str = "sigmoid"):
        super().__init__()

        func_map = {
            "relu":      lambda x: torch.nn.functional.relu(x),
            "softplus":  lambda x: torch.nn.functional.softplus(x),
            "softplus1": lambda x: torch.nn.functional.softplus(x - 1),  # starts ~0, smoother than relu
            "sigmoid":  lambda x: torch.nn.functional.sigmoid(x),
        }
        
        if function not in func_map:
            raise ValueError(f"Function {function} not supported. Choose from {list(func_map.keys())}")
        self.func = func_map[function]
        self.h = torch.nn.Parameter(torch.tensor([0.0], dtype=dtype))

    def forward(self):
        return self.func(self.h)



class BlockWrapper(torch.nn.Module):
    def __init__(self, block, hidden_dim, vec: Optional[torch.Tensor] = None, buffer: bool = False, gate_function:Optional[str] = None, skip:Optional[str] = None, k1:float = 1.):
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
            if skip == 'llds':
                self.gate_mask= MaskGate(hidden_dim = hidden_dim, dtype=self.init_dtype, function=gate_function)
            elif skip == 'adap':
                self.gate_mask = MaskGate2(hidden_dim = hidden_dim, dtype=self.init_dtype, function=gate_function)
            
        else:
            self.gate_mask = None

        self.skip =skip
        self.gen_step = 0 
        self.buffer = buffer
        self.buffer_space = []
        self.cosine_space = []
        self.rel_norm_space = []
    
    def forward(self, hidden_states, *args, **kwargs):
        output = self.block(hidden_states, *args, **kwargs)
        out_tensor = output[0] if isinstance(output, tuple) else output
        avg_output = out_tensor.detach().mean(dim=1)
        current_vec = (self.multiplier * self.vec).to(avg_output.device)
        
        cos_sim = torch.nn.functional.cosine_similarity(avg_output, current_vec, dim=-1)
        cos_sim_c = torch.clamp(cos_sim, min=-1.0, max=1.0) 
        cos_dis = 1 - cos_sim_c

        # 2. Tracking Block
        with torch.no_grad():
            self.cosine_space.append(cos_dis.cpu()) # Move to CPU to save VRAM if only for logging

            vec_norm = torch.norm(current_vec, p=2, dim=-1)
            output_norm = torch.norm(avg_output, p=2, dim=-1)
            output_norm_safe = torch.clamp(output_norm, min=1e-8)
            rel_norm = vec_norm / output_norm_safe
            self.rel_norm_space.append(rel_norm.cpu())
            
        mask = self.multiplier * self.k1 

        if self.skip == 'adap' and self.gate_mask is not None:        
            mask = mask * torch.nn.functional.relu(self.gate_mask().to(output[0].device) - cos_sim_c.to(output[0].device))

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
        if self.gate_mask:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            state_dict = torch.load(path, map_location=device, weights_only=True)
            
            self.gate_mask.load_state_dict(state_dict)
            self.gate_mask.to(device)
            self.gate_mask.eval()
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

    def get_cosine_statistics(self):
        if not self.cosine_space:
            # Return 4 zeros to match the new tuple signature
            return 0.0, 0.0, 0.0, 0.0
            
        all_distances = torch.cat(self.cosine_space, dim=0)
        all_norm = torch.cat(self.rel_norm_space, dim=0)
        
        mean_val = all_distances.mean().item()
        std_val = all_distances.std().item() if all_distances.numel() > 1 else 0.0
        max_val = all_distances.max().item()
        min_val = all_distances.min().item()
        rel_norm = all_norm.mean().item()
        self.cosine_space = []
        return mean_val, std_val, max_val, min_val,rel_norm