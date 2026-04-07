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
    def forward(self, x):
        return self.func(self.h)


class MaskGate2(torch.nn.Module):
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
        self.h2 = torch.nn.Parameter(torch.tensor([0.0], dtype=dtype))
    def forward(self, x, param):
        if param == 's':
            return self.func(self.h)
        elif param == 'b':
            return self.func(self.h2)



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
            if skip == 'llds':
                self.gate_mask= MaskGate(hidden_dim = hidden_dim, dtype=self.init_dtype, function=gate_function)
            elif skip == 'llbds':
                self.gate_mask = MaskGate2(hidden_dim = hidden_dim, dtype=self.init_dtype, function=gate_function)
            
        else:
            self.gate_mask = None

        self.skip =skip

        self.buffer = buffer
        self.buffer_space = []
        self.cosine_space = []

    def forward(self, hidden_states, *args, **kwargs):
        output = self.block(hidden_states, *args, **kwargs)

        with torch.no_grad():
            out_tensor = output[0] if isinstance(output, tuple) else output
            avg_hidden = hidden_states.detach().mean(dim=1)
            avg_output = out_tensor.detach().mean(dim=1)
            cos_sim = torch.nn.functional.cosine_similarity(avg_hidden, avg_output, dim=-1)
            cos_sim_c= torch.clamp(cos_sim, min=-1.0, max=1.0) 
            cos_dis = 1 - cos_sim_c
       
        mask = self.multiplier 
        
        if self.skip:
            if self.skip == 'llbds':
                llbds =  1 + self.gate_mask(cos_dis,'s').to(output[0].device)*(cos_dis - self.gate_mask(cos_dis, 'b').to(output[0].device))
                mask = mask * llbds

        if isinstance(mask, torch.Tensor):
            while mask.dim() < hidden_states.dim():
                mask = mask.unsqueeze(-1)

        # if isinstance(output, tuple):
        #     self.buffer_space.append(output[0].detach().mean(dim=1).cpu())
        #     modified_hidden = output[0] + (mask * self.vec.to(output[0].device))
        #     output = (modified_hidden,) + output[1:]
            
        # elif isinstance(output, torch.Tensor):
        #     self.buffer_space.append(output.detach().mean(dim=1).cpu())
        #     output = output + (mask * self.vec.to(output.device))

        out_target = output[0] if isinstance(output, tuple) else output

        t = 0.20 

        with torch.no_grad():
            v_mul = self.multiplier * self.vec.to(out_target.device)
            
            # Mean centering
            mean_h = out_target.mean(dim=-1, keepdim=True)
            mean_v = v_mul.mean()
            
            centered_h = out_target - mean_h
            centered_v = v_mul - mean_v
            
            # Covariance & Variance
            covariance = (centered_h * centered_v).mean(dim=-1, keepdim=True)
            var_h = centered_h.pow(2).mean(dim=-1, keepdim=True)
            var_v = centered_v.pow(2).mean()
            
            # Pearson Correlation
            std_product = torch.sqrt(var_h * var_v) + 1e-6
            correlation = covariance / std_product
            abs_correlation = torch.abs(correlation)
            
            # 1. Buat Gerbang Biner (Lolos Threshold atau Tidak)
            binary_gate = (abs_correlation < 0.25).to(out_target.dtype)
            
            # --- MODIFIKASI SOFT MASKING DI SINI ---
            # 2. Kalikan gerbang dengan nilai korelasinya
            soft_mask =  binary_gate 

        # 3. Kalkulasi Injeksi
        # 'mask' (self.multiplier) dikalikan dengan soft_mask dan vektor v
        print(out_target.shape)
        print(soft_mask.shape)
        print(soft_mask.sum()/ len(soft_mask))
        raise ValueError
        injection = soft_mask * (self.multiplier * self.vec.to(out_target.device))

        # 4. Implementasi ke dalam arsitektur
        if isinstance(output, tuple):
            self.buffer_space.append(out_target.detach().mean(dim=1).cpu())
            modified_hidden = out_target + injection
            output = (modified_hidden,) + output[1:]
            
        elif isinstance(output, torch.Tensor):
            self.buffer_space.append(out_target.detach().mean(dim=1).cpu())
            output = out_target + injection

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

    def get_cosine_statistics(self):
        if not self.cosine_space:
            return 0.0, 0.0
            
        all_distances = torch.cat(self.cosine_space, dim=0)
        
        mean_val = all_distances.mean().item()
        std_val = all_distances.std().item() if all_distances.numel() > 1 else 0.0
        
        return mean_val, std_val
    

    def apply_sparse_steering(self, hidden_states):
        t = 0.25 

        with torch.no_grad():
            v = self.vec.to(hidden_states.device)
            
            mean_h = hidden_states.mean(dim=-1, keepdim=True)
            mean_v = v.mean()

            centered_h = hidden_states - mean_h
            centered_v = v - mean_v
            
            # Hitung Kovarians
            covariance = (centered_h * centered_v).mean(dim=-1, keepdim=True)
            abs_covariance = torch.abs(covariance)
            
            mask = (abs_covariance >= t).to(hidden_states.dtype)
        
        # 2. Injeksi terarah DI LUAR no_grad
        injection = mask * (self.multiplier * v)
        steered_states = hidden_states + injection
        
        return steered_states
