from typing import  Dict, Optional, override
import torch
from torch.utils.data import Dataset


MODEL_TEMPLATE_MAP: Dict[str, str]= {
    'meta-llama/Llama-2-7b-chat-hf': 'llama-2',
    'mistralai/Mistral-7B-Instruct-v0.2': 'mistral',
    'google/gemma-3-1b-it': 'gemma-3',
    'Qwen/Qwen3-8B': 'qwen-7b'
}

VALID_APPLY_TYPES = {'base','layer', 'sequence'}



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
            if skip == 'adap':
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
            
        mask = self.multiplier 

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

class CAABlockWrapper(torch.nn.Module):
    def __init__(self, block, hidden_dim, vec: Optional[torch.Tensor] = None, apply_type: str = 'layer'):
        super().__init__()
        self.multiplier = 1.0
        self.block = block
        self.is_extract = False  # initialize properly
        try:
            ref_param = next(block.parameters())
            self.init_dtype = ref_param.dtype
        except StopIteration:
            self.init_dtype = torch.float32

        if vec is not None:
            self.vec = vec.to(self.init_dtype)
        else:
            self.vec = torch.nn.Parameter(torch.zeros(hidden_dim, dtype=self.init_dtype))

        self.caa_buffer = {'pos': [], 'neg': []}

        if apply_type not in VALID_APPLY_TYPES:
            raise ValueError(f"apply_type must be 'layer', or 'sequence', got {apply_type!r}")
        self.apply_type = apply_type

    def extract(self, status: bool):
        self.is_extract = status

    def set_multiplier(self, mul: float) -> None:
        self.multiplier = mul

    def set_vec(self, vec) -> None:
        self.vec = vec

    def record(self, activation):
        if self.is_extract:
            if self.multiplier > 0:
                self.caa_buffer['pos'].append(activation)
            elif self.multiplier < 0:
                self.caa_buffer['neg'].append(activation)

    def extract_vec(self, clear: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.caa_buffer['pos'] and not self.caa_buffer['neg']:
            raise ValueError("Buffer is empty, run extraction first.")
        
        pos_stacked = torch.stack(self.caa_buffer['pos'])
        neg_stacked = torch.stack(self.caa_buffer['neg'])

        if pos_stacked.dim() > 2:
            vec_pos = pos_stacked.mean(dim=(0, 1))  # [D]
            vec_neg = neg_stacked.mean(dim=(0, 1))  # [D]
        else:
            # Fallback if B was 1 and squeezed, shape is [N, D]
            vec_pos = pos_stacked.mean(dim=0)       # [D]
            vec_neg = neg_stacked.mean(dim=0)       # [D]
        
        if clear:
            self.caa_buffer = {'pos': [], 'neg': []}
        
        # Calculate the difference vector directly here
        sv_pos = vec_pos - vec_neg
        sv_neg = vec_neg - vec_pos
        
        return sv_pos, sv_neg
    @override
    def forward(self, hidden_states, *args, **kwargs):
        def __cosine_distance(vec_1, vec_2):
            if vec_2.dim() == 3 and vec_1.dim() == 1:
                vec_1_expanded = vec_1.unsqueeze(0).unsqueeze(0).expand_as(vec_2)
            elif vec_2.dim() == 2 and vec_1.dim() == 1:
                vec_1_expanded = vec_1.unsqueeze(0).expand_as(vec_2)
            else:
                vec_1_expanded = vec_1

            cos_sim = torch.nn.functional.cosine_similarity(vec_1_expanded, vec_2, dim=-1)
            cos_sim_c = torch.clamp(cos_sim, min=0.25, max=1.0)
            cos_dis = cos_sim_c
            return cos_dis

        output = self.block(hidden_states, *args, **kwargs)
        out_tensor = output[0] if isinstance(output, tuple) else output
        avg_output = out_tensor.detach().mean(dim=1)  # [B, D]

        if self.is_extract:
            self.record(avg_output.cpu())
        else:
            current_vec = (self.multiplier * self.vec).to(out_tensor.device)
            mask = self.multiplier 

            if self.apply_type == 'layer':
                mask = mask * __cosine_distance(current_vec, avg_output)   

            elif self.apply_type == 'sequence':
                mask = mask * __cosine_distance(current_vec, out_tensor.detach())  

            # Ensure the mask can be multiplied against [B, T, D] or [B, D]
            if isinstance(mask, torch.Tensor):
                while mask.dim() < out_tensor.dim():
                    mask = mask.unsqueeze(-1) # [B, 1] or [B, T, 1]

            # Apply the steering
            if isinstance(output, tuple):
                modified_hidden = output[0] + (mask * self.vec.to(output[0].device))
                output = (modified_hidden,) + output[1:]
            elif isinstance(output, torch.Tensor):
                output = output + (mask * self.vec.to(output.device))

        return output