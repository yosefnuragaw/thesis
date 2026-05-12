from typing import  Dict, Optional, override
import torch
from torch.utils.data import Dataset


MODEL_TEMPLATE_MAP: Dict[str, str]= {
    'meta-llama/Llama-2-7b-chat-hf': 'llama-2',
    'mistralai/Mistral-7B-Instruct-v0.2': 'mistral',
    'google/gemma-3-1b-it': 'gemma-3',
    'Qwen/Qwen3-8B': 'qwen-7b'
}

VALID_APPLY_TYPES = {'base','layer', 'sequence','sequence2'}



class MaskGate(torch.nn.Module):
    def __init__(self, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.sigmoid =  torch.nn.functional.sigmoid()
        self.h = torch.nn.Parameter(torch.tensor([0.0], dtype=dtype))

    def forward(self):
        return self.sigmoid(self.h)

class BlockWrapper(torch.nn.Module):
    def __init__(self, block, hidden_dim, vec: Optional[torch.Tensor] = None, buffer: bool = False, apply_type: str = 'layer', treshold: float = 1.0):
        super().__init__()
        self.multiplier = 1.0
        self.block = block

        if apply_type not in VALID_APPLY_TYPES:
            raise ValueError(f"apply_type must be 'base', 'layer', or 'sequence', got {apply_type!r}")
        self.apply_type = apply_type
        self.treshold = treshold

        try:
            ref_param = next(block.parameters())
            self.init_dtype = ref_param.dtype
        except StopIteration:
            self.init_dtype = torch.float32
            
        if vec is not None:
            self.vec = vec.to(self.init_dtype)
        else:
            self.vec = torch.nn.Parameter(torch.zeros(hidden_dim, dtype= self.init_dtype))

        self.gate = MaskGate(dtype=self.init_dtype)
        self.gen_step = 0 
        self.buffer = buffer
        self.buffer_space = []
        self.cosine_space = []
        self.rel_norm_space = []
    
    def forward(self, hidden_states, *args, **kwargs):
        def __cosine_similarity(vec_1, vec_2):
            if vec_2.dim() == 3 and vec_1.dim() == 1:
                vec_1_expanded = vec_1.unsqueeze(0).unsqueeze(0).expand_as(vec_2)
            elif vec_2.dim() == 2 and vec_1.dim() == 1:
                vec_1_expanded = vec_1.unsqueeze(0).expand_as(vec_2)
            else:
                vec_1_expanded = vec_1

            return torch.nn.functional.cosine_similarity(vec_1_expanded, vec_2, dim=-1)
          
        
        output = self.block(hidden_states, *args, **kwargs)
        out_tensor = output[0] if isinstance(output, tuple) else output
        avg_output = out_tensor.detach().mean(dim=1)
        current_vec = (self.multiplier * self.vec).to(out_tensor.device)
        mask = self.multiplier 
        if self.apply_type == 'layer':
            cos_sim = __cosine_similarity(current_vec, avg_output)
            cos_sim_c = cos_sim.clamp(min=self.treshold, max=1.0)
            mask = mask * cos_sim_c

        elif self.apply_type == 'sequence':
            cos_sim = __cosine_similarity(current_vec, out_tensor.detach())
            cos_sim_c = cos_sim.clamp(min=self.treshold, max=1.0)
            mask = mask * cos_sim_c

        elif self.apply_type == 'sequence2':
            cos_sim = __cosine_similarity(current_vec, out_tensor.detach())  # [B, T]
            bonus = (1 + self.gain * (cos_sim - self.gate.forward()) / (1 - self.gate.forward())).clamp(min=0)
            mask  = mask * cos_sim.clamp(min=0) * bonus

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
        self.vec = torch.nn.Parameter(vec.to(self.init_dtype), requires_grad=False)

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

class CAABlockWrapper(torch.nn.Module):
    def __init__(self, block, hidden_dim, vec: Optional[torch.Tensor] = None, apply_type: str = 'layer', treshold: float = 0.25):
        super().__init__()
        self.multiplier = 1.0
        self.block = block
        self.is_extract = False  # initialize properly
        self.treshold = 0.
        # self.gain = 2.0
        self._prev_token = None 
        self.drift_history = []
        try:
            ref_param = next(block.parameters())
            self.init_dtype = ref_param.dtype
        except StopIteration:
            self.init_dtype = torch.float32

        if vec is not None:
            self.vec = vec.to(self.init_dtype)
        else:
            print("Non==============")
            self.vec = torch.zeros(hidden_dim, dtype=self.init_dtype)

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
        def __cosine_similarity(vec_1, vec_2):
            if vec_2.dim() == 3 and vec_1.dim() == 1:
                vec_1_expanded = vec_1.unsqueeze(0).unsqueeze(0).expand_as(vec_2)
            elif vec_2.dim() == 2 and vec_1.dim() == 1:
                vec_1_expanded = vec_1.unsqueeze(0).expand_as(vec_2)
            else:
                vec_1_expanded = vec_1
            return torch.nn.functional.cosine_similarity(vec_1_expanded, vec_2, dim=-1)

        output = self.block(hidden_states, *args, **kwargs)
        out_tensor = output[0] if isinstance(output, tuple) else output
        detached_tensor = out_tensor.detach() # [B, D]

        if self.is_extract:
            response_mask = kwargs.get("response_mask", getattr(self, "current_mask", None))
            if response_mask is not None:
                mask_expanded = response_mask.unsqueeze(-1).to(detached_tensor.dtype)
                masked_tensor = detached_tensor * mask_expanded
                avg_output = masked_tensor.mean(dim=1)
            else:
                raise ValueError('Response Mask Not Found')
            
            batch_avg = avg_output.mean(dim=0)
            self.record(batch_avg.cpu())
        else:
            current_vec = (self.multiplier * self.vec).to(out_tensor.device)
            mask = self.multiplier

            if self.apply_type == 'layer':
                cos_sim = __cosine_similarity(current_vec, avg_output)
                cos_sim_c = cos_sim.clamp(min=self.treshold, max=1.0)
                mask = mask * cos_sim_c

            elif self.apply_type == 'sequence':
                cos_sim = __cosine_similarity(current_vec, out_tensor.detach())
                cos_sim_c = cos_sim.clamp(min=self.treshold, max=1.0)
                mask = mask * cos_sim_c

            elif self.apply_type == 'sequence2':
                last_token = out_tensor.detach()[:, -1, :]
            
                if out_tensor.shape[1] > 1:
                    prev_token = out_tensor.detach()[:, -2, :]
                elif self._prev_token is not None:
                    prev_token = self._prev_token.to(out_tensor.device)
                else:
                    prev_token = last_token
            
                self._prev_token = last_token.clone()
            
                # --- Variance: low = repeating, high = drifting ---
                # --- Variance: low = repeating, high = drifting ---
                token_seq = torch.stack([prev_token, last_token], dim=1)  # [B, 2, D]
                seq_var = token_seq.var(dim=1).mean(dim=-1, keepdim=True)  # [B, 1]

                # normalize by average token norm squared — per sample, scale-free
                token_norm_sq = token_seq.pow(2).mean(dim=(1, 2), keepdim=True)  # [B, 1, 1]
                var_norm = (seq_var / (token_norm_sq.squeeze(1) + 1e-6)).clamp(0.0, 1.0)

                direction = var_norm * 2.0 - 1.0  # [-1, +1]
                
                # --- Alignment: how far current token is from concept ---
                cos_sim_curr = __cosine_similarity(last_token, current_vec).clamp(-1.0, 1.0)  # [B]
                alignment = (1.0 - cos_sim_curr).unsqueeze(-1)  # [B, 1], high = far from concept
                
                # --- Direction: sign from variance, magnitude from alignment (no quadratic shrink) ---
                # alignment scales the mask directly; direction only flips the sign
                signed_alignment = alignment * direction.sign()          # [-1,+1] sign, [0,1] magnitude
                signed_alignment = signed_alignment.clamp(min=-0.25)     # dampen subtraction
                
                mask = mask * signed_alignment
            
                self.drift_history.append(signed_alignment.squeeze(-1).detach().cpu())

            if isinstance(mask, torch.Tensor):
                while mask.dim() < out_tensor.dim():
                    mask = mask.unsqueeze(-1)  # [B, 1] or [B, T, 1]

            if isinstance(output, tuple):
                modified_hidden = output[0] + (mask * self.vec.to(output[0].device))
                output = (modified_hidden,) + output[1:]
            elif isinstance(output, torch.Tensor):
                output = output + (mask * self.vec.to(output.device))

        return output
    
   

    def extract_and_clear_drift(self):
        """Calculates the average drift and clears the cache for the next run."""
        if not self.drift_history:
            return None
        
        # FIX: Flatten each tensor to 1D and concatenate them. 
        # This completely bypasses the batch size mismatch error.
        all_drifts = torch.cat([d.flatten() for d in self.drift_history])
        
        # Calculate the global average drift for this layer
        avg_drift = torch.mean(all_drifts) 
        
        # Clear to prevent bleeding into the next multiplier evaluation
        self.drift_history.clear()
        
        return avg_drift