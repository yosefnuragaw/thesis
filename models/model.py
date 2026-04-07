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

        # with torch.no_grad():
        #     # Pastikan dimensi sesuai untuk broadcasting: [1, 1, hidden]
        #     v_mul = self.multiplier * self.vec.to(out_target.device).view(1, 1, -1)
            
        #     # 1. Konversi ke Distribusi Probabilitas (Softmax)
        #     # v_mul sebagai distribusi target ideal (P)
        #     # out_target sebagai distribusi tebakan model saat ini (Q)
        #     p_probs = torch.nn.functional.softmax(v_mul, dim=-1)
            
        #     # Untuk Q, kita gunakan log_softmax demi stabilitas numerik dan kemudahan rumus CE
        #     q_log_probs = torch.nn.functional.log_softmax(out_target, dim=-1)
            
        #     # 2. Hitung Cross-Entropy per Token
        #     # Rumus: CE = -sum(P * log(Q)) pada dimensi hidden (dim=-1)
        #     cross_entropy = -(p_probs * q_log_probs).sum(dim=-1, keepdim=True)
            
        #     # 3. Ubah Cross-Entropy menjadi Gerbang/Masking (Range 0 sampai 1)
        #     # Nilai CE berkisar dari 0 (identik) hingga tak terhingga (sangat berbeda).
        #     # Kita gunakan fungsi eksponensial (e^-x) untuk memetakannya ke rentang 0-1.
            
        #     # OPSI A: Injeksi kuat jika pola mirip (CE rendah -> mask mendekati 1)
        #     # soft_mask = torch.exp(-cross_entropy)
            
        #     # OPSI B (Jika Anda ingin sebaliknya): Injeksi kuat jika pola sangat BERBEDA
        #     soft_mask = 1.0 - torch.exp(-cross_entropy)
        with torch.no_grad():
            v_mul = self.multiplier * self.vec.to(out_target.device).view(1, 1, -1)
            
            # 1. UPCAST KE FP32 (Wajib untuk 4096 dimensi)
            out_fp32 = out_target.to(torch.float32)
            v_mul_fp32 = v_mul.to(torch.float32)
            
            # 2. COSINE SIMILARITY MENTAH (Bukan Probabilitas)
            # Rentang hasil alami: -1.0 (Berlawanan) hingga 1.0 (Searah)
            cos_sim_raw = F.cosine_similarity(out_fp32, v_mul_fp32, dim=-1).unsqueeze(-1)
            
            # --- CEK STATISTIK MENTAH ---
            print(f"Max Cosine Mentah: {cos_sim_raw.max().item():.4f}")
            print(f"Min Cosine Mentah: {cos_sim_raw.min().item():.4f}")
            print(f"Var Cosine Mentah: {cos_sim_raw[0].var().item():.6f}")
            print("RAW Cosine Token Tengah:\n", cos_sim_raw[0, 150:160, :].squeeze())
            
            # 3. MIN-MAX SCALING (Menciptakan Varians 0.0 - 1.0 yang Sehat)
            # Kita cari kemiripan tertinggi dan terendah dalam SATU kalimat
            cos_min = cos_sim_raw.min(dim=1, keepdim=True)[0]
            cos_max = cos_sim_raw.max(dim=1, keepdim=True)[0]
            
            # Hindari error bagi-dengan-nol
            cos_range = torch.clamp(cos_max - cos_min, min=1e-6)
            
            # Normalisasi: Token paling mirip jadi 1.0, paling beda jadi 0.0
            soft_mask_fp32 = (cos_sim_raw - cos_min) / cos_range
            
            # (Opsional) Ketajaman Filter - Memangkas token yang cuma "agak" mirip
            # Gunakan pangkat (power). Semakin tinggi pangkatnya (misal 3.0), semakin selektif.
            soft_mask_fp32 = soft_mask_fp32 ** 3.0
            
            # 4. DOWNCAST KE BFLOAT16
            soft_mask = soft_mask_fp32.to(out_target.dtype)

        # 5. KALKULASI INJEKSI AKHIR
        injection = soft_mask * v_mul
        
        # --- LOGGING AKHIR ---
        soft_mask_check = soft_mask[0].to(torch.float32)
        print("\nMask Token Tengah:\n", soft_mask_check[150:160].squeeze())
        print(f"Mean Injeksi (Sampel 0): {soft_mask_check.mean().item():.4f}")
        print(f"Variance Mask Akhir: {soft_mask_check.var().item():.6f}")

        # --- LOGGING ---
        sum_per_sample = soft_mask.sum(dim=(1, 2)) 
        ratio_per_sample = (sum_per_sample / out_target.shape[1]) * 100

        print("10 Pertama:\n", soft_mask[0, :10, :].squeeze())
        print("10 Terakhir:\n", soft_mask[0, -10:, :].squeeze())

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
