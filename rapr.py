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
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from datasets import load_dataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap

from models.model import BlockWrapper
from models.prompts import SYSTEM_PROMPT
from utils import set_seed


# ---------------------------------------------------------
# Dataset
# ---------------------------------------------------------

def rapr_collate_fn(batch: list) -> dict:
    return {
        "input_ids":       torch.stack([s["input_ids"]       for s in batch]),
        "attention_mask":  torch.stack([s["attention_mask"]   for s in batch]),
        "d":               torch.stack([s["d"]                for s in batch]),
        "question_length": torch.tensor([s["question_length"] for s in batch]),
        "label":           [s["label"]  for s in batch],
        "decode":          [s["decode"] for s in batch],
    }


class PaddedMultipleOptionDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        prompts: List[List[str]],
        questions: List[str],
        labels: List[str],
        max_length: int | None = None,
        padding_side: str = "right",
    ):
        super().__init__()
        self.tokenizer    = tokenizer
        self.prompts      = prompts
        self.questions    = questions
        self.labels       = labels
        self.padding_side = padding_side

        self.eos_token = tokenizer.eos_token or "<end_of_turn>"
        self.pad_id    = (
            tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else tokenizer.eos_token_id
        )

        self._cache = self._build_cache()

        data_max = max(
            ids.shape[0]
            for sample in self._cache
            for ids in sample["raw_ids"]
        )
        self.max_length = min(max_length, data_max) if max_length else data_max

    def _build_cache(self) -> list:
        cache = []
        for q, opts, lbl in zip(self.questions, self.prompts, self.labels):
            q_tok = self.tokenizer(q, return_tensors="pt", add_special_tokens=False)
            q_len = q_tok.input_ids.shape[1]

            raw_ids, raw_masks, directions, decoded = [], [], [], []
            for i, option in enumerate(opts):
                full = f"{q}{option}{self.eos_token}"
                tok  = self.tokenizer(full, return_tensors="pt", add_special_tokens=False)
                raw_ids.append(tok.input_ids.squeeze(0))
                raw_masks.append(tok.attention_mask.squeeze(0))
                directions.append(1 if i == 0 else -1)
                decoded.append(
                    self.tokenizer.convert_ids_to_tokens(tok.input_ids.squeeze(0))
                )

            cache.append(dict(
                question_length=q_len,
                raw_ids=raw_ids,
                raw_masks=raw_masks,
                directions=directions,
                label=lbl,
                decode=decoded,
            ))
        return cache

    def _pad(self, tensor: torch.Tensor, target_len: int, pad_val: int) -> torch.Tensor:
        cur = tensor.shape[0]
        if cur >= target_len:
            return tensor[:target_len]
        pad = torch.full((target_len - cur,), pad_val, dtype=tensor.dtype)
        return torch.cat(
            [tensor, pad] if self.padding_side == "right" else [pad, tensor]
        )

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, index: int) -> dict:
        sample = self._cache[index]
        L = self.max_length

        padded_ids   = torch.stack(
            [self._pad(ids, L, self.pad_id) for ids in sample["raw_ids"]]
        )
        padded_masks = torch.stack(
            [self._pad(msk, L, 0) for msk in sample["raw_masks"]]
        )

        return {
            "question_length": sample["question_length"],
            "input_ids":       padded_ids,
            "attention_mask":  padded_masks,
            "d":               torch.tensor(sample["directions"], dtype=torch.long),
            "label":           sample["label"],
            "decode":          sample["decode"],
        }


# ---------------------------------------------------------
# Engine
# ---------------------------------------------------------

class PatcherEngine:
    @abstractmethod
    def compute_matrix_sweep(self, multipliers: List[float]):
        pass

    @abstractmethod
    def compute_weight(self):
        pass

    @abstractmethod
    def _init_model(self) -> AutoModelForCausalLM:
        pass


class RAPR(PatcherEngine):
    def __init__(
        self,
        model_name: str,
        vec_dir: str,
        layers: List[int],
        eval_epoch: int,
        loader: DataLoader,
        verbose: bool = False,
    ):
        self.model_name = model_name
        self.vec_dir    = vec_dir
        self.layers     = layers
        self.eval_epoch = eval_epoch
        self.loader     = loader
        self.verbose    = verbose

    def compute_matrix_sweep(self, multipliers: List[float]):
        N    = len(self.layers)
        sweep_results = {1: {}, -1: {}}

        model = self._init_model()
        base  = sorted(self.layers, reverse=True)

        # ── Cache vectors from disk once ───────────────────────────────────
        print("Caching steering vectors...")
        vec_cache = {}
        for layer in self.layers:
            vec_path = f"{self.vec_dir}/vec_ep{self.eval_epoch}_layer{layer}.pt"
            if not os.path.exists(vec_path):
                raise ValueError(f"Vector not found at {vec_path}")
            vec_cache[layer] = torch.load(vec_path, map_location="cpu")

        # ── Set vectors ONCE — they never change across the entire sweep ───
        for layer in self.layers:
            if isinstance(model.model.layers[layer], BlockWrapper):
                layer_device = next(model.model.layers[layer].parameters()).device
                model.model.layers[layer].set_vector(vec_cache[layer].to(layer_device))
                model.model.layers[layer].set_multiplier(0.0)   # start inactive
        print(f"Vectors set on all {len(self.layers)} layers.")

        # ── Pre-collect all samples from loader once ───────────────────────
        print("Pre-collecting dataset into memory...")
        all_pos_ids, all_pos_masks = [], []
        all_neg_ids, all_neg_masks = [], []

        for batch in self.loader:
            d_vals = batch["d"]
            ids    = batch["input_ids"]
            masks  = batch["attention_mask"]
            for i in range(ids.shape[0]):
                for j in range(ids.shape[1]):
                    if d_vals[i, j].item() > 0:
                        all_pos_ids.append(ids[i, j])
                        all_pos_masks.append(masks[i, j])
                    else:
                        all_neg_ids.append(ids[i, j])
                        all_neg_masks.append(masks[i, j])

        # Pre-stack and move to GPU once
        pos_ids_gpu   = torch.stack(all_pos_ids).to(model.device)
        pos_masks_gpu = torch.stack(all_pos_masks).to(model.device)
        neg_ids_gpu   = torch.stack(all_neg_ids).to(model.device)
        neg_masks_gpu = torch.stack(all_neg_masks).to(model.device)
        print(f"Collected {pos_ids_gpu.shape[0]} pos / {neg_ids_gpu.shape[0]} neg samples.")

        # ── Sweep ──────────────────────────────────────────────────────────
        for m in multipliers:
            stat = {
                1:  np.full((N, N), np.nan),
                -1: np.full((N, N), np.nan),
            }

            for direction in [1, -1]:
                pbar = (
                    tqdm(base, desc=f"[Dir: {direction:+d} | Mul: {m}]", ncols=100)
                    if self.verbose else base
                )

                for idx, _ in enumerate(pbar):
                    current_layers = base[: idx + 1]

                    # Only touch the multiplier scalar — vector already loaded
                    for layer in self.layers:
                        if isinstance(model.model.layers[layer], BlockWrapper):
                            if layer in current_layers:
                                model.model.layers[layer].set_multiplier(direction * m)
                            else:
                                model.model.layers[layer].set_multiplier(0.0)

                    # ── Forward pass ───────────────────────────────────────
                    with torch.no_grad():
                        if direction == 1:
                            chunk_size = 32
                            for start in range(0, pos_ids_gpu.shape[0], chunk_size):
                                model(
                                    input_ids=pos_ids_gpu[start:start + chunk_size],
                                    attention_mask=pos_masks_gpu[start:start + chunk_size],
                                )
                        else:
                            model(
                                input_ids=neg_ids_gpu,
                                attention_mask=neg_masks_gpu,
                            )

                    # ── Harvest statistics ─────────────────────────────────
                    for layer in range(N):
                        if (
                            isinstance(model.model.layers[layer], BlockWrapper)
                            and layer in current_layers
                        ):
                            mean, std, max_val, min_val, rel_norm = (
                                model.model.layers[layer].get_cosine_statistics()
                            )
                            stat[direction][N - idx - 1, layer] = mean

            sweep_results[1][m]  = stat[1]
            sweep_results[-1][m] = stat[-1]

        return sweep_results

    def compute_weight(self, sweep_results, direction):
        # 1. Gather sensitivities across the sweep
        # Shape of each heatmap: (32, 32) -> Rows=Readout, Cols=Intervention
        muls = list(sweep_results[direction].keys())
        # sens_matrix shape: (len(muls), 32, 32)
        sens_matrix = np.array([1.0 - sweep_results[direction][m].T for m in muls])
        print(sens_matrix)
        
        # 2. Influence Vector (Column-wise mean)
        # "Which layer, when steered, has the most reach?"
        # We mean over muls and then over the readout rows
        influence_vec = np.nanmean(sens_matrix, axis=(0, 1))
        
        # 3. Friction Vector (Row-wise mean of the DIFF matrix)
        # "At which depth does the model naturally resist this direction?"
        opp_direction = -1 * direction
        # Compute the Diff Matrix: (Target Sens - Distractor Sens)
        diff_matrix = sens_matrix - np.array([1.0 - sweep_results[opp_direction][m].T for m in muls])
        
        # Friction is the row-wise average of this diff
        # High negative value = high resistance at that readout depth
        friction_vec = np.nanmean(diff_matrix, axis=(0, 2))
        
        # 4. Combine: Weight = Base_Influence * Dynamic_Gain(Friction)
        def finalize_opo_weights(influence, friction):
            # Normalize influence to [0, 1]
            norm_inf = (influence - np.min(influence)) / (np.max(influence) - np.min(influence) + 1e-8)
            
            # Normalize friction to [0, 1] 
            # (Where 0 = Max Resistance/Deep Red or Blue, 1 = Max Yield)
            norm_fric = (friction - np.min(friction)) / (np.max(friction) - np.min(friction) + 1e-8)
            
            # Inverse friction logic: Spike gain where the row-wise resistance is high
            # Gain range [0.5, 1.5]
            gain_multiplier = 1.25 - (norm_fric * 1.0)
            
            # The final weight is the influence throttled by the structural friction
            return norm_inf * gain_multiplier

        return finalize_opo_weights(influence_vec, friction_vec)
    
    def _init_model(self) -> AutoModelForCausalLM:
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
                vec=torch.zeros(model.config.hidden_size, dtype=model.dtype),
            )
        model.eval()
        return model


# ---------------------------------------------------------
# Data helpers
# ---------------------------------------------------------

def get_prompts(
    tokenizer,
    behavior,
    system_prompt=SYSTEM_PROMPT,
    generation_prompt: bool = True,
    k: int = 100,
    seed: int = 42,
):
    path = f"./data/{behavior}/train.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")

    dataset = load_dataset("csv", data_files=path, split="train")

    if k is not None:
        sample_size = min(k, len(dataset))
        dataset = dataset.shuffle(seed=seed).select(range(sample_size))

    questions, prompts, labels = [], [], []

    for row in dataset:
        if row["question"] is not None:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": row["question"]},
            ]
            full_prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=generation_prompt
            )
            questions.append(full_prompt)
            current_options = [
                row[col]
                for col in ["matching", "not_matching"]
                if col in row and row[col] is not None
            ]
            prompts.append(current_options)
            labels.append("")

    return {"questions": questions, "prompts": prompts, "labels": labels}


def produce_dataloader(behavior: str, tokenizer: AutoTokenizer) -> DataLoader:
    data = get_prompts(tokenizer=tokenizer, behavior=behavior)

    dataset = PaddedMultipleOptionDataset(
        tokenizer=tokenizer,
        questions=data["questions"],
        prompts=data["prompts"],
        labels=data["labels"],
    )

    return DataLoader(
        dataset=dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        collate_fn=rapr_collate_fn,
    )




def _draw_heatmap(ax, fig, rgba: np.ndarray, norm, cmap, N: int):

    """

    Display the heatmap with:

    x = Steered Layer (32 on left → 1 on right)

    y = Layer Index (0 at bottom → 31 at top, displayed 31 top → 0 bottom)

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

    ax.set_xlabel("Layer Index (0 → 31)", fontsize=11, labelpad=6)

    ax.set_ylabel("Layer Index (31 → 0)", fontsize=11, labelpad=6)

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

def plot_sweep_heatmaps(sweep_results, multipliers_tested, build_heatmap_rgba, _draw_heatmap):
    """
    For each direction (+1, -1), plots all multiplier heatmaps
    concatenated horizontally into a single wide figure.
    """
    for direction in [1, -1]:
        n_muls = len(multipliers_tested)
        fig, axes = plt.subplots(
            1, n_muls,
            figsize=(6 * n_muls, 6),
            sharey=True,
        )
        if n_muls == 1:
            axes = [axes]

        for ax, m in zip(axes, multipliers_tested):
            matrix      = sweep_results[direction][m].T
            active_mask = ~np.isnan(matrix)
            cmap        = plt.get_cmap("viridis")
            rgba, norm  = build_heatmap_rgba(matrix, active_mask, cmap)
            N           = matrix.shape[0]

            _draw_heatmap(ax, fig, rgba, norm, cmap, N)
            ax.set_title(f"Mul: {m}", fontsize=11, pad=6, fontweight="bold")

            # Only leftmost axis keeps the y-label
            if ax != axes[0]:
                ax.set_ylabel("")

        dir_label = "+1 (Unsafe)" if direction == 1 else "-1 (Safe)"
        fig.suptitle(
            f"Cosine-Distance Heatmap Sweep — Direction: {dir_label}",
            fontsize=13,
            fontweight="bold",
            y=1.02,
        )

        plt.tight_layout()
        fname = f"heatmap_sweep_dir{'pos' if direction == 1 else 'neg'}.png"
        plt.savefig(fname, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {fname}")
        


def plot_sweep_diff_heatmap(sweep_results, multipliers_tested, build_heatmap_rgba, _draw_heatmap):
    def _minmax_scale(matrix: np.ndarray) -> np.ndarray:
        """Scale a matrix to [0, 1] using its own finite min/max, NaNs preserved."""
        out  = np.full_like(matrix, np.nan, dtype=float)
        mask = ~np.isnan(matrix)
        if mask.any():
            lo, hi = matrix[mask].min(), matrix[mask].max()
            if hi > lo:
                out[mask] = (matrix[mask] - lo) / (hi - lo)
            else:
                out[mask] = 0.0
        return out

    n_muls = len(multipliers_tested)
    fig, axes = plt.subplots(
        1, n_muls,
        figsize=(6 * n_muls, 6),
        sharey=True,
        constrained_layout=True,   # replaces tight_layout — handles colorbars cleanly
    )
    fig.patch.set_facecolor("white")
    if n_muls == 1:
        axes = [axes]

    cmap = LinearSegmentedColormap.from_list(
        "red_white_blue",
        ["#d62728", "#ffffff", "#1f77b4"],
        N=256,
    )

    all_diffs = []
    for m in multipliers_tested:
        scaled_pos = _minmax_scale(sweep_results[1][m].T)
        scaled_neg = _minmax_scale(sweep_results[-1][m].T)
        all_diffs.append(scaled_pos - scaled_neg)

    finite_vals = np.concatenate([d[~np.isnan(d)] for d in all_diffs])
    vmax = np.nanmax(np.abs(finite_vals))
    norm = plt.Normalize(vmin=-vmax, vmax=vmax)

    for ax, m, diff in zip(axes, multipliers_tested, all_diffs):
        ax.set_facecolor("white")
        N           = diff.shape[0]
        active_mask = ~np.isnan(diff)

        rgba               = cmap(norm(np.where(active_mask, diff, 0.0)))
        rgba[~active_mask] = [0.85, 0.85, 0.85, 1.0]

        ax.imshow(
            rgba, aspect="auto", origin="upper",
            extent=[-0.5, N - 0.5, N - 0.5, -0.5],
        )
        ax.set_title(f"Mul: {m}", fontsize=11, pad=6, fontweight="bold")
        ax.set_xlabel("Layer index", fontsize=9)
        if ax == axes[0]:
            ax.set_ylabel("Layer subset depth", fontsize=9)
        else:
            ax.set_ylabel("")

        ax.set_xticks(range(0, N, 1))
        ax.set_yticks(range(0, N, 1))
        ax.tick_params(labelsize=7)

        for spine in ax.spines.values():
            spine.set_edgecolor("#cccccc")
            spine.set_linewidth(0.8)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    # Use fig.colorbar with a dedicated inset — constrained_layout manages spacing
    cbar = fig.colorbar(sm, ax=axes, fraction=0.015, pad=0.02, shrink=0.85)
    cbar.set_label("Δ Cosine distance  (+1) − (−1)  [min-max scaled]", fontsize=10)
    cbar.ax.tick_params(labelsize=8)

    fig.text(
        0.5, -0.01,
        "Blue = positive steering weaker   |   Red = negative steering weaker",
        ha="center", fontsize=10, color="#444444",
    )
    fig.suptitle(
        "Diff Heatmap Sweep (−1)  [min-max scaled]",
        fontsize=13, fontweight="bold",
    )

    # No plt.tight_layout() call — constrained_layout handles it automatically
    fname = "heatmap_sweep_diff.png"
    plt.savefig(fname, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {fname}")

if __name__ == "__main__":
    model_id = "meta-llama/Llama-3.1-8B-Instruct"

    device    = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    loader = produce_dataloader(behavior="wealth-seeking", tokenizer=tokenizer)

    engine = RAPR(
        model_name=model_id,
        vec_dir="pretrained_vector/wealth-seeking/llama-3/all",
        layers=list(range(32)),
        eval_epoch=9,
        loader=loader,
        verbose=True,
    )

    multipliers_to_test = [0.5]
    print(f"\nStarting Calibration Sweep across Multipliers: {multipliers_to_test}")

    sweep_results = engine.compute_matrix_sweep(multipliers_to_test)
    plot_sweep_heatmaps(sweep_results, multipliers_to_test, build_heatmap_rgba, _draw_heatmap)
    plot_sweep_diff_heatmap(sweep_results, multipliers_to_test, build_heatmap_rgba, _draw_heatmap)

    w_pos,w_neg = engine.compute_weight(sweep_results,direction=1),engine.compute_weight(sweep_results,direction=-1)

    print(f'pos_weight: {w_pos.tolist()}')
    print(f'neg_weight: {w_neg.tolist()}')