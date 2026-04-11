from typing import List, Dict
import torch
from torch.utils.data import Dataset

class MultipleOptionDataset(Dataset):
    def __init__(self, tokenizer, prompts: List[List[str]], questions: List[str], labels: List[str]):
        super().__init__()
        self.tokenizer = tokenizer
        self.prompts = prompts   
        self.questions = questions 
        self.labels = labels
        
        # Cache the EOS token to avoid repeated attribute lookups
        self.eos_token = self.tokenizer.eos_token if self.tokenizer.eos_token else "<end_of_turn>"

    def __getitem__(self, index: int):
        context_str = self.questions[index]
       
        tokenized_question = self.tokenizer(
            context_str, 
            return_tensors='pt', 
            add_special_tokens=False
        )
        question_len = tokenized_question.input_ids.shape[1]

        tokenized_row_ids = []
        tokenized_row_mask = []

        
        directions = []
        for i, option in enumerate(self.prompts[index]):
            full_text = f"{context_str}{option}{self.eos_token}"
            
            tok = self.tokenizer(
                full_text, 
                return_tensors='pt', 
                add_special_tokens=False
            )
            
            tokenized_row_ids.append(tok.input_ids.squeeze(0))
            tokenized_row_mask.append(tok.attention_mask.squeeze(0))
            
            # Map index 0 to Direction 1, and index 1 to Direction -1
            # (Adjust this math if you actually want 0 and 1!)
            dir_value = 1 if i == 0 else -1 
            directions.append(dir_value)

        decoded = [self.tokenizer.convert_ids_to_tokens(ids) for ids in tokenized_row_ids]
        return {
            "question_length": question_len,
            "input_ids": tokenized_row_ids,
            "attention_mask": tokenized_row_mask,
            "label": self.labels[index],
            'd':directions,
            "decode": decoded
        }
    
    def __len__(self) -> int:
        return len(self.prompts)

    
class PromptDataset(Dataset):
    def __init__(self, prompts):
        self.prompts = prompts
    def __len__(self):
        return len(self.prompts)
    def __getitem__(self, i):
        return self.prompts[i]
    

from typing import List
import torch
from torch.utils.data import Dataset


class PaddedMultipleOptionDataset(Dataset):
    """
    Like MultipleOptionDataset but pads all sequences to a uniform
    length so samples can be stacked into batches without collision.

    Each __getitem__ returns:
      - input_ids:      [num_options, max_seq_len]  (int64)
      - attention_mask: [num_options, max_seq_len]  (int64, 0 on pad tokens)
      - question_length: scalar int
      - d:              [num_options]  (int, +1 / -1)
      - label:          str
      - decode:         list[list[str]]  token strings per option
    """

    def __init__(
        self,
        tokenizer,
        prompts: List[List[str]],
        questions: List[str],
        labels: List[str],
        max_length: int | None = None,   # hard cap; None = derive from data
        padding_side: str = "right",     # "right" matches most causal-LM conventions
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.prompts = prompts
        self.questions = questions
        self.labels = labels
        self.padding_side = padding_side

        self.eos_token = tokenizer.eos_token or "<end_of_turn>"
        self.pad_id = (
            tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else tokenizer.eos_token_id
        )

        # Pre-tokenise everything once so __getitem__ is O(1) and
        # so we can compute a data-driven max_length in one pass.
        self._cache = self._build_cache()
        
        data_max = max(
            ids.shape[0]
            for sample in self._cache
            for ids in sample["raw_ids"]
        )
        self.max_length = min(max_length, data_max) if max_length else data_max

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_cache(self) -> list:
        cache = []
        for q, opts, lbl in zip(self.questions, self.prompts, self.labels):

            q_tok = self.tokenizer(q, return_tensors="pt", add_special_tokens=False)
            q_len = q_tok.input_ids.shape[1]

            raw_ids, raw_masks, directions, decoded = [], [], [], []
            for i, option in enumerate(opts):
                full = f"{q}{option}{self.eos_token}"
                tok = self.tokenizer(full, return_tensors="pt", add_special_tokens=False)
                raw_ids.append(tok.input_ids.squeeze(0))        # [seq_len]
                raw_masks.append(tok.attention_mask.squeeze(0)) # [seq_len]
                directions.append(1 if i == 0 else -1)
                decoded.append(self.tokenizer.convert_ids_to_tokens(tok.input_ids.squeeze(0)))

            cache.append(
                dict(
                    question_length=q_len,
                    raw_ids=raw_ids,
                    raw_masks=raw_masks,
                    directions=directions,
                    label=lbl,
                    decode=decoded,
                )
            )
        return cache

    def _pad(self, tensor: torch.Tensor, target_len: int, pad_val: int) -> torch.Tensor:
        """Pad or truncate a 1-D tensor to target_len."""
        cur = tensor.shape[0]
        if cur >= target_len:
            return tensor[:target_len]
        pad = torch.full((target_len - cur,), pad_val, dtype=tensor.dtype)
        return torch.cat([tensor, pad] if self.padding_side == "right" else [pad, tensor])

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, index: int) -> dict:
        sample = self._cache[index]
        L = self.max_length

        padded_ids   = torch.stack(
            [self._pad(ids,  L, self.pad_id) for ids  in sample["raw_ids"]]
        )  # [num_options, L]
        padded_masks = torch.stack(
            [self._pad(msk, L, 0)            for msk  in sample["raw_masks"]]
        )  # [num_options, L]

        return {
            "question_length": sample["question_length"],
            "input_ids":       padded_ids,
            "attention_mask":  padded_masks,
            "d":               torch.tensor(sample["directions"], dtype=torch.long),
            "label":           sample["label"],
            "decode":          sample["decode"],
        }