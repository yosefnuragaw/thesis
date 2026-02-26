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

        for option in self.prompts[index]:
            full_text = f"{context_str}{option}{self.eos_token}"
            
            tok = self.tokenizer(
                full_text, 
                return_tensors='pt', 
                add_special_tokens=False
            )
            
            tokenized_row_ids.append(tok.input_ids.squeeze(0))
            tokenized_row_mask.append(tok.attention_mask.squeeze(0))

        return {
            "question_length": question_len,
            "input_ids": tokenized_row_ids,
            "attention_mask": tokenized_row_mask,
            "label": self.labels[index],
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