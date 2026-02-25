from typing import List, Dict
import torch
from torch.utils.data import Dataset
from fastchat.conversation import Conversation


SYSTEM_PROMPT: str = "You are a helpful, honest and concise assistant."
MODEL_TEMPLATE_MAP: Dict[str, str]= {
    'meta-llama/Llama-2-7b-chat-hf': 'llama-2',
    'mistralai/Mistral-7B-Instruct-v0.2': 'mistral',
    'google/gemma-3-1b-it': 'gemma-3',
    'Qwen/Qwen3-8B': 'qwen-7b'
}

class BlockWrapper(torch.nn.Module):
    def __init__(self, block, hidden_dim, vec=None):
        super().__init__()
        self.multiplier = 1.0
        self.block = block

        try:
            ref_param = next(block.parameters())
            init_dtype = ref_param.dtype
        except StopIteration:
            init_dtype = torch.float32
            
        if vec is not None:
            self.vec = vec
        else:
            self.vec = torch.nn.Parameter(torch.zeros(hidden_dim, dtype=init_dtype))

    def forward(self, *args, **kwargs):
        output = self.block(*args, **kwargs)
        if isinstance(output, tuple):
            modified_hidden = output[0] + (self.multiplier * self.vec)
            return (modified_hidden,) + output[1:]
        elif isinstance(output, torch.Tensor):
            return output + (self.multiplier * self.vec)
        else:
            return output

    def set_multiplier(self, multiplier):
        self.multiplier = multiplier

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.block, name)
        
class Gemma3Conversation(Conversation):
    def __init__(self):
        super().__init__(
            name="gemma-3",
            system_template="<bos><start_of_turn>system\n{system_message}<end_of_turn>\n",
            roles=("user", "assistant"),
            messages=[],
            sep="",
            sep2="",
            stop_str="<end_of_turn>",
            stop_token_ids=[1],  
        )

    def append_message(self, role, message):
        if role == "user":
            formatted = f"<start_of_turn>user\n{message}<end_of_turn>\n"
            self.messages.append((role, formatted))
        elif role == "assistant":
            if message is None:
                return 
            formatted = f"<start_of_turn>model\n{message}<end_of_turn>\n"
            self.messages.append((role, formatted))
        else:
            raise ValueError(f"Unknown role: {role}")

    def get_prompt(self):
        prompt = ""
        if self.system_message:
            prompt += self.system_template.format(system_message=self.system_message)
        
        for _, content in self.messages:
            prompt += content
            
        prompt += "<start_of_turn>model\n"
        return prompt

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

        
class Gemma3Conversation(Conversation):
    def __init__(self):
        super().__init__(
            name="gemma-3",
            system_template="<bos><start_of_turn>system\n{system_message}<end_of_turn>\n",
            roles=("user", "assistant"),
            messages=[],
            sep="",
            sep2="",
            stop_str="<end_of_turn>",
            stop_token_ids=[1],  
        )

    def append_message(self, role, message):
        if role == "user":
            formatted = f"<start_of_turn>user\n{message}<end_of_turn>\n"
            self.messages.append((role, formatted))
        elif role == "assistant":
            if message is None:
                return 
            formatted = f"<start_of_turn>model\n{message}<end_of_turn>\n"
            self.messages.append((role, formatted))
        else:
            raise ValueError(f"Unknown role: {role}")

    def get_prompt(self):
        prompt = ""
        if self.system_message:
            prompt += self.system_template.format(system_message=self.system_message)
        
        for _, content in self.messages:
            prompt += content
            
        prompt += "<start_of_turn>model\n"
        return prompt
    