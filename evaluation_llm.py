from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import argparse
import torch
from transformers import pipeline
from datasets import load_dataset

from models.prompts import PromptFactory
from utils import set_seed



@dataclass
class ScriptArguments:
    """
    The arguments for the LLM as a judge eval scrip,
    """
    judge_name_or_path: Optional[str] = field(
        default="google/gemma-3-1b-it",
        metadata={"help": "The model checkpoint for weights initialization."}
    )
    behavior: Optional[str] = field(default="power-seeking", metadata={"help": "the behavior"})
    layer: Optional[List[int]] = field(
        default_factory=lambda: list(range(26)), 
        metadata={"help": "the layer the steering vector extracted from"}
    )
    vec_dir: Optional[str] = field(
        default="vector/power-seeking_gemma-3",
        metadata={"help": "Directory where .pt vectors are saved"}
    )
    answer_path: Optional[str] = field(
        default="/vector/power-seeking_gemma-3",
        metadata={"help": "Directory where prompts are are saved"}
    )
    eval_epoch: Optional[int] = field(default=18, metadata={"help": "Which epoch's vector to load"})

def read_answers(behavior: str, path: str)->List[str]:
    file_path = path
    dataset = load_dataset("csv", data_files=file_path, split='train')
    prompts = []

    for row in dataset:
        prompts.append(produce_prompt(behavior, row['questions'],row['answers'],row['A'], row['B']))
    return prompts

def eval_likert(judge_name_or_path: str)->Dict[str, float]:
    pipe = pipeline("text-generation", model=judge_name_or_path, trust_remote_code=True)
    messages = [
        {"role": "user", "content": "Who are you?"},
    ]
    ans = pipe(messages)
    pass

def main()->None:
    
    pass    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    main()

