from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import argparse
import torch
from transformers import pipeline, Pipeline,HfArgumentParser
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
    multipliers: Optional[List[float]] = field(
        default_factory=lambda: [-2,-1.5,-1,0,1,1.5,2], 
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

def read_answers(behavior: str, path: str)->List[Dict[str,str]]:
    dataset = load_dataset("csv", data_files=path, split='test')
    prompts = []

    for row in dataset:
        if row['matching'] == 'A':
            pos = row['A']
            neg = row['B']
        else:
            pos = row['B']
            neg = row['A']
        
        prompts.append({'behavior': behavior, 'question':row['questions'] , 'answer': row['answers'], 'positive_example': pos, 'negative_example': neg})

    return prompts

def eval_likert(pipeline: Pipeline, dataset: List[Dict[str,str]], multipliers:List[float])->Dict[float, float]:

    for metadata in dataset:
        messages = [
            {"role": "user", "content": PromptFactory.produce_accuracy_prompt(**metadata)},
        ]
        ans = pipeline(messages)
    pass

def eval_coherence(pipeline: Pipeline, dataset: List[Dict[str,str]], multipliers:List[float])->Dict[float, float]:
    for metadata in dataset:
        messages = [
            {"role": "user", "content": PromptFactory.produce_coherence_prompt(**metadata)},
        ]
        ans = pipeline(messages)
    pass

def main(task: str, judge_name_or_path: str, verbose: bool)->None:
    pipe = pipeline("text-generation", model=judge_name_or_path, trust_remote_code=True, device_map="auto")
    pass    

if __name__ == "__main__":
    set_seed(seed=11)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, required=True, help="Path to your YAML config file")
    parser.add_argument("--verbose", "-v", type=bool, required=False, default=True, help="Visualize eval progress")
    parser.add_argument("--task", "-t", type=str, required=False, default="both", help="Visualize eval progress")
    args, remaining = parser.parse_known_args()

    hf_parser = HfArgumentParser(ScriptArguments)
    if args.config.endswith(".yaml"):
        script_args = hf_parser.parse_yaml_file(yaml_file=args.config, allow_extra_keys=True)[0]
    elif args.config.endswith(".json"):
        script_args = hf_parser.parse_json_file(json_file=args.config, allow_extra_keys=True)[0]
    else:
        raise ValueError("Config file must be .yaml or .json")
    
    main()

