


from dataclasses import dataclass, field
import pandas as pd
from tqdm import tqdm
from typing import Dict, List, Optional
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    pipeline,
    HfArgumentParser
)
import os
from datasets import load_dataset
from pathlib import Path
import argparse

from models.prompts import SYSTEM_PROMPT
from utils import set_seed
from evaluation import init_model
from models.dataset import PromptDataset



@dataclass
class ScriptArguments:
    """
    The arguments for the LLM as a judge eval scrip,
    """
    model_name_or_path: Optional[str] = field(
        default="google/gemma-3-1b-it",
        metadata={"help": "Model Answer Folder"}
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
        default="/kaggle/working/BiPO/vector/power-seeking_gemma-3",
        metadata={"help": "Directory where .pt vectors are saved"}
    )
    eval_epoch: Optional[int] = field(default=18, metadata={"help": "Which epoch's vector to load"})
    max_new_tokens: Optional[int] = field(default=200, metadata={"help": "Max new generation tokens"})
    temperature: Optional[float] = field(default=0.7, metadata={"help": "LLM generation temperature"})

def read_dataset(
        behavior: str,
        tokenizer: AutoTokenizer
    )-> Dict[str, List]:

    path = f"./data/{behavior}/test_infer.csv"
    if not os.path.exists(path):
         raise FileNotFoundError(f"Data file not found: {path}")
    dataset = load_dataset("csv", data_files=path, split='train')

    results:Dict[str, str] = {'prompts':[], 'questions':[], 'A':[], 'B':[], 'answers':[], 'matching':[]}
  
    for idx, row in enumerate(dataset):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": row['question']},
        ]

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        try:
            pos, neg = row['A'], row['B']
        except:
            raise ValueError(f'Expected column A and B in row {idx}')
        
        results['questions'].append(row['question'])
        results['prompts'].append(prompt)
        results['A'].append(pos)
        results['B'].append(neg)
        results['matching'].append(row['matching'])

    return results

def generate_answers(
    model: AutoModelForCausalLM, 
    tokenizer: AutoTokenizer, 
    dataset: Dict[str, List], 
    max_new_tokens: int, 
    temperature: float,
    batch_size: int = 32
) -> Dict[str, List]:

    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto" 
    )

    prompt_loader = PromptDataset(dataset['prompts'])

    results = []
    for output in tqdm(generator(
        prompt_loader,
        max_new_tokens=max_new_tokens,
        min_new_tokens=16,
        do_sample=True,
        temperature=temperature,
        batch_size=batch_size # Crucial for performance
    ), total=len(prompt_loader), desc="Generating Answers"):
        
        generated_text = output[0]["generated_text"]

        # 4. Trimming logic
        if "model" in generated_text:
            trimmed = generated_text[generated_text.find("model") + 5:].strip()
        else:
            trimmed = generated_text
        
        results.append(trimmed)

    dataset['answers'] = results
    return dataset


def main(args: ScriptArguments)->None:
    for multiplier in args.multipliers:
        model, tokenizer = init_model(
            model_name=args.model_name_or_path,
            vec_dir=args.vec_dir,
            epoch=args.eval_epoch,
            layers=args.layer,
            multiplier=multiplier
        )

        dataset = read_dataset(
            behavior=args.behavior,
            tokenizer=tokenizer
        )

        updated_dataset = generate_answers(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            max_new_tokens= args.max_new_tokens, 
            temperature = args.temperature
        )    
        df = pd.DataFrame(updated_dataset)

        # Saving
        output_dir = "generation_results"
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        safe_model_name = args.model_name_or_path.replace("/", "_")
        file_name = f"results_{args.behavior}_{safe_model_name}_{args.layer}_{multiplier}.csv"

        output_path = os.path.join(output_dir, file_name)
        df.to_csv(output_path, index=False)
        print(f"Results saved to: {output_path}")

if __name__ == "__main__":
    set_seed(seed=11)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to your YAML config file")
    args, remaining = parser.parse_known_args()

    hf_parser = HfArgumentParser(ScriptArguments)
    if args.config.endswith(".yaml"):
        script_args = hf_parser.parse_yaml_file(yaml_file=args.config, allow_extra_keys=True)[0]
    elif args.config.endswith(".json"):
        script_args = hf_parser.parse_json_file(json_file=args.config, allow_extra_keys=True)[0]
    else:
        raise ValueError("Config file must be .yaml or .json")

    main(script_args)
    

