


from dataclasses import dataclass, field
import pandas as pd
from typing import Dict, List, Optional
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    pipeline,
    HfArgumentParser
)
import os
import argparse

from models import SYSTEM_PROMPT
from utils import set_seed
from evaluation import init_model


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

    vec_dir: Optional[str] = field(
        default="/kaggle/working/BiPO/vector/power-seeking_gemma-3",
        metadata={"help": "Directory where .pt vectors are saved"}
    )

    eval_epoch: Optional[int] = field(default=18, metadata={"help": "Which epoch's vector to load"})
    max_new_tokens: Optional[int] = field(default=200, metadata={"help": "Max new generation tokens"})
    temperature: Optional[float] = field(default=0.9, metadata={"help": "LLM generation temperature"})

def load_dataset(
        behavior: str,
        tokenizer: AutoTokenizer
    )-> Dict[str, List]:

    path = f"./data/{behavior}/test_infer.csv"
    if not os.path.exists(path):
         raise FileNotFoundError(f"Data file not found: {path}")
    dataset = load_dataset("csv", data_files=path, split='train')

    results:Dict[str, str] = {'prompts':[], 'questions':[], 'A':[], 'B':[], 'answers':[]}
  
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

    return results

def generate_answers(
    model: AutoModelForCausalLM, tokenizer: AutoTokenizer, dataset: Dict[str, List], max_new_tokens: int, temperature: float
    )->Dict[str, List]:

    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=model.device 
    )

    for prompt in dataset['prompt']:
        output = generator(
            prompt,
            max_new_tokens=max_new_tokens,
            min_new_tokens=16,
            do_sample=True,
            temperature=temperature,   
            generation_config=None,   
        )[0]["generated_text"]

        if "model" in output:
            trimmed = output[output.find("model") + 5:].strip()
        else:
            trimmed = output

        dataset['answers'].append(trimmed)
    return dataset

def main(multiplier:float, args: HfArgumentParser )->None:
    
    model, tokenizer = init_model(
        model_name=args.model_name_or_path,
        vec_dir=args.vec_dir,
        epoch=args.eval_epoch,
        layer=args.layer,
        multiplier=multiplier
    )

    dataset = load_dataset(
        behavior=args.behavior,
        tokenizer=tokenizer
    )

    df = generate_answers(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        max_new_tokens= args.max_new_tokens, 
        temperature = args.temperature
    )    
    output_path = f"evaluation_results_{args.behavior}_{args.model_name_or_path}_{multiplier}.csv"
    df.to_csv(output_path, index=False)
    print(f"Results saved successfully to {output_path}")

if __name__ == "__main__":
    set_seed(seed=11)
    parser = argparse.ArgumentParser()
    parser.add_argument("--multiplier", "-m", type=float, required=True, help="Multiplier for steering vector")
    args, remaining = parser.parse_known_args()

    hfargs = HfArgumentParser(ScriptArguments)
    main(args.multiplier, hfargs)
    

