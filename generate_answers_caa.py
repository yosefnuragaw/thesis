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
import torch
from datasets import load_dataset
from pathlib import Path
import argparse

from models.prompts import SYSTEM_PROMPT
from utils import set_seed
from models.dataset import PromptDataset
from models.model import CAABlockWrapper


@dataclass
class ScriptArguments:
    id: Optional[str] = field(default="baseline", metadata={"help": "Run id"})
    model_name_or_path: Optional[str] = field(
        default="google/gemma-3-1b-it",
        metadata={"help": "The model checkpoint"}
    )
    behavior: Optional[str] = field(default="power-seeking", metadata={"help": "the behavior"})
    layer: Optional[List[int]] = field(
        default_factory=lambda: list(range(32)),
        metadata={"help": "Layers to apply steering"}
    )
    total_layer: Optional[int] = field(default=32, metadata={"help": "LLM total number of layers"})
    multipliers: Optional[List[float]] = field(
        default_factory=lambda: [2., 1.5, 1., 0.5, -0.5, -1., -1.5, -2.],
        metadata={"help": "Steering multipliers to evaluate"}
    )
    vec_dir: Optional[str] = field(default=None, metadata={"help": "Directory where .pt vectors are saved"})
    answer_dir: Optional[str] = field(default="generation_results", metadata={"help": "Directory where CSVs will be saved"})
    apply_type: Optional[str] = field(default="layer", metadata={"help": "layer or sequence"})
    max_new_tokens: Optional[int] = field(default=200, metadata={"help": "Max new generation tokens"})
    temperature: Optional[float] = field(default=0.7, metadata={"help": "LLM generation temperature"})


def init_model(
        model_name: str,
        vec_dir: str,
        layers: List[int],
        apply_type: str,
        total_layer: int = 32,
        baseline: bool = False,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        attn_implementation="sdpa",
        # attn_implementation="flash_attention_2",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.warnings_issued = {}
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    for layer in range(total_layer):
        if baseline:
            vec = torch.zeros(model.config.hidden_size, dtype=model.dtype)
        else:
            vec_path = os.path.join(vec_dir, f"pos_layer_{layer}.pt")
            neg_path = os.path.join(vec_dir, f"neg_layer_{layer}.pt")
            if os.path.exists(vec_path) and os.path.exists(neg_path):
                vec_pos = torch.load(vec_path, map_location="cpu")
                vec_neg = torch.load(neg_path, map_location="cpu")
                vec = (vec_pos - vec_neg).to(model.dtype)  # CAA direction
            else:
                vec = torch.zeros(model.config.hidden_size, dtype=model.dtype)

        model.model.layers[layer] = CAABlockWrapper(
            model.model.layers[layer],
            hidden_dim=model.config.hidden_size,
            vec=vec,
            apply_type=apply_type,
        )
        model.model.layers[layer].extract(False)  # inference mode

    model.config.use_cache = False
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def read_dataset(behavior: str, tokenizer: AutoTokenizer, multiplier: float = 0) -> Dict[str, List]:
    path = f"./data/{behavior}/test_infer.csv"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")
    dataset = load_dataset("csv", data_files=path, split='train')

    results: Dict[str, List] = {'prompts': [], 'questions': [], 'A': [], 'B': [], 'answers': [], 'matching': []}

    for idx, row in enumerate(dataset):
        user_text = row.get('question', "")
        if user_text is not None:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": str(user_text)},
            ]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            results['questions'].append(row['question'])
            results['prompts'].append(prompt)

            if behavior != 'jailbreak':
                try:
                    pos, neg = row['A'], row['B']
                except:
                    raise ValueError(f'Expected column A and B in row {idx}')
                results['A'].append(pos)
                results['B'].append(neg)
                if multiplier < 0:
                    results['matching'].append('B' if row['matching'] == 'A' else 'A')
                else:
                    results['matching'].append(row['matching'])
            else:
                results['A'].append('')
                results['B'].append('')
                results['matching'].append(row['not_matching'] if multiplier < 0 else row['matching'])

    return results


def generate_answers(
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        dataset: Dict[str, List],
        max_new_tokens: int,
        temperature: float,
        batch_size: int = 128
) -> Dict[str, List]:

    generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")
    prompt_loader = PromptDataset(dataset['prompts'])

    results = []
    for output in tqdm(generator(
        prompt_loader,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=1.0,
        num_beams=1,
        batch_size=batch_size,
        return_full_text=False,
        generation_config=None
    ), total=len(prompt_loader), desc="Generating Answers"):
        results.append(output[0]["generated_text"].strip())

    dataset['answers'] = results
    del dataset['prompts']
    return dataset


def save(output_dir: str, file_name: str, df: pd.DataFrame) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = os.path.join(output_dir, file_name)
    df.to_csv(output_path, index=False)
    print(f"Results saved to: {output_path}")


def main(baseline: bool, args: ScriptArguments) -> None:
    model, tokenizer = init_model(
        model_name=args.model_name_or_path,
        vec_dir=args.vec_dir,
        layers=args.layer,
        apply_type=args.apply_type,
        total_layer=args.total_layer,
        baseline=baseline,
    )

    if not baseline:
        for multiplier in args.multipliers:
            for idx in args.layer:
                if isinstance(model.model.layers[idx], CAABlockWrapper):
                    model.model.layers[idx].set_multiplier(multiplier)

            dataset = read_dataset(behavior=args.behavior, tokenizer=tokenizer, multiplier=multiplier)
            updated_dataset = generate_answers(
                model=model, tokenizer=tokenizer, dataset=dataset,
                max_new_tokens=args.max_new_tokens, temperature=args.temperature
            )
            df = pd.DataFrame(updated_dataset)
            file_name = f"results_{args.behavior}_{args.id}_{multiplier}.csv"
            save(args.answer_dir, file_name, df)
    else:
        dataset = read_dataset(behavior=args.behavior, tokenizer=tokenizer)
        updated_dataset = generate_answers(
            model=model, tokenizer=tokenizer, dataset=dataset,
            max_new_tokens=args.max_new_tokens, temperature=args.temperature
        )
        df = pd.DataFrame(updated_dataset)
        safe_model_name = args.model_name_or_path.replace("/", "_")
        file_name = f"results_{args.behavior}_{safe_model_name}_baseline.csv"
        save(args.answer_dir, file_name, df)


if __name__ == "__main__":
    set_seed(seed=42)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--baseline", action='store_true')
    args, remaining = parser.parse_known_args()

    hf_parser = HfArgumentParser(ScriptArguments)
    if args.config.endswith(".yaml"):
        script_args = hf_parser.parse_yaml_file(yaml_file=args.config, allow_extra_keys=True)[0]
    elif args.config.endswith(".json"):
        script_args = hf_parser.parse_json_file(json_file=args.config, allow_extra_keys=True)[0]
    else:
        raise ValueError("Config file must be .yaml or .json")

    main(baseline=args.baseline, args=script_args)