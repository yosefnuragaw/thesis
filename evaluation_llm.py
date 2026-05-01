from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import argparse
import torch
from transformers import HfArgumentParser, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
from vllm import LLM, SamplingParams
import os

from models.prompts import PromptProcessor
from utils import set_seed


@dataclass
class ScriptArguments:
    id: Optional[str] = field(default="baseline", metadata={"help": "Run id"})
    model_name_or_path: Optional[str] = field(default="google/gemma-3-1b-it", metadata={"help": "Model Answer Folder"})
    judge_name: Optional[str] = field(default="openai/gpt-oss-20b", metadata={"help": "Judge Model id"})
    behavior: Optional[str] = field(default="power-seeking", metadata={"help": "the behavior"})
    layer: Optional[List[int]] = field(default_factory=lambda: list(range(26)), metadata={"help": "the layer the steering vector extracted from"})
    multipliers: Optional[List[float]] = field(default_factory=lambda: [2., 1.5, 1., 0.5, -0.5, -1., -1.5, -2.], metadata={"help": "the multipliers evaluated"})
    vec_dir: Optional[str] = field(default="vector/power-seeking_gemma-3", metadata={"help": "Directory where .pt vectors are saved"})
    answer_dir: Optional[str] = field(default="generation_results/gemma3-1b", metadata={"help": "Directory where answers are saved"})
    eval_epoch: Optional[int] = field(default=18, metadata={"help": "Which epoch's vector to load"})
    batch_size: Optional[int] = field(default=256, metadata={"help": "Batch size for the judge pipeline"})


def init_judge(model_name: str) -> tuple[LLM, AutoTokenizer]:
    llm = LLM(
        model=model_name,
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=4096,
        gpu_memory_utilization=0.9,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return llm, tokenizer


def read_answers(behavior: str, path: str) -> List[Dict[str, str]]:
    dataset = load_dataset("csv", data_files=path, split='train')
    prompts = []
    if behavior == 'jailbreak':
        for row in dataset:
            prompts.append({
                'behavior': behavior,
                'question': row.get('questions', row.get('question', '')),
                'answer': row.get('answers', row.get('answer', '')),
                'positive_example': None,
                'negative_example': None
            })
    else:
        for row in dataset:
            if row.get('matching') == 'A':
                pos, neg = row.get('A', ''), row.get('B', '')
            else:
                pos, neg = row.get('B', ''), row.get('A', '')

            prompts.append({
                'behavior': behavior,
                'question': row.get('questions', row.get('question', '')),
                'answer': row.get('answers', row.get('answer', '')),
                'positive_example': pos,
                'negative_example': neg
            })
    return prompts


def evaluate_batch(
    llm: LLM,
    tokenizer: AutoTokenizer,
    factory: PromptProcessor,
    prompts: List[str],
    desc: str,
) -> float:

    sampling_params = SamplingParams(
        max_tokens=1024,
        min_tokens=1,
        temperature=1.0,
        top_p=1.0,
        repetition_penalty=1.0,
    )

    chat_prompts = []
    for p in prompts:
        messages = [{"role": "user", "content": p}]
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        chat_prompts.append(formatted_prompt)

    outputs = llm.generate(chat_prompts, sampling_params)

    for out in tqdm(outputs, desc=f"Parsing {desc} Results"):
        generated_text = out.outputs[0].text
        factory.read_response(generated_text)

    valid_scores = [s for s in factory.score if s > 0]
    if not valid_scores:
        return 0.0
    return sum(valid_scores) / len(valid_scores)


def get_model_dir(model_name_or_path: str) -> str:
    if 'gemma' in model_name_or_path:
        return 'gemma'
    elif 'llama' in model_name_or_path:
        return 'llama'
    else:
        return 'mistral'


def main(baseline: bool, args: ScriptArguments) -> None:
    llm, tokenizer = init_judge(args.judge_name)

    accuracy_likert: Dict[float, float] = {}
    coherence_likert: Dict[float, float] = {}
    model_dir = get_model_dir(args.model_name_or_path)

    # derive reasoning dir from answer_dir
    base_dir = os.path.dirname(args.answer_dir.rstrip("/"))  # e.g. "caa/"
    reasoning_base = os.path.join(base_dir, "reasoning", args.behavior, model_dir)

    if baseline:
        file_path = f"{args.answer_dir}/results_{args.behavior}_{args.model_name_or_path.replace('/', '_')}_{args.behavior}-baseline.csv"
        datasets = {0: read_answers(behavior=args.behavior, path=file_path)}
    else:
        datasets = {}
        for multiplier in args.multipliers:
            file_path = f"{args.answer_dir}/results_{args.behavior}_{args.id}_{multiplier}.csv"
            datasets[multiplier] = read_answers(behavior=args.behavior, path=file_path)

    for mul, dataset in datasets.items():
        if not dataset:
            continue

        print(f"[Evaluating Multiplier:] {mul}")

        coherence_factory = PromptProcessor()
        behavior_factory = PromptProcessor()
        coherence_raw_prompts = []
        behavior_raw_prompts = []

        for row in dataset:
            coherence_raw_prompts.append(coherence_factory.produce_prompt(
                behavior="utility",
                question=row['question'],
                answer=row['answer'],
                positive_example=None,
                negative_example=None
            ))
            behavior_raw_prompts.append(behavior_factory.produce_prompt(
                behavior=args.behavior,
                question=row['question'],
                answer=row['answer'],
                positive_example=row['positive_example'],
                negative_example=row['negative_example']
            ))

        c_score = evaluate_batch(llm, tokenizer, coherence_factory, coherence_raw_prompts, "Utility")
        coherence_likert[mul] = c_score
        sub_dir = args.model_name_or_path.replace('/', '_') if mul == 0 else args.id
        coherence_factory.save(os.path.join(reasoning_base, sub_dir, f"reasoning_utility_mul_{mul}.csv"))

        b_score = evaluate_batch(llm, tokenizer, behavior_factory, behavior_raw_prompts, "Behavior")
        accuracy_likert[mul] = b_score
        behavior_factory.save(os.path.join(reasoning_base, sub_dir, f"reasoning_behavior_mul_{mul}.csv"))

        print(f'\n[Multiplier {mul} Complete]')
        print(f'Accuracy Likert (Scale 5): {accuracy_likert[mul]:.2f}')
        print(f'Utility Likert (Scale 5): {coherence_likert[mul]:.2f}')

    print(f'\nFinal Accuracy Likert: {accuracy_likert}')
    print(f'Final Utility Likert:  {coherence_likert}')


if __name__ == "__main__":
    set_seed(seed=42)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, required=True, help="Path to your YAML config file")
    parser.add_argument("--baseline", action='store_true', help="Run only the baseline (multiplier 0)")
    args, remaining = parser.parse_known_args()

    hf_parser = HfArgumentParser(ScriptArguments)
    if args.config.endswith(".yaml"):
        script_args = hf_parser.parse_yaml_file(yaml_file=args.config, allow_extra_keys=True)[0]
    elif args.config.endswith(".json"):
        script_args = hf_parser.parse_json_file(json_file=args.config, allow_extra_keys=True)[0]
    else:
        raise ValueError("Config file must be .yaml or .json")

    main(baseline=args.baseline, args=script_args)