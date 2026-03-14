from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import argparse
import torch
from transformers import HfArgumentParser
from datasets import load_dataset
from tqdm import tqdm  
import wandb 
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    pipeline,
    HfArgumentParser
)



from models.prompts import PromptProcessor
from utils import set_seed
@dataclass
class ScriptArguments:
    id: Optional[str] = field(default="baseline", metadata={"help": "Run id"})
    model_name_or_path: Optional[str] = field(default="google/gemma-3-1b-it", metadata={"help": "Model Answer Folder"})
    id: Optional[str] = field(default="gemma-3-1b-it", metadata={"help": "Model id"})
    judge_name: Optional[str] = field(default="openai/gpt-oss-20b", metadata={"help": "Judge Model id"})
    behavior: Optional[str] = field(default="power-seeking", metadata={"help": "the behavior"})
    layer: Optional[List[int]] = field(default_factory=lambda: list(range(26)), metadata={"help": "the layer the steering vector extracted from"})
    multipliers: Optional[List[float]] = field(default_factory=lambda: [2,1.5,1,0.5,-0.5,-1,-1.5,-2], metadata={"help": "the multipliers evaluated"})
    vec_dir: Optional[str] = field(default="vector/power-seeking_gemma-3", metadata={"help": "Directory where .pt vectors are saved"})
    answer_dir: Optional[str] = field(default="generation_results/gemma3-1b", metadata={"help": "Directory where answers are saved"})
    eval_epoch: Optional[int] = field(default=18, metadata={"help": "Which epoch's vector to load"})
    batch_size: Optional[int] = field(default=256, metadata={"help": "Batch size for the judge pipeline"})

def init_judge(model_name: str) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        use_cache=True,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.warnings_issued = {}
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.config.use_cache = False
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer  

def read_answers(behavior: str, path: str) -> List[Dict[str, str]]:
    dataset = load_dataset("csv", data_files=path, split='train')
    prompts = []
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
    pipe: pipeline, 
    tokenizer: AutoTokenizer, 
    factory: PromptProcessor, 
    prompts: List[str], 
    desc: str
) -> float:
    
    chat_prompts = []
    for p in prompts:
        messages = [{"role": "user", "content": p}]
        formatted_prompt = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        chat_prompts.append(formatted_prompt)


    outputs = pipe(
         chat_prompts,
        max_new_tokens=4096, 
        temperature=0.2,   
        top_p=0.9,
        do_sample=True,        
        return_full_text=False,
        batch_size=pipe._batch_size
    )

    for out in tqdm(outputs, desc=f"Parsing {desc} Results"):
        generated_text = out[0]['generated_text']
        factory.read_response(generated_text)
    
    valid_scores = [s for s in factory.score if s > 0]
    if not valid_scores:
        return 0.0
    return sum(valid_scores) / len(valid_scores)

def main(baseline: bool, args: ScriptArguments) -> None:
    model, tokenizer = init_judge(args.judge_name)
    
    judge_pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        device_map="auto"
    )

    accuracy_likert: Dict[float, float] = {}
    coherence_likert: Dict[float, float] = {}

    if baseline:
        file_path = f"{args.answer_dir}/results_{args.behavior}_{args.model_name_or_path.replace('/', '_')}_{args.behavior}-baseline.csv"
        datasets = {0: read_answers(behavior=args.behavior, path=file_path)}  
    else:
        datasets = {}
        for multiplier in args.multipliers:
            file_path = f"{args.answer_dir}/results_{args.behavior}_{args.id}_{multiplier}_{args.eval_epoch}.csv"
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
            c_prompt = coherence_factory.produce_prompt(
                behavior="utility", 
                question=row['question'], 
                answer=row['answer'],
                positive_example = None,
                negative_example = None
            )
            coherence_raw_prompts.append(c_prompt)
            
            b_prompt = behavior_factory.produce_prompt(
                behavior=args.behavior, 
                question=row['question'], 
                answer=row['answer'],
                positive_example=row['positive_example'],
                negative_example=row['negative_example']
            )
            behavior_raw_prompts.append(b_prompt)

        c_score = evaluate_batch(judge_pipe, tokenizer, coherence_factory, coherence_raw_prompts, "Utility")
        coherence_likert[mul] = c_score
        coherence_factory.save(f"reasoning/{args.id}/reasoning_{args.behavior}_utility_mul{mul}.csv")

        b_score = evaluate_batch(judge_pipe, tokenizer, behavior_factory, behavior_raw_prompts, "Behavior")
        accuracy_likert[mul] = b_score
        behavior_factory.save(f"reasoning/{args.id}/reasoning_{args.behavior}_behavior_mul{mul}.csv")

        print(f'\n[Multiplier {mul} Complete]')
        print(f'Accuracy Likert (Scale 5): {accuracy_likert[mul]:.2f}')
        print(f'Utility Likert (Scale 5): {coherence_likert[mul]:.2f}')

    print(f'Accuracy Likert (Scale 5): {accuracy_likert}')
    print(f'Utility Likert (Scale 5): {coherence_likert}')



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