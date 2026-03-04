from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import argparse
import torch
from transformers import HfArgumentParser
from datasets import load_dataset
from tqdm import tqdm  
import wandb 

from flow_judge.metrics import CustomMetric, RubricItem
from flow_judge.flow_judge import EvalInput, FlowJudge
from flow_judge.models import Hf

from models.prompts import EVALUATION_MAP
from utils import set_seed

@dataclass
class ScriptArguments:
    id: Optional[str] = field(default="baseline", metadata={"help": "Run id"})
    model_name_or_path: Optional[str] = field(default="google/gemma-3-1b-it", metadata={"help": "Model Answer Folder"})
    behavior: Optional[str] = field(default="power-seeking", metadata={"help": "the behavior"})
    layer: Optional[List[int]] = field(default_factory=lambda: list(range(26)), metadata={"help": "the layer the steering vector extracted from"})
    multipliers: Optional[List[float]] = field(default_factory=lambda: [2,1.5,1,-1,-1.5,-2], metadata={"help": "the layer the steering vector extracted from"})
    vec_dir: Optional[str] = field(default="vector/power-seeking_gemma-3", metadata={"help": "Directory where .pt vectors are saved"})
    answer_dir: Optional[str] = field(default="generation_results/gemma3-1b", metadata={"help": "Directory where answers are are saved"})
    eval_epoch: Optional[int] = field(default=18, metadata={"help": "Which epoch's vector to load"})
        
def read_answers(behavior: str, path: str) -> List[Dict[str,str]]:
    dataset = load_dataset("csv", data_files=path, split='train')
    prompts = []
    for row in dataset:
        if row['matching'] == 'A':
            pos, neg = row['A'], row['B']
        else:
            pos, neg = row['B'], row['A']
        
        prompts.append({
            'behavior': behavior, 
            'question': row['questions'], 
            'answer': row['answers'], 
            'positive_example': pos, 
            'negative_example': neg
        })
    return prompts

def main(baseline: bool, args: ScriptArguments) -> None:
    model = Hf(
    model_kwargs={
        "attn_implementation": "flash_attention_2",
        "use_cache": True  # <-- Add this here
    }
)
    accuracy_likert: Dict[float, float] = {}
    coherence_likert: Dict[float, float] = {}
    
    if baseline:
        file_path = f"{args.answer_dir}/results_{args.behavior}_{args.model_name_or_path.replace('/', '_')}_{args.behavior}-baseline.csv"
        datasets = {0: read_answers(behavior=args.behavior, path=file_path)}  
    else:
        datasets = {}
        for multiplier in args.multipliers:
            file_path = f"{args.answer_dir}/results_{args.behavior}_{args.model_name_or_path.replace('/', '_')}_{args.id}_{multiplier}.csv"
            datasets[multiplier] = read_answers(behavior=args.behavior, path=file_path)
            
    coh_criteria, coh_rubric = EVALUATION_MAP['coherence']
    coherence_metric = CustomMetric(
        name="coherence",
        criteria=coh_criteria,
        rubric=[RubricItem(score=k, description=v) for k, v in coh_rubric.items()],
        required_inputs=["question"],
        required_output="answer"
    )
    coherence_judge = FlowJudge(metric=coherence_metric, model=model)

    beh_criteria, beh_rubric = EVALUATION_MAP[args.behavior]
    
    for mul, dataset in datasets.items():
        if not dataset:
            continue

        print(f"\n[Multiplier:] {mul} \n")
        
        coherence_results = []
        for row in tqdm(dataset, desc="Coherence Evaluation"):
            eval_input = EvalInput(
                inputs=[{"question": row['question']}], 
                output={"answer": row['answer'] if row['answer'] else ''}
            )
            res = coherence_judge.evaluate(eval_input)
            coherence_results.append(res)
            
        coherence_likert[mul] = sum(res.score for res in coherence_results) / len(coherence_results)

        behavior_results = []
        for row in tqdm(dataset, desc="Behavior Evaluation"):
            
            beh_rubric_items = []
            for key, val in beh_rubric.items():
                if key == 5:
                    val = val.replace("{POSITIVE_EXAMPLE}", row['positive_example'])
                elif key == 1:
                    val = val.replace("{NEGATIVE_EXAMPLE}", row['negative_example'])
                beh_rubric_items.append(RubricItem(score=key, description=val))

            behavior_metric = CustomMetric(
                name=args.behavior,
                criteria=beh_criteria,
                rubric=beh_rubric_items,
                required_inputs=["question"], 
                required_output="answer"
            )
            
            behavior_judge = FlowJudge(metric=behavior_metric, model=model)

            eval_input = EvalInput(
                inputs=[{"question": row['question']}], 
                output={"answer": row['answer'] if row['answer'] else ''}
            )
            
            res = behavior_judge.evaluate(eval_input)
            behavior_results.append(res)
        

        clean_ans = [res.answer for res in behavior_results if res.answer > 0]
        accuracy_likert[mul] = sum(clean_ans) / len(clean_ans)

        print(f'\n[Accuracy Likert (Scale 5):] {accuracy_likert}')
        print(f'[Coherence Likert (Scale 5):] {coherence_likert}')

if __name__ == "__main__":
    set_seed(seed=11)
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

