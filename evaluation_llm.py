from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import argparse
import torch
from transformers import HfArgumentParser
from datasets import load_dataset
from tqdm import tqdm  

from flow_judge.metrics import CustomMetric, RubricItem
from flow_judge.flow_judge import EvalInput, FlowJudge
from flow_judge.models import Vllm 

from models.prompts import EVALUATION_MAP
from utils import set_seed

@dataclass
class ScriptArguments:
    """
    The arguments for the LLM as a judge eval scrip,
    """
    id: Optional[str] = field(
        default="baseline",
        metadata={"help": "Run id"}
    )
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
        default_factory=lambda: [-2,-1.5,-1,1,1.5,2], 
        metadata={"help": "the layer the steering vector extracted from"}
    )
    vec_dir: Optional[str] = field(
        default="vector/power-seeking_gemma-3",
        metadata={"help": "Directory where .pt vectors are saved"}
    )
    answer_dir: Optional[str] = field(
        default="generation_results/gemma3-1b",
        metadata={"help": "Directory where answers are are saved"}
    )
    eval_epoch: Optional[int] = field(default=18, metadata={"help": "Which epoch's vector to load"})

def judge(model: Hf, domain:str, row:Dict[str,str])->FlowJudge:
    criteria, rubric = EVALUATION_MAP[domain]

    rubric_items = []
    if domain == 'coherence':
        for key,val in rubric.items():
            rubric_items.append(RubricItem(score = key, description = val))
    else:
        for key,val in rubric.items():
            if key == 5:
                val = val.format(POSITIVE_EXAMPLE=row['positive_example'])
            elif key == 1:
                val = val.format(NEGATIVE_EXAMPLE=row['negative_example'])

            rubric_items.append(RubricItem(score = key, description = val))
        

    required_input = ["question"]
    required_output = "answer"

    domain_coverage = CustomMetric(
        name="behavior",
        criteria=criteria,
        rubric= rubric_items,
        required_inputs=required_input,
        required_output=required_output
    )

    eval_input = EvalInput(
        inputs=[{"question": row['question']}],
        output={"answer": row['answer']}
    )


    judge = FlowJudge(metric=domain_coverage, model=model)
    result = judge.evaluate(eval_input)

    return {'feedback': result.feedback, 'score': result.score}
        

def read_answers(behavior: str, path: str)->List[Dict[str,str]]:
    dataset = load_dataset("csv", data_files=path, split='train')
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

def main(baseline:bool, args:ScriptArguments)->None:
    model = Vllm()
    accuracy_likert: Dict[float, float] = {}
    coherence_likert: Dict[float, float] = {}
    
    
    if baseline:
        file_path = f"{args.answer_dir}/results_{args.behavior}_{args.model_name_or_path.replace("/", "_")}_{args.behavior}-baseline.csv"
        datasets = {0: read_answers(behavior=args.behavior, path = file_path)}  
        accuracy_likert[0] = 0
        coherence_likert[0] = 0

    else:
        datasets = {}
        for multiplier in args.multipliers:
            file_path = f"{args.answer_dir}/results_{args.behavior}_{args.model_name_or_path.replace("/", "_")}_{args.id}_{multiplier}.csv"
            datasets[multiplier] = read_answers(behavior=args.behavior, path = file_path)

            accuracy_likert[multiplier] = 0
            coherence_likert[multiplier] = 0
        
    
    for mul, dataset in datasets.items():
        count = 0
        for row in tqdm(dataset, desc=f"Processing {mul}"):
            result_accuracy = judge(model, args.behavior, row)
            result_coherence = judge(model, 'coherence', row)
            accuracy_likert[mul] += result_accuracy['score']
            coherence_likert[mul] += result_coherence['score']
            count += 1

        accuracy_likert[mul] /= count
        coherence_likert[mul] /= count

    print(f'[Accuracy Likert (Scale 5):] {accuracy_likert}')
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

