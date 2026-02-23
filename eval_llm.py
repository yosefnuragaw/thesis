


from dataclasses import dataclass, field
from typing import Dict, List, Optional
import argparse

@dataclass
class ScriptArguments:
    """
    The arguments for the LLM as a judge eval scrip,
    """
    model_name_or_path: Optional[str] = field(
        default="google/gemma-3-1b-it",
        metadata={"help": "Model Answer Folder"}
    )

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
        default="/kaggle/working/BiPO/vector/power-seeking_gemma-3",
        metadata={"help": "Directory where .pt vectors are saved"}
    )

    eval_epoch: Optional[int] = field(default=18, metadata={"help": "Which epoch's vector to load"})
 

def read_answers(path: str)->List[str]:
    pass

def init_judge(judge_name_or_path: str):
    pass

def eval_accuracy()->Dict[str, float]:
    pass

def main()->None:
    
    pass    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    main()

