import argparse
from dataclasses import dataclass, field
from typing import List, Optional
import os
import wandb
from torch.utils.data import Dataset, DataLoader
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser

from trl import BiPOTrainer, DPOConfig, BiPOTrainerEXP

from utils import get_data, print_trainable_parameters, set_seed
from models.model import (
    BlockWrapper,
    MODEL_TEMPLATE_MAP,
    )
from models.scheduler import QuantileSchedulerCallback



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
        default_factory=lambda: list(range(32)), 
        metadata={"help": "the layer the steering vector extracted from"}
    )
    pos_weight: Optional[List[int]] = field(
        default_factory=lambda: list(1 for x in range(32)), 
        metadata={"help": "Weight layer"}
    )
    neg_weight: Optional[List[int]] = field(
        default_factory=lambda: list(1 for x in range(32)), 
        metadata={"help": "Weight layer"}
    )
    total_layer: Optional[int] = field(default=200, metadata={"help": "LLM total number of layers"})

    multipliers: Optional[List[float]] = field(
        default_factory=lambda: [2,1.5,1.,0.5,-0.5,-1.,-1.5,-2.], 
        # default_factory=lambda: [2], 
        metadata={"help": "the layer the steering vector extracted from"}
    )
    vec_dir: Optional[str] = field(
        default= None,
        metadata={"help": "Directory where .pt vectors are saved"}
    )
    gate_dir: Optional[str] = field(
        default= None,
        metadata={"help": "Directory where .pt vectors are saved"}
    )
    answer_dir: Optional[str] = field(
        default="/kaggle/working/BiPO/vector/power-seeking_gemma-3",
        metadata={"help": "Directory where .csw will be saved"}
    )
    eval_epoch: Optional[int] = field(default=18, metadata={"help": "Which epoch's vector to load"})
    max_new_tokens: Optional[int] = field(default=200, metadata={"help": "Max new generation tokens"})
    temperature: Optional[float] = field(default=0.1, metadata={"help": "LLM generation temperature"})
    gate_function: Optional[str] = field(default=None, metadata={"help" : "mask gate activation function None | sigmoid | tanh"})
    skip: Optional[str] = field(default=None, metadata={"help" : "cosine scaler None | distance | similarity"})
    k1: Optional[float] = field(default=0., metadata={"help": "Quantile for selecting top-K neuron"})



if __name__ == "__main__":
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

    run_name = f"{script_args.behavior}-{script_args.id}"

    os.environ["WANDB_NAME"] = run_name
    set_seed(seed=42)
    
    # 2. Determine Template Name
    if script_args.model_name_or_path not in MODEL_TEMPLATE_MAP:
        print(f"Warning: {script_args.model_name_or_path} not in supported list: {list(MODEL_TEMPLATE_MAP.keys())}")
    template_name = MODEL_TEMPLATE_MAP.get(script_args.model_name_or_path, 'llama-2')

    print(f"Loaded config from {args.config}")
    print(f"[Behavior:] {script_args.behavior} | [Layer:] {script_args.layer} | [Model:] {script_args.model_name_or_path} | [Experiment:] {script_args.experiment} | [Moving:] {script_args.moving} | [Skip:] {script_args.skip} [k1:] {script_args.k1} | [Scale:] {script_args.scale}")

    # 3. Load & Configure Models
    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.warnings_issued = {}
    model.config.use_cache = False

    # Inject BlockWrappers
    for layer in script_args.layer:
        model.model.layers[layer] = BlockWrapper(model.model.layers[layer], hidden_dim=model.config.hidden_size, gate_function=script_args.gate_function, skip= script_args.skip, k1 = script_args.k1)

    if script_args.ignore_bias_buffers:
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    # Load Reference Model
    model_ref = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token

    # 4. Freeze/Unfreeze Logic
    print('Freezing base model parameters...') 
    for param in model_ref.parameters():
        param.requires_grad = False
    for param in model.parameters():
        param.requires_grad = False

    
    print('Unfreezing steering vectors...')
    for layer in script_args.layer:
        model.model.layers[layer].vec.requires_grad = True 
        

    # 5. Load Datasets
    train_dataset = get_data(tokenizer = tokenizer, behavior=script_args.behavior, train=True) 
    test_dataset = get_data(tokenizer = tokenizer,behavior=script_args.behavior, train=False) 


    # 6. Initialize Training Args
    training_args = DPOConfig(
        output_dir="placeholder",
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        per_device_eval_batch_size=script_args.per_device_eval_batch_size,
        num_train_epochs=script_args.num_train_epochs,
        logging_steps=script_args.logging_steps,
        save_strategy="no",
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        gradient_checkpointing=script_args.gradient_checkpointing,
        learning_rate=script_args.learning_rate,
        eval_strategy="epoch",
        report_to=script_args.report_to,
        lr_scheduler_type=script_args.lr_scheduler_type,
        warmup_steps=script_args.warmup_steps,
        optim=script_args.optimizer_type,
        bf16=False,
        remove_unused_columns=False,
        max_prompt_length=script_args.max_prompt_length,
        max_length=script_args.max_length,
        beta=script_args.beta,
    )

    if script_args.experiment:
        dpo_trainer = BiPOTrainerEXP(
            model=model,
            ref_model=model_ref,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset={'test_dataset_add': test_dataset, 'test_dataset_sub': test_dataset},
            processing_class=tokenizer,
            behavior=script_args.behavior,
            layer=script_args.layer,
            name=template_name,
            quantile=script_args.quantile,
            filter_step=script_args.filter_step,
            pipeline = script_args.pipeline,
            masking_type = script_args.masking_type,
            num_layer = script_args.total_layer,
            moving= script_args.moving,
            scale = script_args.scale
        )

        if script_args.quantile_scheduler:
            print(f"[Scheduler:] {script_args.quantile_scheduler_type} | [Start:] {script_args.quantile} ")
            scheduler_callback = QuantileSchedulerCallback(
                    start_val=script_args.quantile, 
                    schedule_type= script_args.quantile_scheduler_type
                )
            dpo_trainer.add_callback(scheduler_callback)

    else:
        dpo_trainer = BiPOTrainer(
            model=model,
            ref_model=model_ref,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset={'test_dataset_add': test_dataset, 'test_dataset_sub': test_dataset},
            processing_class=tokenizer,
            behavior=script_args.behavior,
            layer=script_args.layer,
            name=template_name,
        )

    print_trainable_parameters(model)
    dpo_trainer.train()