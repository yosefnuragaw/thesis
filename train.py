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



# --- Arguments ---
@dataclass
class ScriptArguments:
    seed: int = field(default=42, metadata={"help": "Environment Seed"})
    data_root: str = field(default='/kaggle/input/datasets/limbodhiwijaya/phase-1-uba', metadata={"help": "Path to data root"})
    label_dir: str = field(default='Label', metadata={"help": "Directory for labels"})
    checkpoint_dir: str = field(default='checkpoints_modular', metadata={"help": "Directory to save/load checkpoints"})
    
    load_checkpoint: bool = field(default=True, metadata={"help": "Whether to load existing checkpoints"})
    force_rebuild_features: bool = field(default=False, metadata={"help": "Force rebuilding of features"})
    force_retrain: bool = field(default=False, metadata={"help": "Force retraining even if checkpoints exist"})
    epoch_log: int = field(default=5, metadata={"help": "Log interval for epochs"})
    fast_cuda: bool = field(default=True, metadata={"help": "Enable fast CUDA options"})
    mixed_precision: bool = field(default=False, metadata={"help": "Use mixed precision (False = Use full precision)"})
    dataloader: int = field(default=2, metadata={"help": "Number of dataloader workers"})

    # Experimentation-Pipeline
    train_ratio: float = field(default=0.70, metadata={"help": "Ratio of training data"})
    valid_ratio: float = field(default=0.15, metadata={"help": "Ratio of validation data"})
    test_ratio: float = field(default=0.15, metadata={"help": "Ratio of test data"})
    batch_size: int = field(default=1024, metadata={"help": "Batch size for training"})

    # Experimentation-Models
    iso_forest_contamination_rate: float = field(default=0.005, metadata={"help": "Contamination rate for Isolation Forest"})
    iso_forest_quantile: int = field(default=99, metadata={"help": "Quantile threshold for Isolation Forest"})
    
    auto_encoder_quantile: int = field(default=95, metadata={"help": "Quantile threshold for AutoEncoder"})
    
    ens_quantile: int = field(default=99, metadata={"help": "Quantile threshold for Ensemble model"})

    # Experimentation-Metrics
    top_k: List[int] = field(
        default_factory=lambda: [10, 50, 100, 1000], 
        metadata={"help": "Top-K values for evaluation metrics"}
    )




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to your YAML config file")
    args, remaining = parser.parse_known_args()

    if args.config.endswith(".yaml"):
    elif args.config.endswith(".json"):
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