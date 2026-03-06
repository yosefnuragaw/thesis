from typing import Optional, override
import torch
from .bipo_trainer import BiPOTrainer

try:
    import wandb
    has_wandb = True
except ImportError:
    has_wandb = False



class BiPOTrainerEXP(BiPOTrainer):
    def __init__(self, *args,experiment_pipeline:str = 'both', masking_type:str = 'soft', quantile: Optional[float] = 0.0, num_layer: Optional[int] = 26,  filter_step:int = 4,**kwargs):
        super().__init__(*args, **kwargs)
        self.fisher_accumulator = {}
        self.importance_map = {}
        self.quantile_threshold = quantile
        self.filter_step = filter_step
        self.experiment_pipeline = experiment_pipeline
        self.masking_type = masking_type
        self.idx_param = {}

        # self.idx_layer_tensor = torch.arange(num_layer, dtype=torch.float32)
        self.layer_weight = torch.arange(num_layer, dtype=torch.float32)

        # for name, p in self.model.named_parameters():
        #     if "vec" in name:
        #         self.fisher_accumulator[name] = torch.zeros_like(p)
    
    # EXP 9: Gradual Unfreezing with soft masking  
    # @override
    # def training_step(self, model, inputs,num_items_in_batch=None):
        
    #     # Gradual Freezing
    #     if self.experiment_pipeline != 'two':  
    #         self.quantile_threshold = getattr(self.state, "custom_quantile_threshold", self.quantile_threshold)
    #         threshold = torch.quantile(self.idx_layer_tensor, 1-self.quantile_threshold) 
    #         hard_mask = self.idx_layer_tensor >= threshold
    #         vec_idx = 0
    #         for name, param in model.named_parameters():
    #             if "vec" in name:
    #                 print(f"[Layer:] {vec_idx} Learning" if hard_mask[vec_idx].item() else f"[Layer:] {vec_idx} Freezing")
    #                 param.requires_grad = hard_mask[vec_idx].item()
    #                 vec_idx += 1

    #     loss = super().training_step(model, inputs,num_items_in_batch)


    #     with torch.no_grad():
    #         if self.state.global_step % self.filter_step == 0 and self.state.global_step > 0:
    #             for name, param in model.named_parameters():
    #                 if "vec" in name and param.grad is not None:
    #                     self.importance_map[name] = param.grad.pow(2)

    #         if self.experiment_pipeline != 'one':
    #             for name, param in model.named_parameters():
    #                 if name in self.importance_map and param.grad is not None:
    #                     importance = self.importance_map[name].float()

    #                     if self.masking_type == 'soft':
    #                         min_val = importance.min()
    #                         max_val = importance.max()
                            
    #                         soft_mask = (importance - min_val) / (max_val - min_val+ 1e-8)
    #                         param.grad.mul_(soft_mask)

    #                     if self.masking_type == 'hard':
    #                         importance = self.importance_map[name].float()
    #                         threshold = torch.quantile(importance, 0.9)
    #                         hard_mask = (importance >= threshold).float()
                            
    #                         param.grad.mul_(hard_mask)

  

    #     return loss

     # EXP 11: Fisher Layer selection
    @override
    def training_step(self, model, inputs, num_items_in_batch=None):
        # 1. Initial mapping
        if self.state.global_step == 0:  
            vec_idx = 0
            for name, param in model.named_parameters():
                if "vec" in name:
                    self.idx_param[vec_idx] = param
                    vec_idx += 1

        if self.state.global_step % self.filter_step == 0:
            for param in self.idx_param.values():
                param.requires_grad = True
                
        loss = super().training_step(model, inputs, num_items_in_batch)

        with torch.no_grad():
            if self.state.global_step % self.filter_step == 0:
                
                for idx, param in self.idx_param.items():
                    if param.grad is not None:
                        self.layer_weight[idx] = param.grad.pow(2).mean()

                self.quantile_threshold = getattr(self.state, "custom_quantile_threshold", self.quantile_threshold)
                threshold = torch.quantile(self.layer_weight, 1 - self.quantile_threshold) 
                hard_mask = self.layer_weight >= threshold

                # Apply the freeze
                for idx, param in self.idx_param.items():
                    is_learning = hard_mask[idx].item()
                    param.requires_grad = is_learning
                    
                    if is_learning:
                        pass
                        # print(f"[Layer:] {idx} Learning")
                    else:
                        # print(f"[Layer:] {idx} Freezing")
                        param.grad = None 

        return loss
