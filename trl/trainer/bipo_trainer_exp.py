from typing import Optional, override
import torch
from .bipo_trainer import BiPOTrainer

try:
    import wandb
    has_wandb = True
except ImportError:
    has_wandb = False



class BiPOTrainerEXP(BiPOTrainer):
    def __init__(self, *args, quantile: Optional[float] = 0.0, num_layer: Optional[int] = 26, filter_step:int = 4,**kwargs):
        super().__init__(*args, **kwargs)
        self.fisher_accumulator = {}
        self.importance_map = {}
        self.quantile_threshold = quantile
        self.filter_step = filter_step
        self.idx_layer_tensor = torch.arange(num_layer, dtype=torch.float32)

        for name, p in self.model.named_parameters():
            if "vec" in name:
                self.fisher_accumulator[name] = torch.zeros_like(p)
    
    # EXP 1 AND 2
    # @override
    # def training_step(self, model, inputs,num_items_in_batch=None):
    #     loss = super().training_step(model, inputs,num_items_in_batch)

    #     with torch.no_grad():
    #         for name, param in model.named_parameters():
    #             if "vec" in name and param.grad is not None:
    #                 self.fisher_accumulator[name] += param.grad.pow(2)

    #         if self.state.global_step % self.filter_step == 0:
    #             for name, param in model.named_parameters():
    #                 if "vec" in name:
    #                     self.importance_map[name] = self.fisher_accumulator[name].clone()
    #                     self.fisher_accumulator[name].zero_()


    #         for name, param in model.named_parameters():
    #             if name in self.importance_map:
    #                 importance = self.importance_map[name].float()
    #                 threshold = torch.quantile(importance, self.quantile )
    #                 mask = (importance >= threshold).float()
                    
    #                 if param.grad is not None:
    #                     param.grad.mul_(mask)

    #     return loss
    
    # EXP 3:
    # @override
    # def training_step(self, model, inputs,num_items_in_batch=None):
    #     loss = super().training_step(model, inputs,num_items_in_batch)

    #     with torch.no_grad():
    #         for name, param in model.named_parameters():
    #             if "vec" in name and param.grad is not None:
    #                 self.fisher_accumulator[name] += param.grad.pow(2)

    #         if self.state.global_step % self.filter_step == 0:
    #             for name, param in model.named_parameters():
    #                 if "vec" in name:
    #                     self.importance_map[name] = self.fisher_accumulator[name].clone()
    #                     self.fisher_accumulator[name].zero_()


    #         for name, param in model.named_parameters():
    #             if name in self.importance_map:
    #                 importance = self.importance_map[name].float()
                    
    #                 min_val = importance.min()
    #                 max_val = importance.max()
                    
    #                 soft_mask = (importance - min_val) / (max_val - min_val + 1e-8)
    #                 param.grad.mul_(soft_mask)

    #     return loss

    # EXP 4: Combined soft masking based on selected neuron with hard masking
    # @override
    # def training_step(self, model, inputs,num_items_in_batch=None):
    #     loss = super().training_step(model, inputs,num_items_in_batch)

    #     with torch.no_grad():
    #         for name, param in model.named_parameters():
    #             if "vec" in name and param.grad is not None:
    #                 self.fisher_accumulator[name] += param.grad.pow(2)

    #         if self.state.global_step % self.filter_step == 0:
    #             for name, param in model.named_parameters():
    #                 if "vec" in name:
    #                     self.importance_map[name] = self.fisher_accumulator[name].clone()
    #                     self.fisher_accumulator[name].zero_()


    #         for name, param in model.named_parameters():
    #             if name in self.importance_map:
    #                 importance = self.importance_map[name].float()
    #                 threshold = torch.quantile(importance, self.quantile_threshold)
                    
    #                 bool_mask = importance >= threshold
                    
    #                 if bool_mask.any():
    #                     selected_vals = importance[bool_mask]
                        
    #                     min_val = selected_vals.min()
    #                     max_val = selected_vals.max()
                        
    #                     normalized_importance = (importance - min_val) / (max_val - min_val + 1e-8)
    #                     soft_mask = normalized_importance * bool_mask.float()
    #                     param.grad.mul_(soft_mask)
               

    #     return loss

    # EXP 5 AND 6: Combined hard masking and soft masking with scheduler
    # @override
    # def training_step(self, model, inputs,num_items_in_batch=None):
    #     loss = super().training_step(model, inputs,num_items_in_batch)

    #     with torch.no_grad():
    #         for name, param in model.named_parameters():
    #             if "vec" in name and param.grad is not None:
    #                 self.fisher_accumulator[name] += param.grad.pow(2)

    #         if self.state.global_step % self.filter_step == 0:
    #             for name, param in model.named_parameters():
    #                 if "vec" in name:
    #                     self.importance_map[name] = self.fisher_accumulator[name].clone()
    #                     self.fisher_accumulator[name].zero_()


    #         for name, param in model.named_parameters():
    #             if name in self.importance_map:
    #                 importance = self.importance_map[name].float()
    #                 threshold = torch.quantile(importance, self.quantile_threshold)
    #                 hard_mask = (importance >= threshold).float()
                    
    #                 min_val = importance.min()
    #                 max_val = importance.max()
                    
    #                 soft_mask = (importance - min_val) / (max_val - min_val)
    #                 mask = hard_mask * soft_mask
    #                 param.grad.mul_(mask)

    #     return loss

    # EXP 7 AND 8: Hard masking with Scheduler
    # @override
    # def training_step(self, model, inputs,num_items_in_batch=None):
    #     loss = super().training_step(model, inputs,num_items_in_batch)

    #     with torch.no_grad():
    #         for name, param in model.named_parameters():
    #             if "vec" in name and param.grad is not None:
    #                 self.fisher_accumulator[name] += param.grad.pow(2)

    #         if self.state.global_step % self.filter_step == 0:
    #             for name, param in model.named_parameters():
    #                 if "vec" in name:
    #                     self.importance_map[name] = self.fisher_accumulator[name].clone()
    #                     self.fisher_accumulator[name].zero_()


    #         for name, param in model.named_parameters():
    #             if name in self.importance_map:
    #                 importance = self.importance_map[name].float()
    #                 threshold = torch.quantile(importance, self.quantile_threshold)
    #                 hard_mask = (importance >= threshold).float()
    #                 param.grad.mul_(hard_mask)

    #     return loss

    # EXP 9: Gradual Unfreezing with soft masking  
    @override
    def training_step(self, model, inputs,num_items_in_batch=None):
        # Gradual Freezing
        self.quantile_threshold = getattr(self.state, "custom_quantile_threshold", self.quantile_threshold)
        threshold = torch.quantile(self.idx_layer_tensor, 1-self.quantile_threshold) 
        hard_mask = self.idx_layer_tensor >= threshold
        vec_idx = 0
        for name, param in model.named_parameters():
            if "vec" in name:
                print(f"[Layer:] {vec_idx} Learning" if hard_mask[vec_idx].item() else f"[Layer:] {vec_idx} Freezing")
                param.requires_grad = hard_mask[vec_idx].item()
                vec_idx += 1

        loss = super().training_step(model, inputs,num_items_in_batch)

        with torch.no_grad():
            if self.state.global_step % self.filter_step == 0:
                for name, param in model.named_parameters():
                    if "vec" in name and param.grad is not None:
                        self.importance_map[name] = param.grad.pow(2)


            for name, param in model.named_parameters():
                if name in self.importance_map:
                    importance = self.importance_map[name].float()
                    
                    min_val = importance.min()
                    max_val = importance.max()
                    
                    soft_mask = (importance - min_val) / (max_val - min_val+ 1e-8)
                    param.grad.mul_(soft_mask)

         # Log to wandb
        if has_wandb and wandb.run is not None:
            if self.state.global_step > 0:
                wandb.log(
                    {"custom/quantile_threshold": self.quantile_threshold,
                     "custom/layer_unfreezed": hard_mask.sum().item()}, 
                    step=self.state.global_step
                )

        return loss
