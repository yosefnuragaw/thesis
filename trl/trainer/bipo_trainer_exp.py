from .bipo_trainer import BiPOTrainer
import torch


class BiPOTrainerEXP(BiPOTrainer):
    def __init__(self, *args, quantile: float = 0.9, **kwargs):
        super().__init__(*args, **kwargs)
        self.fisher_accumulator = {}
        self.importance_map = {}
        self.quantile = quantile
        
        for name, p in self.model.named_parameters():
            if "vec" in name:
                self.fisher_accumulator[name] = torch.zeros_like(p)
        

    def training_step(self, model, inputs,num_items_in_batch=None):
        loss = super().training_step(model, inputs,num_items_in_batch)

        with torch.no_grad():
            for name, param in model.named_parameters():
                if "vec" in name and param.grad is not None:
                    self.fisher_accumulator[name] += param.grad.pow(2)

            if self.state.global_step % 4 == 0:
                for name, param in model.named_parameters():
                    if "vec" in name:
                        self.importance_map[name] = self.fisher_accumulator[name].clone()
                        self.fisher_accumulator[name].zero_()


            for name, param in model.named_parameters():
                if name in self.importance_map:
                    importance = self.importance_map[name].float()
                    threshold = torch.quantile(importance, self.quantile )
                    mask = (importance >= threshold).float()
                    
                    if param.grad is not None:
                        param.grad.mul_(mask)

        return loss
    
