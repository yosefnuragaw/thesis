import math
from transformers import TrainerCallback


class QuantileSchedulerCallback(TrainerCallback):
    def __init__(self, start_val, schedule_type="linear"):
        self.start_val = start_val
        self.end_val = 1.
        self.schedule_type = schedule_type

    def on_step_begin(self, args, state, control, model, **kwargs):
        if state.max_steps == 0:
            return
            
        # Ensure progress doesn't exceed 1.0
        progress = min(1.0, state.global_step / state.max_steps)
        delta = self.end_val - self.start_val
        
        if self.schedule_type == "linear":
            current_val = self.start_val + progress * delta
            
        elif self.schedule_type == "quadratic":
            # Slow start, fast finish
            current_val = self.start_val + (progress ** 2) * delta
            
        elif self.schedule_type == "exponential":
            # Constant percentage growth
            current_val = self.start_val * (self.end_val / self.start_val) ** progress
            
        elif self.schedule_type == "cosine":
            # Smooth S-curve transition
            current_val = self.start_val + delta * 0.5 * (1 - math.cos(math.pi * progress))
        
        else:
            current_val = self.start_val

        # Update model attribute
        state.custom_quantile_threshold = current_val
            
