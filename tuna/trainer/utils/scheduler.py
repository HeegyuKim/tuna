import math
import torch
from torch.optim.lr_scheduler import _LRScheduler
from typing import Optional


class AnnealingWarmupScheduler(_LRScheduler):
    """
        optimizer (Optimizer): Wrapped optimizer.
        first_cycle_steps (int): First cycle step size.
        cycle_mult(float): Cycle steps magnification. Default: -1.
        max_lr(float): First cycle's max learning rate. Default: 0.1.
        min_lr(float): Min learning rate. Default: 0.001.
        warmup_steps(int): Linear warmup step size. Default: 0.
        gamma(float): Decrease rate of max learning rate by cycle. Default: 1.
        last_epoch (int): The index of last epoch. Default: -1.
    """
    
    def __init__(self,
                 optimizer : torch.optim.Optimizer,
                 total_steps : int,
                 max_lr : float = 1e-4,
                 last_lr : float = 0.0,
                 warmup_steps : int = 0,
                 mode: Optional[str] = None # none, linear, cosine
        ):
        assert warmup_steps < total_steps
        
        self.total_steps = total_steps # first cycle step size
        self.base_max_lr = max_lr # first max learning rate
        self.max_lr = max_lr # max learning rate in the current cycle
        self.min_lr = last_lr # min learning rate
        self.warmup_steps = warmup_steps # warmup step size
        self.mode = mode
        self.curr_steps = 0
        
        super(AnnealingWarmupScheduler, self).__init__(optimizer, -1)

        # set learning rate min_lr
        self.init_lr()
    
    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
            self.base_lrs.append(self.min_lr)
    
    def get_lr(self):
        progress_rate = 1 - (self.curr_steps - self.warmup_steps) / (self.total_steps-self.warmup_steps)

        if self.curr_steps <= self.warmup_steps:
            return [(self.max_lr - base_lr
                     ) / self.warmup_steps * self.curr_steps + base_lr for base_lr in self.base_lrs]
        elif self.mode is None:
            return [self.max_lr for _ in self.base_lrs]
        elif self.curr_steps > self.total_steps:
            return self.base_lrs # min_lrs
        elif self.mode == "linear":
            return [base_lr + (self.max_lr - base_lr) * progress_rate for base_lr in self.base_lrs]
        elif self.mode == "cosine":
            return [base_lr + (self.max_lr - base_lr) \
                    * (1 - (1 + math.cos(math.pi * progress_rate))/ 2 )
                    for base_lr in self.base_lrs]
        else:
            return None

    def step(self, epoch=None):
        self.curr_steps += 1
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr