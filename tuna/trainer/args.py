from typing import List, Dict, Any, Optional, Iterable, Union
from dataclasses import dataclass

@dataclass
class BaseTrainingArguments():
    seed: int = 42
    device: str = "auto"
    
    # logger
    logger: Optional[str] = None # wandb
    logging_steps: int = 128
    project: str = "reward_model"
    run_name: Optional[str] = None

    # training
    do_train: bool = True
    total_epochs: int = 3
    total_steps: Optional[int] = None
    train_total_batch_size: Optional[int] = None
    train_batch_size_per_device: int = 8
    amp: bool = True

    # evaluation
    do_eval: bool = False
    eval_batch_size_per_device: int = 8
    eval_strategy: str = "epoch" # steps, epoch, last
    eval_epochs: Optional[float] = None
    eval_per_epoch: int = 1

    ## optimizer, lr decay
    # optimizer: str = "adamw"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    weight_decay: float = 1e-2
    learning_rate: float = 5e-5
    last_learning_rate: Optional[float] = None
    last_learning_rate_ratio: Optional[float] = None
    lr_scheduler: Optional[str] = "linear"
    lr_decay_steps: Optional[int] = None # no_decay(none or -1)
    lr_warmup_steps: Optional[int] = 0 # no_decay(none or -1)
    lr_warmup_ratio: Optional[int] = None # no_decay(none or -1)

    # checkpoint
    save_strategy: Optional[str] = None # steps, epoch, last, eval
    save_epochs: int = 1
    save_per_epoch: Optional[int] = None 
    save_steps: int = 10000
    save_format: Optional[str] = None 

    output_dir: Optional[str] = "./checkpoint"
    push_to_hub: bool = False

    def setup_logger(self):
        if self.config.logger == "wandb":
            from utils.wandb_logger import WandbLogger
            self.logger = WandbLogger(self.config)
        else:
            from utils.logger import BaseLogger
            self.logger = BaseLogger(self.config)