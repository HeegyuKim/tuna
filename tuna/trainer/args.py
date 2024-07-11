from typing import List, Dict, Any, Optional, Iterable, Union
from dataclasses import dataclass

@dataclass
class BaseTrainingArguments():
    seed: int = 42
    device: str = "auto"
    
    # logger
    logger: Optional[str] = None # wandb
    logging_steps: Optional[int] = None
    project: str = "reward_model"
    run_name: Optional[str] = None

    # training
    do_train: bool = True
    total_epochs: int = 3
    total_steps: Optional[int] = None
    train_total_batch_size: Optional[int] = None
    train_batch_size_per_device: int = 8
    amp: bool = False
    dtype: str = "bfloat16"
    param_dtype: str = None
    compile_step: bool = False

    # evaluation
    do_eval: bool = False
    eval_batch_size_per_device: int = 8
    eval_strategy: str = "epoch" # steps, epoch, last
    eval_epochs: Optional[float] = None
    eval_per_epoch: int = 1
    eval_steps: Optional[int] = None
    eval_batch_limit: Optional[int] = None

    ## optimizer, lr decay
    optimizer: str = "adamw"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    weight_decay: float = 0
    learning_rate: float = 5e-5
    last_learning_rate: Optional[float] = None
    last_learning_rate_ratio: Optional[float] = None
    gradient_clipping: float = 1.0
    lr_scheduler: Optional[str] = "linear"
    lr_decay_ratio: float = 1.0
    lr_decay_steps: Optional[int] = None # no_decay(none or -1)
    lr_warmup_steps: Optional[int] = 0 # no_decay(none or -1)
    lr_warmup_ratio: Optional[float] = None # no_decay(none or -1)

    # checkpoint
    save_strategy: Optional[str] = None # steps, epoch, last, eval
    save_epochs: int = 1
    save_per_epoch: Optional[int] = None 
    save_steps: int = 10000
    save_format: Optional[str] = "bf16" 

    output_dir: Optional[str] = "./checkpoint"
    push_to_hub: bool = False
    push_to_hub_id: Optional[str] = None
    revision_prefix: Optional[str] = None

    def setup_logger(self):
        if self.config.logger == "wandb":
            from utils.wandb_logger import WandbLogger
            self.logger = WandbLogger(self.config)
        else:
            from utils.logger import BaseLogger
            self.logger = BaseLogger(self.config)