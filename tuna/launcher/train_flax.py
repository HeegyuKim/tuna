# promote_dtypes does not reconized the qax implicit array.
# import flax.linen.dtypes
# from ..model.flax.lora_util import promote_dtype_lora_compat
# flax.linen.dtypes.promote_dtype = promote_dtype_lora_compat


import torch
from dataclasses import dataclass
from typing import Optional
from transformers import HfArgumentParser
from ..task import DatasetArguments, DatasetLoader
from ..task.flax import flax_tasks
from ..trainer.utils.wandb_logger import WandbLogger
from ..trainer import flax_trainer

@dataclass
class LauncherArguments:
    task: str = "lm"
    trainer: Optional[str] = None
    debug: bool = False

def eval_env_device(trainer: Optional[str] = None):
    if trainer is None or trainer == "default":
        from ..trainer import flax_base
        trainer_cls = flax_base.FlaxBaseTrainer
    elif trainer == "dpo":
        from ..trainer import flax_base
        trainer_cls = flax_trainer.FlaxDPOTrainer
    else:
        raise ValueError(f"Unknown trainer: {trainer}")

    return trainer_cls

def main():
    args, _ = HfArgumentParser([LauncherArguments]).parse_args_into_dataclasses(return_remaining_strings=True)

    task_cls = flax_tasks.get(args.task)

    trainer_cls = eval_env_device(args.trainer)

    print("Task:", task_cls, task_cls.ARG_CLASS)
    print("Trainer:", trainer_cls)

    _, task_args, data_args, trainer_args = HfArgumentParser([
        LauncherArguments,
        task_cls.ARG_CLASS,
        DatasetArguments,
        trainer_cls.ARG_CLASS,
        ]).parse_args_into_dataclasses()

    if not args.debug:
        logger = WandbLogger(trainer_args)
        
    dataloader = DatasetLoader(data_args)

    task = task_cls(task_args)
    dataloader.dataset = task.encode_datasets(dataloader.dataset)
    
    if args.debug:
        print("Debugging dataset")
        for k in dataloader.dataset:
            for i in range(10):
                print(f"train#{i:2d}", dataloader.dataset[k][i])
        return
    else:
        trainer = trainer_cls(
            args=trainer_args,
            logger=logger,
            task=task,
            train_dataset=dataloader.train_dataset,
            eval_dataset=dataloader.test_dataset,
        )
        trainer.launch()

if __name__ == "__main__":
    main()