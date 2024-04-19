import torch
from dataclasses import dataclass
from typing import Optional
from transformers import HfArgumentParser
from ..task import DatasetArguments, DatasetLoader
from ..task.flax import flax_tasks
from ..trainer.utils.wandb_logger import WandbLogger


@dataclass
class LauncherArguments:
    task: str = "lm"
    trainer: Optional[str] = None

def eval_env_device(trainer: Optional[str] = None):
    if trainer is None or trainer == "default":
        from ..trainer import flax_base
        trainer_cls = flax_base.FlaxBaseTrainer

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

    logger = WandbLogger(trainer_args)
    dataloader = DatasetLoader(data_args)

    task = task_cls(task_args)
    dataloader.dataset = task.encode_datasets(dataloader.dataset)
    
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