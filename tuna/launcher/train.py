import torch
from dataclasses import dataclass
from typing import Optional
from transformers import HfArgumentParser
from ..task import tasks, DatasetArguments, DatasetLoader
from ..model import models
from ..trainer.utils.wandb_logger import WandbLogger


@dataclass
class LauncherArguments:
    task: str
    model_arch: str
    trainer: Optional[str] = None

def eval_env_device(trainer: Optional[str] = None):
    if trainer is None:
        trainer_cls = trainer
        device = "auto"
        
        if torch.cuda.is_available():
            trainer = "accelerator"
        else:
            trainer = "spmd"
        trainer_cls

    if trainer == "accelerator":
        from ..trainer import accel
        trainer_cls = accel.AcceleratorTrainer
        if device == "auto":
            device = "cuda:0"
    elif trainer == "spmd":
        from ..trainer import spmd
        trainer_cls = spmd.SPMDTrainer
        if device == "auto":
            device = "xla:0"

    return trainer_cls, device

def main():
    args, _ = HfArgumentParser([LauncherArguments]).parse_args_into_dataclasses(return_remaining_strings=True)

    task_cls = tasks.get(args.task)
    model_cls = models.get(args.model_arch)

    trainer_cls, device = eval_env_device(args.trainer)

    
    print("Task:", task_cls, task_cls.ARG_CLASS)
    print("Model:", model_cls)
    print("Trainer:", trainer_cls)

    _, task_args, data_args, model_args, trainer_args = HfArgumentParser([
        LauncherArguments,
        task_cls.ARG_CLASS,
        DatasetArguments,
        model_cls.ARG_CLASS,
        trainer_cls.ARG_CLASS,
        ]).parse_args_into_dataclasses()
    
    if trainer_args.device == "auto":
        trainer_args.device = device
    print("Device:", trainer_args.device)
    

    logger = WandbLogger(trainer_args)
    dataloader = DatasetLoader(data_args)

    model_loader = model_cls(model_args)
    artifacts = model_loader.load_artifacts()
        
    task = task_cls(
        task_args,
        artifacts,
        trainer_args.device
        )
    dataloader.dataset = task.encode_datasets(dataloader.dataset)

    print("load model")
    task.model = model_loader.load_model()
    
    
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