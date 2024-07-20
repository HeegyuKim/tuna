import torch
from dataclasses import dataclass
from typing import Optional
from transformers import HfArgumentParser, TrainingArguments
from ..task import tasks, DatasetArguments, DatasetLoader
from ..model import models
from ..trainer.utils.wandb_logger import WandbLogger
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported


@dataclass
class UnslothArguments:
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    use_lora: bool = False
    model_name_or_path: str = "unsloth/Mistral-Nemo-Base-2407"



def main():
    task_cls = tasks.get("chat-lm")

    unsloth_args, task_args, data_args, trainer_args = HfArgumentParser([
        UnslothArguments,
        task_cls.ARG_CLASS,
        DatasetArguments,
        TrainingArguments,
        ]).parse_args_into_dataclasses()
    
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=unsloth_args.model_name_or_path,
        max_seq_length=task_args.max_length,
        dtype=None,
        load_in_4bit=unsloth_args.load_in_4bit,
    )
    if unsloth_args.use_lora:
        from ..common.lora_util import find_lora_targets

        model = FastLanguageModel.get_peft_model(
            model,
            r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
            target_modules=find_lora_targets(model),
            lora_alpha = 32,
            lora_dropout = 0, # Supports any, but = 0 is optimized
            bias = "none",    # Supports any, but = "none" is optimized
            # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
            use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
            random_state = 3407,
            use_rslora = False,  # We support rank stabilized LoRA
            loftq_config = None, # And LoftQ
        )
    
    task = task_cls(
        task_args,
        dict(tokenizer=tokenizer),
        trainer_args.device
        )
    dataloader = DatasetLoader(data_args)
    dataloader.dataset = task.encode_datasets(dataloader.dataset, data_args)
    
    trainer_args.bf16 = is_bfloat16_supported()
    trainer_args.fp16 = not trainer_args.bf16

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset=dataloader.train_dataset,
        eval_dataset=dataloader.test_dataset,
        max_seq_length=task_args.max_length,
        dataset_num_proc=1,
        packing = False, # Can make training 5x faster for short sequences.
        args=trainer_args,        
    )
    trainer_stats = trainer.train()
    print(trainer_stats)


if __name__ == "__main__":
    main()