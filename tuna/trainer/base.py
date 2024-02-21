from typing import List, Dict, Any, Optional, Iterable, Union
from dataclasses import dataclass
from tqdm.auto import tqdm
from pprint import pprint
import tempfile

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

import transformers
from transformers import HfArgumentParser, set_seed
from datasets import disable_caching, load_dataset, IterableDataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    AutoConfig,
    AutoModelForTokenClassification,
    TrainingArguments,
)
from huggingface_hub import HfApi

from .utils import convert_dict_tensor_devices, detach_tensors, BaseLogger
from ..task.collator import DefaultCollator
from .utils.scheduler import AnnealingWarmupScheduler
from .args import BaseTrainingArguments
from ..task import Task
from ..common import Registry


trainers = Registry("trainer")

@trainers.register("base")
class BaseTrainer:
    ARG_CLASS = BaseTrainingArguments
    
    def __init__(
            self,
            args: BaseTrainingArguments,
            logger: BaseLogger,
            task: Task,
            train_dataset: Optional[Iterable] = None,
            eval_dataset: Optional[Iterable] = None,
            ) -> None:
        self.args = args
        self.task = task
        self.model = task.model
        self.tokenizer = task.tokenizer
        self.logger = logger
        self.collator = task.collate_batch
        self.device = args.device
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

    def setup(self):
        self.logger.log("setup dataloader")
        self.setup_dataloader()

        if self.args.do_train:
            self.setup_optimizer()
            self.setup_scheduler()

    def _create_dataloader(self, dataset, batch_size, shuffle):
        if isinstance(dataset, IterableDataset):
            num_workers = 0
            shuffle = False
        
        return DataLoader(
            dataset,
            batch_size,
            shuffle=shuffle,
            collate_fn=self.collator,
            drop_last=True
        )
    
    def setup_dataloader(self):
        if self.args.do_train:
            self.train_loader = self._create_dataloader(
                self.train_dataset,
                self.args.train_batch_size_per_device,
                True
            )
        else:
            self.train_loader = None

        if self.eval_dataset:
            self.eval_loader = self._create_dataloader(
                self.eval_dataset,
                self.args.eval_batch_size_per_device,
                False
            )
        else:
            self.eval_loader = None

    def setup_optimizer(self):
        self.optimizer = optim.AdamW(
            self.task.get_trainable_parameters(),
            self.args.learning_rate,
            [self.args.adam_beta1, self.args.adam_beta2],
            weight_decay=self.args.weight_decay
        )

    def setup_scheduler(self):
        optimizer = self.optimizer
        if self.args.total_steps is not None:
            total_steps = self.args.total_steps
        else:
            total_steps = (len(self.train_dataset) // self.args.train_total_batch_size) * self.args.total_epochs

        warmup_steps = int(self.args.lr_warmup_ratio * total_steps) if self.args.lr_warmup_ratio else self.args.lr_warmup_steps
        decay_steps = self.args.lr_decay_steps if self.args.lr_decay_steps else total_steps

        if self.args.last_learning_rate:
            last_lr = self.args.last_learning_rate
        elif self.args.last_learning_rate_ratio:
            last_lr = self.args.learning_rate * self.args.last_learning_rate_ratio
        else:
            last_lr = 0.0

        self.lr_scheduler = AnnealingWarmupScheduler(
            optimizer,
            decay_steps,
            max_lr=self.args.learning_rate,
            last_lr=last_lr,
            warmup_steps=warmup_steps,
            mode=self.args.lr_scheduler
        )

    def launch(self):
        print("launch!")
        self.setup()

        if self.args.do_train:
            self.train()
        elif self.args.do_eval:
            self.evaluate(0, 0)

    @property
    def process_count(self):
        return 1

    def train(self):
        print("train!")
        global_step = 0
        optimizer_step = 0
        args = self.args
        if self.args.total_steps is not None:
            epoch_steps = None
            total_steps = self.args.total_steps
        else:
            epoch_steps = len(self.train_dataset) // self.args.train_batch_size_per_device
            total_steps = epoch_steps * self.args.total_epochs  
        step_outputs = []

        description = f"{self.args.project}/{self.args.run_name}"

        if args.eval_strategy == "epoch" and args.eval_per_epoch > 1:
            args.eval_strategy = "steps"
            args.eval_steps = epoch_steps // args.eval_per_epoch

        if args.save_per_epoch is not None and "epoch" in args.save_strategy:
            args.save_strategy = "steps"
            args.save_steps = epoch_steps // args.save_per_epoch
        
        if self.args.train_total_batch_size:
            assert self.args.train_total_batch_size % self.args.train_batch_size_per_device == 0
            gradient_accumulation_steps = self.args.train_total_batch_size // self.args.train_batch_size_per_device // self.process_count
        else:
            gradient_accumulation_steps = 1

        print("## Training specs")
        print("total batch", self.args.train_total_batch_size)
        print("grad_accum", gradient_accumulation_steps)
        print("train_batch_size_per_device", self.args.train_batch_size_per_device)
        print("process count", self.process_count)

        progress = tqdm(
            total=total_steps,
            desc=description,
            ascii=''
            )
        epoch = 0
        autocast_device = self.device.split(":")[0] if isinstance(self.device, str) else self.device.type

        while True:
            self.task.model.train()

            if args.total_steps is None and epoch >= args.total_epochs:
                break
            
            for step_in_epoch, batch in enumerate(self.train_loader):
                global_step += 1
                if args.total_steps is not None:
                    epoch_float = float(epoch)  
                    progress.desc = f"{description}, total {global_step / total_steps:.4f}"    
                else:
                    epoch_float = epoch + (step_in_epoch + 1) / epoch_steps
                    progress.desc = f"{description} epoch: {epoch_float:.4f}"

                batch = convert_dict_tensor_devices(batch, self.device)
                with torch.autocast(autocast_device, enabled=args.amp):
                    step_output = self.task.train_step(batch, global_step)

                    if torch.is_tensor(step_output):
                        loss = step_output
                    else:
                        loss = step_output['loss']

                    loss = loss / gradient_accumulation_steps
                self.backward_loss(loss)
                step_outputs.append(step_output)


                if (global_step + 1) % gradient_accumulation_steps == 0:
                    self.optimizer_step()

                    optimizer_step += 1

                if (
                    global_step % self.args.logging_steps == 0
                ):  
                    metrics = self.task.collate_train_step_outputs(step_outputs)
                    metrics = {f"train/{k}": v for k, v in metrics.items()}
                    metrics["train/optimizer_step"] = optimizer_step
                    metrics["train/progress_rate"] = global_step / total_steps
                    metrics["train/learning_rate"] = self.lr_scheduler.get_lr()[0] if self.lr_scheduler else self.args.learning_rate
                    metrics["train/loss"] = loss.item() * gradient_accumulation_steps
                    metrics["global_step"] = global_step
                    metrics["epoch"] = epoch_float
                    self.logger.log_metric(metrics)
                    step_outputs = []

                if self.args.save_strategy and "steps" in self.args.save_strategy and global_step % self.args.save_steps == 0:
                    self.save_model(f"steps-{global_step}")

                if (
                    self.args.do_eval
                    and self.args.eval_strategy == "steps"
                    and global_step % self.args.eval_steps == 0
                ):
                    self.evaluate(epoch_float, global_step)
                
                progress.update(1)

                if global_step >= total_steps:
                    break

            
            if "epoch" in self.args.save_strategy and (epoch + 1) % self.args.save_epochs == 0:
                self.save_model(f"epoch-{epoch + 1}")

            if self.args.do_eval and self.args.eval_strategy == "epoch":
                self.evaluate(epoch_float, global_step)

            epoch += 1
            
            if global_step >= total_steps:
                break

        if "last" in self.args.save_strategy:
            self.save_model(f"epoch-{epoch + 1}-last", True)

        print("training is finished!")

    def backward_loss(self, loss):
        loss.backward()
    
    def optimizer_step(self):
        self.optimizer.step()
        if self.lr_scheduler:
            self.lr_scheduler.step()
        self.optimizer.zero_grad()

    def save_model(self, name, is_last=False):
        print("save", name)
        run_name = self.args.run_name.replace("/", "__")
        device = next(self.task.model.parameters()).device
        self.task.model.cpu()

        repo_id = run_name.replace("/", "__").replace(",", "_")

        if self.args.output_dir:
            path = f"{self.args.output_dir}/{run_name}/{name}"
            self.task.save_artifacts(self.task.model, path)

            if self.args.push_to_hub:
                self.push_to_hub_revision(repo_id, path, "main" if is_last else name)

        elif self.args.push_to_hub:
            with tempfile.TemporaryDirectory() as path:
                self.task.save_artifacts(self.task.model, path)
                self.push_to_hub_revision(repo_id, path, "main" if is_last else name)
        
        self.task.model.to(device)

    def push_to_hub_revision(self, repo_id, folder_path, revision_name):
        api = HfApi()
        if "/" not in repo_id:
            name = api.whoami()['name']
            repo_id = f"{name}/{repo_id}"

        api.create_repo(repo_id, private=True, repo_type="model", exist_ok=True)
        api.create_branch(repo_id, branch=revision_name)
        api.upload_folder(
            repo_id=repo_id,
            folder_path=folder_path,
            revision=revision_name,
        )

    @torch.no_grad()
    def evaluate(self, epoch, global_step):
        print("start evaluation", epoch, global_step)
        
        self.task.model.eval()

        if "eval" in self.args.save_strategy:
            self.save_model(f"epoch-{epoch:.2f}")

        progress = tqdm(
            self.eval_loader,
            leave=False,
        )
        
        step_outputs = []
        for step, batch in enumerate(progress):
            batch = convert_dict_tensor_devices(batch, self.device)
            outputs = self.task.evaluation_step(batch, step)
            if torch.is_tensor(outputs):
                outputs = {"loss": outputs}
            step_outputs.append(convert_dict_tensor_devices(outputs, "cpu"))

        eval_results = self.task.collate_evaluation_outputs(step_outputs)
        eval_results = {f"eval/{k}": v.item() if torch.is_tensor(v) else v for k, v in eval_results.items()}
        eval_results["epoch"] = round(epoch, 2)
        eval_results["global_step"] = global_step
        self.logger.log_metric(eval_results)

        self.task.model.train()

        