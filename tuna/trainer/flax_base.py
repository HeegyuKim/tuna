
from typing import List, Dict, Any, Optional, Iterable, Union
from dataclasses import dataclass
from tqdm.auto import tqdm
from pprint import pprint
import tempfile
import termcolor
import os

import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from jax.experimental.pjit import pjit
from jax.experimental.mesh_utils import create_device_mesh
from jax.sharding import PartitionSpec
import flax
from flax.training.train_state import TrainState

from torch.utils.data import DataLoader
import transformers
from datasets import IterableDataset

from huggingface_hub import HfApi
from ..model.flax.py_flax_utils import load_flax_weights_in_pytorch_model

from fjformer.xrapture import XRapTureConfig, XRapTure
from fjformer import (
    match_partition_rules,
)

from fjformer.xrapture import use_implicit_args

from .utils import convert_dict_tensor_devices, detach_tensors, BaseLogger, upload_local_directory_to_gcs
from .args import BaseTrainingArguments
from ..task.flax.flax_base import FlaxTask
from .flax.partition_rules import get_partition_rules
from .flax import optimizer_utils as opt_utils, sharding, profiler
from ..common import Registry
from .. import flax_model


@dataclass
class FlaxTrainingArguments(BaseTrainingArguments):
    mesh: str = "fsdp"
    fully_sharded: bool = False
    bf16_momentum: bool = False

    use_lora: bool = False
    lora_r: int = 8
    lora_alpha: int = 32
    lora_ft_params: Optional[str] = None


MESH_SHAPES = {
    "fsdp": (1, -1, 1, 1),
    "dp": (-1, -1, 1, 1),
    "mp": (1, 1, 1, -1),
    "sp": (1, 1, -1, 1),
}
    
STR_DTYPE_TO_JNP = {
    "float32": jnp.float32,
    "float64": jnp.float64,
    "float16": jnp.float16,
    "bfloat16": jnp.bfloat16,
    "fp32": jnp.float32,
    "fp64": jnp.float64,
    "fp16": jnp.float16,
    "bf16": jnp.bfloat16,
}

trainers = Registry("flax-trainer")

@trainers.register("base")
class FlaxBaseTrainer:
    ARG_CLASS = FlaxTrainingArguments
    
    def __init__(
            self,
            args: BaseTrainingArguments,
            logger: BaseLogger,
            task: FlaxTask,
            train_dataset: Optional[Iterable] = None,
            eval_dataset: Optional[Iterable] = None,
            ) -> None:
        self.args = args
        self.task = task
        self.logger = logger
        self.collator = task.collate_batch
        self.device = args.device
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

    def setup(self):
        self.logger.log("setup dataloader")
        self.setup_dataloader()
        self.logger.log("setup sharding")
        self.setup_sharding()
        profiler.print_memory_usage()

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
    
    def setup_sharding(self):
        self.dtype = STR_DTYPE_TO_JNP[self.args.dtype]
        self.param_dtype = STR_DTYPE_TO_JNP[self.args.param_dtype or self.args.dtype]
        self.total_steps = self.estimate_total_steps()

        self.logger.log("init optimizer")
        self.init_optimizer(self.total_steps)

        self.logger.log("init mesh")
        self.shard_params()

    def estimate_total_steps(self):
        if self.args.total_steps is not None:
            return self.args.total_steps
        else:
            if isinstance(self.train_dataset, IterableDataset):
                raise ValueError("total_steps must be provided for IterableDataset")
            return len(self.train_dataset) * self.args.total_epochs // self.args.train_batch_size_per_device

    def apply_lora_params(self, model, tx, params):
        print("Applying LoRA!")
        from ..common.lora_util import find_lora_targets_from_config

        self.rapture = XRapTure(
            XRapTureConfig(
                lora_dim=self.args.lora_r,
                fully_fine_tune_parameters=self.args.lora_ft_params.split(",") if self.args.lora_ft_params else None,
                lora_fine_tune_parameters=find_lora_targets_from_config(model.config),
                tune_vectors=False,
                verbose=True
            )
        )
        self.lora_modules = self.rapture.apply_lora(
            module=model,
            parameters=params,
            alpha=self.args.lora_alpha,
            tx=tx,
        )

        params = self.lora_modules.lora_parameters
        if self.dtype == jnp.bfloat16:
            params = self.task.model.to_bf16(params)
        elif self.dtype == jnp.float16:
            params = self.task.model.to_fp16(params)

        self.task.params = params
        self.task.use_lora = True
        # self.task.model = self.lora_modules.lora_module

    def init_optimizer(self, total_steps):
        if self.args.last_learning_rate is not None or self.args.last_learning_rate_ratio is not None:
            end_value=self.args.last_learning_rate or (self.args.learning_rate * self.args.last_learning_rate_ratio)
            scheduler=self.args.lr_scheduler
        else:
            end_value = self.args.learning_rate
            scheduler=None
        
        if self.args.lr_decay_steps:
            lr_decay_steps = self.args.lr_decay_steps
        else:
            lr_decay_steps = total_steps

        if self.args.lr_warmup_steps:
            lr_warmup_steps = self.args.lr_warmup_steps
        elif self.args.lr_warmup_ratio:
            lr_warmup_steps = int(total_steps * self.args.lr_warmup_ratio)
        else:
            lr_warmup_steps = 0

        gradient_accumulation_steps = self.args.train_total_batch_size // self.args.train_batch_size_per_device
        
        if self.args.optimizer == "adamw":
            extra_optimizer_kwargs = {
                "b1": self.args.adam_beta1,
                "b2": self.args.adam_beta2,
                "eps": 1e-8,
                "weight_decay": self.args.weight_decay,
            }
            
            tx, sc = opt_utils.get_adamw(
                learning_rate_start=self.args.learning_rate,
                steps=lr_decay_steps,
                learning_rate_end=end_value,
                gradient_accumulation_steps=gradient_accumulation_steps,
                warmup_steps=lr_warmup_steps,
                scheduler=scheduler,
                gradient_clipping=self.args.gradient_clipping,
                **extra_optimizer_kwargs
            )

        elif self.args.optimizer == "adafactor":
            tx, sc = opt_utils.get_adafactor(
                learning_rate_start=self.args.learning_rate,
                steps=lr_decay_steps,
                learning_rate_end=end_value,
                gradient_accumulation_steps=gradient_accumulation_steps,
                warmup_steps=lr_warmup_steps,
                scheduler=scheduler,
                gradient_clipping=self.args.gradient_clipping,
                factored=False,
                momentum=self.args.adam_beta1,
                decay_rate=self.args.adam_beta2,
                clipping_threshold=None,
                dtype_momentum=self.param_dtype,
            )
        else:
            raise ValueError("unknown optimizer!")

        self.optimizer = tx
        self.learning_rate_schedule = sc

    def init_mesh(self):
        sharding_array = MESH_SHAPES[self.args.mesh]
        available_backends = len(jax.devices())
        array_devices = jnp.ones((available_backends, 1)).reshape(sharding_array)
        self.mesh = Mesh(
            create_device_mesh(
                array_devices.shape
            ),
            self.get_mesh_names()
        )

    @staticmethod
    def get_mesh_names():
        return "dp", "fsdp", "tp", "sp"

    @use_implicit_args
    def shard_params(self):
        self.init_mesh()
        
        self.logger.log("init model")
        self.task.init_model(self.dtype)
        
        with jax.default_device(jax.devices('cpu')[0]):
            params=self.task.params

            if self.args.use_lora:
                self.apply_lora_params(self.task.model, self.optimizer, params)
                params = self.task.params

                self.optimizer = self.lora_modules.lora_tx

        def init_train_state():
            state = TrainState.create(
                apply_fn=None,
                params=params,
                tx=self.optimizer
            )
            return state
        
        def create_train_state(params):
            if self.args.use_lora:
                state = TrainState(
                    step=0,
                    apply_fn=None,
                    params=params,
                    tx=self.optimizer,
                    opt_state=self.lora_modules.lora_opt_state
                )
            else:
                state = TrainState.create(
                    apply_fn=None,
                    params=params,
                    tx=self.optimizer
                )
            return state
        
        state_shape = jax.eval_shape(init_train_state)

        with self.mesh:
            print("matching partition rules")
            partition_specs = match_partition_rules(params=state_shape, rules=get_partition_rules(self.task.model.config, self.args.fully_sharded))
            shard_fns, gather_fns = sharding.make_shard_and_gather_fns_dtype(partition_specs.params, self.mesh, self.param_dtype)
            print(
                "sharding parameters across all of the chosen backend(tpu/gpu/cpu)s"
            )
            params = jax.tree_util.tree_map(
                lambda f, x: f(x),
                shard_fns,
                params
            )

            self.create_sharded_train_state = pjit(
                create_train_state,
                in_shardings=(partition_specs.params,),
                out_shardings=partition_specs,
                donate_argnums=(0,)
            )

            self.sharded_train_step = self.task.create_train_step(
                pjit,
                partition_specs,
                PartitionSpec
            )
            self.sharded_eval_step = self.task.create_eval_step(
                pjit,
                partition_specs,
                PartitionSpec
            )

            sharded_state = self.create_sharded_train_state(params)

        self.state_partition_specs = partition_specs
        self.sharded_state = sharded_state
        self.shard_fns = shard_fns
        self.gather_fns = gather_fns
        self.task.params = None
        

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
    
    def get_current_learning_rate(self):
        if isinstance(self.learning_rate_schedule, float):
            return self.learning_rate_schedule
        
        return self.learning_rate_schedule(
            int(jax.device_get(self.sharded_state.step))
        ).tolist()

    def train(self):
        termcolor.cprint(
            f"Model Contain {sum(n.size for n in jax.tree_util.tree_flatten(flax.core.unfreeze(self.sharded_state.params))[0]) / 1e9} "
            f"Billion Parameters",
            color="red", force_color=True
        )

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

        if self.args.logging_steps is None:
            self.args.logging_steps = gradient_accumulation_steps

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

        with self.mesh:
            while True:
                if args.total_steps is None and epoch >= args.total_epochs:
                    break
                
                if isinstance(self.train_dataset, IterableDataset):
                    self.train_dataset.set_epoch(epoch)
                    
                for step_in_epoch, batch in enumerate(self.train_loader):
                    global_step += 1
                    if args.total_steps is not None:
                        epoch_float = float(epoch)  
                        progress.desc = f"{description}, total {global_step / total_steps:.4f}"    
                    else:
                        epoch_float = epoch + (step_in_epoch + 1) / epoch_steps
                        progress.desc = f"{description} epoch: {epoch_float:.4f}"

                    self.sharded_state, step_output = self.sharded_train_step(self.sharded_state, batch)

                    if isinstance(step_output, tuple):
                        loss = step_output[0]
                    elif isinstance(step_output, dict):
                        loss = step_output['loss']
                    else:
                        loss = step_output
                    if jnp.isnan(loss.mean()):
                        pprint(batch)
                        print("OMG it is NaN")
                        print(self.task.tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=False))
                        return
                    step_outputs.append(step_output)

                    if (global_step + 1) % gradient_accumulation_steps == 0:
                        optimizer_step += 1

                    if (
                        global_step % self.args.logging_steps == 0
                    ):  
                        metrics = self.task.collate_train_step_outputs(step_outputs)
                        metrics = {f"train/{k}": v for k, v in metrics.items()}
                        metrics["train/optimizer_step"] = optimizer_step
                        metrics["train/progress_rate"] = global_step / total_steps
                        metrics["train/learning_rate"] = self.get_current_learning_rate()
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

                    # if global_step == 1:
                    #     profiler.print_memory_usage()
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
            if "last" == self.args.save_strategy:
                self.save_model(f"main", True)
            else:
                self.save_model(f"epoch-{epoch + 1}-last", True)

        print("training is finished!")

    def evaluate(self, epoch, global_step):
        print("start evaluation", epoch, global_step)
        step_outputs=[]
        
        for eval_step, batch in enumerate(tqdm(self.eval_loader, desc="Evaluating...")):
            step_output = self.sharded_eval_step(self.sharded_state, batch)
            step_outputs.append(step_output)

        metrics = self.task.collate_train_step_outputs(step_outputs)
        metrics = {f"eval/{k}": v for k, v in metrics.items()}
        metrics["global_step"] = global_step
        metrics["epoch"] = epoch
        self.logger.log_metric(metrics)
        step_outputs = []
    
    def save_model(self, name, is_last=False):
        print("save", name)
        if self.args.output_dir:
            if self.args.output_dir.startswith("gs://"):
                with tempfile.TemporaryDirectory() as folder_path:
                    saved_path = self.save_checkpoint_to_dir(folder_path, name)
                    self.logger.log("uploading to GCS")

                    # GCS 경로에서 버킷 이름과 경로 추출
                    gcs_path = self.args.output_dir.replace("gs://", "")
                    bucket_name, gcs_folder_path = gcs_path.split("/", 1)

                    # 함수 호출 수정
                    if name != "main":
                        upload_local_directory_to_gcs(saved_path, bucket_name, os.path.join(gcs_folder_path, name))
                    else:
                        upload_local_directory_to_gcs(saved_path, bucket_name, gcs_folder_path)
            else:
                self.save_checkpoint_to_dir(self.args.output_dir, name)
        else:
            with tempfile.TemporaryDirectory() as folder_path:
                self.save_checkpoint_to_dir(folder_path, name)
            

    def save_checkpoint_to_dir(self, folder_path, revision_name):
        state = self.sharded_state
        if self.args.use_lora:
            print("Merging LoRA parameters")
            state = state.replace(
                params=self.rapture.merge_parameters(state.params, False)
            )

        with jax.default_device(jax.devices("cpu")[0]):
            model = self.task.model
            pt_model = transformers.AutoModelForCausalLM.from_config(model.config)
            load_flax_weights_in_pytorch_model(
                pt_model, state.params
            )

            print("Saving huggingface model to local disk")
            pt_model = pt_model.bfloat16()
            if revision_name != "main":
                folder_path = os.path.join(folder_path, revision_name)


            pt_model.save_pretrained(folder_path)
            self.task.tokenizer.save_pretrained(folder_path)
            
            if self.args.push_to_hub:
                repo_id = self.args.push_to_hub_id or self.args.run_name.replace("/", "__")
                api = HfApi()
                if "/" not in repo_id:
                    name = api.whoami()['name']
                    repo_id = f"{name}/{repo_id}"

                if self.args.revision_prefix:
                    revision_name = self.args.revision_prefix + revision_name
                    
                print(f"Start uploading to huggingface model hub {repo_id}:{revision_name}")
                api.create_repo(repo_id, private=True, repo_type="model", exist_ok=True)
                if revision_name != "main":
                    api.create_branch(repo_id, branch=revision_name)
                api.upload_folder(
                    repo_id=repo_id,
                    folder_path=folder_path,
                    revision=revision_name,
                )
        return folder_path