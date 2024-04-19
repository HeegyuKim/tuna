from typing import List, Dict, Any, Optional, Iterable, Union
from dataclasses import dataclass
from ..base import BaseTask, TaskArguments, GenerativeLanguageModelCollator

import jax
import jax.numpy as jnp
import flax
from jax.sharding import PartitionSpec

from fjformer import with_sharding_constraint
from fjformer.func.loss_func import (
    cross_entropy_loss_and_accuracy,
    SpecialLossNormalizingFactor,
    get_loss_normalizing_factor_and_weights,
    compute_weighted_cross_entropy_and_accuracy,
)

import transformers

from ...common import Registry
from ..dataset import NUM_PROC

flax_tasks = Registry("flax-tasks")


@dataclass
class FlaxTaskArguments(TaskArguments):
    model_name_or_path: str = ""

class FlaxTask(BaseTask):
    ARG_CLASS = FlaxTaskArguments

    def __init__(self, args) -> None:
        self.args = args

    def init_model(self, dtype):
        pass
        
    def init_weights(self):
        return self.model.init_weights(
            jax.random.PRNGKey(0), 
            (1, self.args.max_length),
            )

    def create_train_step(self, pjit_func, state_ps, PS):
        def train_step(state, batch):
            pass

        return pjit_func(
            train_step,
            in_shardings=(state_ps, PS()),
            out_shardings=(PS()),
            donate_argnums=(0, 0),
        )

    def create_eval_step(self, pjit_func, state_ps, PS):
        def eval_step(state, batch):
            pass

        return pjit_func(
            eval_step,
            in_shardings=(state_ps, PS()),
            out_shardings=(PS()),
            donate_argnums=(0, 0),
        )
    

@dataclass
class FlaxLMTaskArguments(FlaxTaskArguments):
    label_smoothing_factor: float = 0.0
    z_loss: float = 0.0

@flax_tasks("lm")
class FlaxLMTask(FlaxTask):
    ARG_CLASS = FlaxLMTaskArguments

    def __init__(self, args) -> None:
        super().__init__(args)
        self.init_tokenizer_collator()

    def init_tokenizer_collator(self):
        model_name = self.args.model_name_or_path
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            print("Set pad token to eos token")
        self._init_collator()

    def init_model(self, dtype):
        input_shape = (1, self.args.max_length)
        model_name = self.args.model_name_or_path
        with jax.default_device(jax.devices('cpu')[0]):
            config = transformers.AutoConfig.from_pretrained(model_name)
            flax_model = transformers.FlaxAutoModelForCausalLM.from_config(
                config,
                _do_init=True,
                dtype=dtype,
                # param_dtype=param_dtype,
                # precision=precision,
                input_shape=input_shape
                )

            pt_model = transformers.AutoModelForCausalLM.from_pretrained(model_name)
            pt_state_dict = pt_model.state_dict()

            print("Converting Pytorch parameters to Flax")
            params = transformers.modeling_flax_pytorch_utils.convert_pytorch_state_dict_to_flax(pt_state_dict, flax_model)

            if pt_model.config.tie_word_embeddings:
                print("Tie word embeddings, delete lm_head from flax model!")
                params.pop("lm_head")

            del pt_state_dict
            del pt_model
            import gc
            gc.collect()

        self.model, self.params = flax_model, flax.core.freeze(params)

    def _init_collator(self):
        self.collator = GenerativeLanguageModelCollator(
            self.tokenizer, 
            padding=self.args.padding,
            padding_side=self.args.padding_side,
            max_length=self.args.max_length,
            decoder_max_length=self.args.decoder_max_length,
            return_tensors="np")
        
    def encode_datasets(self, datasets: Dict):
        datasets = super().encode_datasets(datasets)
        if self.args.packing:
            cols = datasets["train"].column_names
            if cols:
                if "input_ids" in cols:
                    cols.remove("input_ids")
                if "labels" in cols:
                    cols.remove("labels")
                for k in datasets:
                    datasets[k] = datasets[k].map(self._pack, load_from_cache_file=False, batched=True, remove_columns=cols, desc="Packing", num_proc=NUM_PROC)
            else: # iterable dataset
                for k in datasets:
                    datasets[k] = datasets[k].map(self._pack, batched=True)
            
        return datasets

    def _pack(self, items):
        outputs = dict(
            input_ids=[],
            attention_mask=[],
            labels=[]
        )
        accum_len = 0

        batch_len = self.args.max_length
        all_input_ids = items["input_ids"]
        all_attention_mask = items.get("attention_mask")
        all_labels = items["labels"]

        batch_ids, batch_mask, batch_labels = [], [], []

        for ids, mask, labels in zip(all_input_ids, all_attention_mask, all_labels):
            accum_len += len(ids)

            batch_ids.extend(ids)
            if all_attention_mask is not None:
                batch_mask.extend(mask)
            batch_labels.extend(labels)

            while accum_len > batch_len:
                outputs["input_ids"].append(batch_ids[:batch_len])
                if all_attention_mask is not None:
                    outputs["attention_mask"].append(batch_mask[:batch_len])
                outputs["labels"].append(batch_labels[1:batch_len + 1])

                batch_ids, batch_labels = batch_ids[batch_len:], batch_labels[batch_len:]
                if all_attention_mask is not None:
                    batch_mask = batch_mask[batch_len:]
                accum_len -= batch_len
        
        if all_attention_mask is None:
            outputs.pop("attention_mask")
        
        return outputs
    
    def collate_batch(self, batch):
        return self.collator(batch)
    
    def collate_step_outputs(self, outputs):
        loss = jnp.stack([x["loss"] for x in outputs]).mean()
        acc = jnp.stack([x["accuracy"] for x in outputs]).mean().tolist()
        return {"loss": loss, "accuracy": acc}
    
    def collate_train_step_outputs(self, outputs):
        return self.collate_step_outputs(outputs)

    def collate_eval_step_outputs(self, outputs):
        return self.collate_step_outputs(outputs)

    @property
    def eval_metric_definitions(self):
        return {"loss": "min", "accuracy": "max"}
    
    def create_train_step(self, pjit_func, state_ps, PS):
        partition_spec = PartitionSpec(("dp", "fsdp"), "sp")
        label_smoothing_factor = self.args.label_smoothing_factor
        z_loss = self.args.z_loss

        def train_step(state, batch):
            batch = with_sharding_constraint(batch, partition_spec)

            def calculate_loss(params):
                labels = batch.pop("labels")[:, 1:]
                logits = self.model(params=params, **batch, return_dict=True).logits
                loss_normalizing_factor = (
                    SpecialLossNormalizingFactor.NUM_REAL_TARGET_TOKENS
                )
                # loss_weights is 1 unless the label is <= 0 or the attention mask is 0
                loss_weights = jnp.where(
                    (batch["attention_mask"][:, 1:] != 0) & (labels >= 0), 1, 0
                )
                lnf, weights = get_loss_normalizing_factor_and_weights(
                    loss_normalizing_factor,
                    {
                        "decoder_target_tokens": labels,
                        "decoder_loss_weights": loss_weights,
                    },
                )
                (
                    loss,
                    z_loss_computed,
                    weight_sum,
                    accuracy,
                ) = compute_weighted_cross_entropy_and_accuracy(
                    logits=logits[:, :-1, :],
                    targets=labels,
                    weights=weights,
                    label_smoothing=label_smoothing_factor,
                    z_loss=z_loss,
                    loss_normalizing_factor=lnf,
                )
                return loss, accuracy
            
            grad_fn = jax.value_and_grad(calculate_loss, has_aux=True)
            (loss__, accuracy__), grad = grad_fn(state.params)
            state = state.apply_gradients(grads=grad)
            return state, {"loss": loss__, "accuracy": accuracy__}

        return pjit_func(
            train_step,
            in_shardings=(state_ps, PS()),
            out_shardings=(state_ps, PS()),
            donate_argnums=(0, 0),
        )

    def create_eval_step(self, pjit_func, state_ps, PS):
        partition_spec = PartitionSpec(("dp", "fsdp"), "sp")
        label_smoothing_factor = self.args.label_smoothing_factor
        z_loss = self.args.z_loss

        def eval_step(state, batch):
            batch = with_sharding_constraint(batch, partition_spec)
            labels = batch.pop("labels")[:, 1:]
            logits = self.model(params=state.params, **batch, return_dict=True).logits
            loss_normalizing_factor = (
                SpecialLossNormalizingFactor.NUM_REAL_TARGET_TOKENS
            )
            # loss_weights is 1 unless the label is <= 0 or the attention mask is 0
            loss_weights = jnp.where(
                (batch["attention_mask"][:, 1:] != 0) & (labels >= 0), 1, 0
            )
            lnf, weights = get_loss_normalizing_factor_and_weights(
                loss_normalizing_factor,
                {
                    "decoder_target_tokens": labels,
                    "decoder_loss_weights": loss_weights,
                },
            )
            (
                loss,
                z_loss_computed,
                weight_sum,
                accuracy,
            ) = compute_weighted_cross_entropy_and_accuracy(
                logits=logits[:, :-1, :],
                targets=labels,
                weights=weights,
                label_smoothing=label_smoothing_factor,
                z_loss=z_loss,
                loss_normalizing_factor=lnf,
            )
            return {"loss": loss, "accuracy": accuracy}

        return pjit_func(
            eval_step,
            in_shardings=(state_ps, PS()),
            out_shardings=(PS()),
            donate_argnums=(0, 0),
        )
    