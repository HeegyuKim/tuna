from dataclasses import dataclass
from .flax_base import FlaxLMTask, flax_tasks, FlaxLMTaskArguments
from ..chat.train_templates import find_template
from ..dpo.collator import DPOCollator
from typing import Optional, Union

import jax, flax
import jax.numpy as jnp

from fjformer import with_sharding_constraint
from fjformer.xrapture import use_implicit_args, LoraWeight
from fjformer.func.loss_func import (
    cross_entropy_loss_and_accuracy,
    SpecialLossNormalizingFactor,
    get_loss_normalizing_factor_and_weights,
    compute_weighted_cross_entropy_and_accuracy,
)
import chex


def get_batch_logps(
    logits: chex.Array,
    labels: chex.Array,
    loss_mask: chex.Array,
    average_log_prob: bool = False
) -> chex.Array:
    per_token_logps = jnp.take_along_axis(
        jax.nn.log_softmax(logits), 
        labels[..., None], 
        axis=-1
    ).squeeze(-1)

    if average_log_prob:
        return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
    else:
        return (per_token_logps * loss_mask).sum(-1)
    
def get_model_batch_logps(model, params, chosen_input, rejected_input, chosen_labels, rejected_labels, chosen_loss_mask, rejected_loss_mask):

    chosen_output = model(params=params, **chosen_input)
    chosen = chosen_output.logits[:, :-1]

    rejected_output = model(params=params, **rejected_input)
    rejected = rejected_output.logits[:, :-1]
    
    chosen_logprobs = get_batch_logps(chosen, chosen_labels, chosen_loss_mask)
    rejected_logprobs = get_batch_logps(rejected, rejected_labels, rejected_loss_mask)

    return chosen_logprobs, rejected_logprobs

def masked_mean(arr, mask):
    return (arr * mask).sum(-1) / mask.sum(-1)

def dpo_loss(
    policy_chosen_logps: chex.Array,
    policy_rejected_logps: chex.Array,
    reference_chosen_logps: chex.Array,
    reference_rejected_logps: chex.Array,
    beta: float,
):
    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps)
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps)

    losses = -jax.nn.log_sigmoid(chosen_rewards - rejected_rewards)

    return losses, chosen_rewards, rejected_rewards
    

@dataclass
class DPOTaskArguments(FlaxLMTaskArguments):
    train_template: Optional[str] = None
    dpo_beta: float = 0.1
    dpo_loss_type: str = "sigmoid"

@flax_tasks.register("dpo")
class DPOTask(FlaxLMTask):
    ARG_CLASS = DPOTaskArguments

    def init_tokenizer_collator(self):
        super().init_tokenizer_collator()
        self.train_template = find_template(self.args.train_template or self.args.model_name_or_path)(self.tokenizer)
    
    def __init__(self, args) -> None:
        super().__init__(args)
        self.beta = args.dpo_beta
        self.loss_type = args.dpo_loss_type
        self.label_pad_token_id = -100
    
    def _init_collator(self):
        self.collator = DPOCollator(
            self.tokenizer, 
            # padding=self.args.padding,
            padding_side=self.args.padding_side,
            max_length=self.args.max_length,
            # decoder_max_length=self.args.decoder_max_length,
            return_tensors="np")
        
    def encode_item(self, item):
        conversation = item["conversations"]
        chosen = self._encode_prompt_response(conversation, item["chosen"])
        rejected = self._encode_prompt_response(conversation, item["rejected"])

        return dict(
            chosen=chosen,
            rejected=rejected
        )

    def filter_item(self, item):
        trainables = sum(x >= 0 for x in item["chosen"]["labels"])
        trainables += sum(x >= 0 for x in item["rejected"]["labels"])
        return trainables > 0
    
    def _encode_prompt_response(self, conversation, response):
        concat_inputs, concat_labels = [], []
        
        for i, uttr in enumerate(conversation):
            content, _ = self.train_template.handle_utterance(uttr, i)

            input_ids = self.tokenizer.encode(content, add_special_tokens=False)
            labels = [-100] * len(input_ids)

            concat_inputs.extend(input_ids)
            concat_labels.extend(labels)

        response_id = self.tokenizer.encode(response + self.tokenizer.eos_token, add_special_tokens=False)
        concat_inputs.extend(response_id)
        concat_labels.extend(response_id)

        return self.truncate_dict({
            "input_ids": concat_inputs,
            "attention_mask": [1] * len(concat_inputs),
            "labels": concat_labels
        })
        
    
    def collate_step_outputs(self, outputs):
        loss = jnp.stack([x["loss"] for x in outputs]).mean()
        acc = jnp.stack([x["accuracy"] for x in outputs]).mean().tolist()
        chosen_rewards = jnp.stack([x["chosen_rewards"] for x in outputs]).mean().tolist()
        rejected_rewards = jnp.stack([x["rejected_rewards"] for x in outputs]).mean().tolist()
        return {"loss": loss, "accuracy": acc, "chosen_rewards": chosen_rewards, "rejected_rewards": rejected_rewards}

    @property
    def eval_metric_definitions(self):
        return {"loss": "min", "accuracy": "max", "chosen_rewards": "max", "rejected_rewards": "min"}
    
    def create_train_step(self, pjit_func, state_ps, PS):
        partition_spec = PS(("dp", "fsdp"), "sp")
        beta = self.args.dpo_beta

        model_func = use_implicit_args(self.model)
        ref_model_func = self.model

        def train_step(state, batch):
            batch = with_sharding_constraint(batch, partition_spec)

            chosen, rejected = batch["chosen"], batch["rejected"]
            chosen_labels, rejected_labels = chosen.pop("labels")[:, 1:], rejected.pop("labels")[:, 1:]

            chosen_loss_mask = chosen_labels >= 0 
            rejected_loss_mask = rejected_labels >= 0

            chosen_labels = jnp.where(chosen_loss_mask, 0, chosen_labels)
            rejected_labels = jnp.where(rejected_loss_mask, 0, rejected_labels)

            ref_chosen_logps, ref_rejected_logps = get_model_batch_logps(
                ref_model_func, state.ref_params, chosen, rejected, chosen_labels, rejected_labels,
                chosen_loss_mask, rejected_loss_mask
            )

            def calculate_loss(params, ref_chosen_logps, ref_rejected_logps, beta):
                policy_chosen_logps, policy_rejected_logps = get_model_batch_logps(
                    model_func, params, chosen, rejected, chosen_labels, rejected_labels,
                    chosen_loss_mask, rejected_loss_mask
                )
                
                losses, chosen_rewards, rejected_rewards = dpo_loss(
                    policy_chosen_logps,
                    policy_rejected_logps,
                    ref_chosen_logps, 
                    ref_rejected_logps,
                    beta
                )
                loss = losses.mean()
                accuracy = (chosen_rewards > rejected_rewards).mean()

                return loss, dict(loss=loss, accuracy=accuracy, chosen_rewards=chosen_rewards, rejected_rewards=rejected_rewards)
            
            grad_fn = jax.value_and_grad(calculate_loss, has_aux=True)
            (loss, aux_output), grad = grad_fn(state.params, ref_chosen_logps, ref_rejected_logps, beta)
            state = state.apply_gradients(grads=grad)
            return state, aux_output

        return pjit_func(
            train_step,
            in_shardings=(state_ps, PS()),
            out_shardings=(state_ps, PS()),
            donate_argnums=(0, 0),
        )

    def create_eval_step(self, pjit_func, state_ps, PS):
        partition_spec = PS(("dp", "fsdp"), "sp")
        beta = self.args.dpo_beta

        model_func = use_implicit_args(self.model)
        ref_model_func = self.model

        def eval_step(state, batch):
            batch = with_sharding_constraint(batch, partition_spec)

            chosen, rejected = batch["chosen"], batch["rejected"]
            chosen_labels, rejected_labels = chosen.pop("labels")[:, 1:], rejected.pop("labels")[:, 1:]

            chosen_loss_mask = chosen_labels >= 0 
            rejected_loss_mask = rejected_labels >= 0

            chosen_labels = jnp.where(chosen_loss_mask, 0, chosen_labels)
            rejected_labels = jnp.where(rejected_loss_mask, 0, rejected_labels)

            ref_chosen_logps, ref_rejected_logps = get_model_batch_logps(
                ref_model_func, state.ref_params or state.params, chosen, rejected, chosen_labels, rejected_labels,
                chosen_loss_mask, rejected_loss_mask
            )

            policy_chosen_logps, policy_rejected_logps = get_model_batch_logps(
                model_func, state.params, chosen, rejected, chosen_labels, rejected_labels,
                chosen_loss_mask, rejected_loss_mask
            )
            
            losses, chosen_rewards, rejected_rewards = dpo_loss(
                policy_chosen_logps,
                policy_rejected_logps,
                ref_chosen_logps, 
                ref_rejected_logps,
                beta
            )
            loss = losses.mean()
            accuracy = (chosen_rewards > rejected_rewards).mean()

            return dict(loss=loss, accuracy=accuracy, chosen_rewards=chosen_rewards, rejected_rewards=rejected_rewards)


        return pjit_func(
            eval_step,
            in_shardings=(state_ps, PS()),
            out_shardings=(PS()),
            donate_argnums=(0, 0),
        )
    