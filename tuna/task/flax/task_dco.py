from dataclasses import dataclass
from .flax_base import FlaxLMTask, flax_tasks, FlaxLMTaskArguments
from ..chat.train_templates import find_template
from ..dpo.collator import DPOCollator
from typing import Optional, Union, List, Dict, Any
from copy import deepcopy

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


class DCOCollator(DPOCollator):

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        chosens = [x["chosen"] for x in features]
        rejections = [x["rejected"] for x in features]
        chosen_declined = [x["chosen_declined"] for x in features]
        rejected_improved = [x["rejected_improved"] for x in features]

        chosens, rejections = self.pad(chosens), self.pad(rejections)
        chosen_declined, rejected_improved = self.pad(chosen_declined), self.pad(rejected_improved)

        return {
            "chosen": chosens,
            "rejected": rejections,
            "chosen_declined": chosen_declined,
            "rejected_improved": rejected_improved
        }





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
    
def get_model_batch_logps(model, params, inputs, labels, loss_mask):
    output = model(params=params, **inputs)
    logits = output.logits[:, :-1]
    logprobs = get_batch_logps(logits, labels, loss_mask)

    return logprobs

def get_model_batch_logps_pair(model, params, chosen_input, rejected_input, chosen_labels, rejected_labels, chosen_loss_mask, rejected_loss_mask):

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
    
def dpo_loss_v2(
    policy_chosen_logps: chex.Array,
    policy_rejected_logps: chex.Array,
    reference_chosen_logps: chex.Array,
    reference_rejected_logps: chex.Array,
    policy_rejected_improved_logps: chex.Array,
    reference_chosen_declined_logps: chex.Array,
    beta: float,
    gamma: float
):
    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps)
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps)
    rejected_improved_rewards = beta * (policy_rejected_improved_logps - reference_chosen_declined_logps)

    losses = -jax.nn.log_sigmoid(gamma * chosen_rewards + (1 - gamma) * rejected_improved_rewards - rejected_rewards)

    return losses, chosen_rewards, rejected_rewards, rejected_improved_rewards
    
def dpo_loss_v3(
    policy_chosen_logps: chex.Array,
    policy_rejected_logps: chex.Array,
    reference_chosen_logps: chex.Array,
    reference_rejected_logps: chex.Array,
    policy_chosen_decliend_logps: chex.Array,
    policy_rejected_improved_logps: chex.Array,
    reference_chosen_decliend_logps: chex.Array,
    reference_rejected_improved_logps: chex.Array,
    beta: float,
    gamma: float
):
    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps)
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps)
    chosen_declined_rewards = gamma * beta * (policy_chosen_decliend_logps - reference_chosen_decliend_logps)
    rejected_improved_rewards = gamma * beta * (policy_rejected_improved_logps - reference_rejected_improved_logps)

    losses = -jax.nn.log_sigmoid(chosen_rewards + rejected_improved_rewards - rejected_rewards - chosen_declined_rewards)

    return losses, chosen_rewards, rejected_rewards, chosen_declined_rewards, rejected_improved_rewards
    

@dataclass
class DCOTaskArguments(FlaxLMTaskArguments):
    train_template: Optional[str] = None
    dpo_beta: float = 0.1
    dco_gamma: float = 0.1
    dpo_loss_type: str = "sigmoid"

@flax_tasks.register("dco")
class DCOTask(FlaxLMTask):
    ARG_CLASS = DCOTaskArguments

    def init_tokenizer_collator(self):
        super().init_tokenizer_collator()
        self.train_template = find_template(self.args.train_template or self.args.model_name_or_path)(self.tokenizer)
    
    def __init__(self, args) -> None:
        super().__init__(args)
        self.beta = args.dpo_beta
        self.loss_type = args.dpo_loss_type
        self.label_pad_token_id = -100
    
    def _init_collator(self):
        self.collator = DCOCollator(
            self.tokenizer, 
            # padding=self.args.padding,
            padding_side=self.args.padding_side,
            max_length=self.args.max_length,
            # decoder_max_length=self.args.decoder_max_length,
            return_tensors="np")
        
    def encode_item(self, item):
        conversation = deepcopy(item["conversations"])
        chosen = self._encode_prompt_response(conversation, item["chosen"])
        rejected = self._encode_prompt_response(conversation, item["rejected"])

        chosen_declined = self._encode_critique(conversation, item["chosen"], item["chosen_critique"], item["rejected"])
        rejected_improved = self._encode_critique(conversation, item["rejected"], item["rejected_critique"], item["chosen"])

        return dict(
            chosen=chosen,
            rejected=rejected,
            chosen_declined=chosen_declined,
            rejected_improved=rejected_improved
        )

    def filter_item(self, item):
        trainables = sum(x >= 0 for x in item["chosen"]["labels"])
        trainables += sum(x >= 0 for x in item["rejected"]["labels"])
        trainables += sum(x >= 0 for x in item["chosen_declined"]["labels"])
        trainables += sum(x >= 0 for x in item["rejected_improved"]["labels"])
        return trainables > 0

    def _encode_critique(self, conversation, response, critique, revision):
        return self._encode_prompt_response(
            conversation + [
                {
                    "role": "assistant",
                    "content": response
                },
                {
                    "role": "user",
                    "content": critique
                }
            ],
            revision
        )
    
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
        keys = list(outputs[0].keys())
        return {k: jnp.stack([x[k] for x in outputs]).mean().tolist() for k in keys}

    @property
    def eval_metric_definitions(self):
        return {"loss": "min", "accuracy": "max", "chosen_rewards": "max", "rejected_rewards": "min"}
    
    def create_train_step(self, pjit_func, state_ps, PS):
        partition_spec = PS(("dp", "fsdp"), "sp")
        beta = self.args.dpo_beta
        gamma = self.args.dco_gamma

        model_func = use_implicit_args(self.model)
        ref_model_func = self.model

        def train_step(state, batch):
            batch = with_sharding_constraint(batch, partition_spec)

            # Preference
            chosen, rejected = batch["chosen"], batch["rejected"]
            chosen_labels, rejected_labels = chosen.pop("labels")[:, 1:], rejected.pop("labels")[:, 1:]

            chosen_loss_mask = chosen_labels >= 0 
            rejected_loss_mask = rejected_labels >= 0

            chosen_labels = jnp.where(chosen_loss_mask, chosen_labels, 0)
            rejected_labels = jnp.where(rejected_loss_mask, rejected_labels, 0)

            # Critique
            chosen_declined, rejected_improved = batch["chosen_declined"], batch["rejected_improved"]
            chosen_declined_labels, rejected_improved_labels = chosen_declined.pop("labels")[:, 1:], rejected_improved.pop("labels")[:, 1:]

            chosen_declined_loss_mask = chosen_declined_labels >= 0
            rejected_improved_loss_mask = rejected_improved_labels >= 0

            chosen_declined_labels = jnp.where(chosen_declined_loss_mask, chosen_declined_labels, 0)
            rejected_improved_labels = jnp.where(rejected_improved_loss_mask, rejected_improved_labels, 0)


            ref_chosen_logps, ref_rejected_logps = get_model_batch_logps_pair(
                ref_model_func, state.ref_params, chosen, rejected, chosen_labels, rejected_labels,
                chosen_loss_mask, rejected_loss_mask
            )
            ref_chosen_logps_declined, ref_rejected_logps_improved = get_model_batch_logps_pair(
                ref_model_func, state.ref_params, chosen_declined, rejected_improved, chosen_declined_labels, rejected_improved_labels,
                chosen_declined_loss_mask, rejected_improved_loss_mask
            )

            def calculate_loss(params, ref_chosen_logps, ref_rejected_logps, beta):
                policy_chosen_logps, policy_rejected_logps = get_model_batch_logps_pair(
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

                policy_chosen_declined_logps, policy_rejected_improved_logps = get_model_batch_logps_pair(
                    model_func, params, chosen_declined, rejected_improved, chosen_declined_labels, rejected_improved_labels,
                    chosen_declined_loss_mask, rejected_improved_loss_mask
                )

                losses_declined, rejected_rewards_improved, chosen_rewards_declined = dpo_loss(
                    policy_rejected_improved_logps,
                    policy_chosen_declined_logps,
                    ref_rejected_logps_improved,
                    ref_chosen_logps_declined,
                    beta
                )

                losses, losses_declined = losses.mean(), losses_declined.mean()
                loss = losses + gamma * losses_declined
                accuracy = (chosen_rewards > rejected_rewards).mean()

                return loss, dict(loss=loss, losses_decline=losses_declined,
                                  accuracy=accuracy, chosen_rewards=chosen_rewards, rejected_rewards=rejected_rewards,
                                  chosen_rewards_declined=chosen_rewards_declined, rejected_rewards_improved=rejected_rewards_improved)
            
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

            chosen_labels = jnp.where(chosen_loss_mask, chosen_labels, 0)
            rejected_labels = jnp.where(rejected_loss_mask, rejected_labels, 0)

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
    


@flax_tasks.register("dco-v2")
class DCOTaskV2(DCOTask):
    def create_train_step(self, pjit_func, state_ps, PS):
        partition_spec = PS(("dp", "fsdp"), "sp")
        beta = self.args.dpo_beta
        gamma = self.args.dco_gamma

        model_func = use_implicit_args(self.model)
        ref_model_func = self.model

        def train_step(state, batch):
            batch = with_sharding_constraint(batch, partition_spec)

            # Preference
            chosen, rejected = batch["chosen"], batch["rejected"]
            chosen_labels, rejected_labels = chosen.pop("labels")[:, 1:], rejected.pop("labels")[:, 1:]

            chosen_loss_mask = chosen_labels >= 0 
            rejected_loss_mask = rejected_labels >= 0

            chosen_labels = jnp.where(chosen_loss_mask, chosen_labels, 0)
            rejected_labels = jnp.where(rejected_loss_mask, rejected_labels, 0)

            # Critique
            chosen_declined, rejected_improved = batch["chosen_declined"], batch["rejected_improved"]
            chosen_declined_labels, rejected_improved_labels = chosen_declined.pop("labels")[:, 1:], rejected_improved.pop("labels")[:, 1:]

            chosen_declined_loss_mask = chosen_declined_labels >= 0
            rejected_improved_loss_mask = rejected_improved_labels >= 0

            chosen_declined_labels = jnp.where(chosen_declined_loss_mask, chosen_declined_labels, 0)
            rejected_improved_labels = jnp.where(rejected_improved_loss_mask, rejected_improved_labels, 0)


            ref_chosen_logps, ref_rejected_logps = get_model_batch_logps_pair(
                ref_model_func, state.ref_params, chosen, rejected, chosen_labels, rejected_labels,
                chosen_loss_mask, rejected_loss_mask
            )
            ref_rejected_logps_improved = get_model_batch_logps(
                ref_model_func, state.ref_params, 
                rejected_improved, 
                rejected_improved_labels,
                rejected_improved_loss_mask
            )

            def calculate_loss(params, ref_chosen_logps, ref_rejected_logps, beta):
                policy_chosen_logps, policy_rejected_logps = get_model_batch_logps_pair(
                    model_func, params, chosen, rejected, chosen_labels, rejected_labels,
                    chosen_loss_mask, rejected_loss_mask
                )
                
                policy_rejected_logps_improved = get_model_batch_logps(
                    model_func, state.params, 
                    rejected_improved, 
                    rejected_improved_labels,
                    rejected_improved_loss_mask
                )

                losses, chosen_rewards, rejected_rewards, rejected_rewards_improved = dpo_loss_v2(
                    policy_chosen_logps,
                    policy_rejected_logps,
                    ref_chosen_logps, 
                    ref_rejected_logps,
                    policy_rejected_logps_improved,
                    ref_rejected_logps_improved,
                    beta,
                    gamma
                )

                loss = losses.mean()
                accuracy = (chosen_rewards > rejected_rewards).mean()

                return loss, dict(loss=loss,
                                  accuracy=accuracy, chosen_rewards=chosen_rewards, rejected_rewards=rejected_rewards,
                                  rejected_rewards_improved=rejected_rewards_improved)
            
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

@flax_tasks.register("dco-v3")
class DCOTaskV3(DCOTask):
    def create_train_step(self, pjit_func, state_ps, PS):
        partition_spec = PS(("dp", "fsdp"), "sp")
        beta = self.args.dpo_beta
        gamma = self.args.dco_gamma

        model_func = use_implicit_args(self.model)
        ref_model_func = self.model

        def train_step(state, batch):
            batch = with_sharding_constraint(batch, partition_spec)

            # Preference
            chosen, rejected = batch["chosen"], batch["rejected"]
            chosen_labels, rejected_labels = chosen.pop("labels")[:, 1:], rejected.pop("labels")[:, 1:]

            chosen_loss_mask = chosen_labels >= 0 
            rejected_loss_mask = rejected_labels >= 0

            chosen_labels = jnp.where(chosen_loss_mask, chosen_labels, 0)
            rejected_labels = jnp.where(rejected_loss_mask, rejected_labels, 0)

            # Critique
            chosen_declined, rejected_improved = batch["chosen_declined"], batch["rejected_improved"]
            chosen_declined_labels, rejected_improved_labels = chosen_declined.pop("labels")[:, 1:], rejected_improved.pop("labels")[:, 1:]

            chosen_declined_loss_mask = chosen_declined_labels >= 0
            rejected_improved_loss_mask = rejected_improved_labels >= 0

            chosen_declined_labels = jnp.where(chosen_declined_loss_mask, chosen_declined_labels, 0)
            rejected_improved_labels = jnp.where(rejected_improved_loss_mask, rejected_improved_labels, 0)


            ref_chosen_logps, ref_rejected_logps = get_model_batch_logps_pair(
                ref_model_func, state.ref_params, chosen, rejected, chosen_labels, rejected_labels,
                chosen_loss_mask, rejected_loss_mask
            )
            ref_chosen_logps_declined, ref_rejected_logps_improved = get_model_batch_logps_pair(
                ref_model_func, state.ref_params, chosen_declined, rejected_improved, chosen_declined_labels, rejected_improved_labels,
                chosen_declined_loss_mask, rejected_improved_loss_mask
            )

            def calculate_loss(params, ref_chosen_logps, ref_rejected_logps, beta):
                policy_chosen_logps, policy_rejected_logps = get_model_batch_logps_pair(
                    model_func, params, chosen, rejected, chosen_labels, rejected_labels,
                    chosen_loss_mask, rejected_loss_mask
                )

                policy_chosen_declined_logps, policy_rejected_improved_logps = get_model_batch_logps_pair(
                    model_func, params, chosen_declined, rejected_improved, chosen_declined_labels, rejected_improved_labels,
                    chosen_declined_loss_mask, rejected_improved_loss_mask
                )

                losses, chosen_rewards, rejected_rewards, chosen_rewards_declined, rejected_rewards_improved = dpo_loss_v3(
                    policy_chosen_logps,
                    policy_rejected_logps,
                    ref_chosen_logps, 
                    ref_rejected_logps,
                    policy_chosen_declined_logps,
                    policy_rejected_improved_logps,
                    ref_chosen_logps_declined,
                    ref_rejected_logps_improved,
                    beta,
                    gamma
                )

                loss = losses.mean()
                accuracy = (chosen_rewards > rejected_rewards).mean()

                return loss, dict(loss=loss, accuracy=accuracy, 
                                  chosen_rewards=chosen_rewards, rejected_rewards=rejected_rewards,
                                  chosen_rewards_declined=chosen_rewards_declined, rejected_rewards_improved=rejected_rewards_improved)
            
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


@flax_tasks.register("dco-v4")
class DCOTaskV4(DCOTask):
    def encode_item(self, item):
        conversation = deepcopy(item["conversations"])
        chosen = self._encode_prompt_response(conversation, item["chosen"])
        rejected = self._encode_prompt_response(conversation, item["rejected"])

        chosen_declined = self._encode_critique(conversation, item["rejected"], item["chosen_critique"], item["chosen"])
        rejected_improved = self._encode_critique(conversation, item["rejected"], item["rejected_critique"], item["chosen"])

        return dict(
            chosen=chosen,
            rejected=rejected,
            chosen_declined=chosen_declined,
            rejected_improved=rejected_improved
        )